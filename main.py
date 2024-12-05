import os
from pathlib import Path
import logging
import pickle

import torch
from torch.utils.data import DataLoader

from utils.arguments import parse_args, get_model_config
from utils.datasets import baseDataset, QuadruplesDataset
from utils.general import increment_path
from utils.logger import set_logger
from utils.trainer import Trainer, ACTrainer
from utils.tester import Tester
from utils.torch_utils import init_seeds
from utils.torch_optimizers import Lion, AdamNorm
# import thirdparty.schedule_free as sf

from models.agent import Agent, AgentAC, AgentCT
from models.environment import Env
from models.episode import Episode, EpisodeAC
from models.policyGradient import PG, ACPG
from models.dirichlet import Dirichlet


def main(args):
    init_seeds(args.random_seed)
    print(f"Random seed set as {args.random_seed}")

    ####################### Set Logger#################################
    curr_dir = Path(__file__).resolve().parent
    
    dataname = args.data_path.split("/")[-1]
    
    name = Path(
        args.save_path + os.sep + args.name + "_" + dataname + "_" + str(args.model_id)
    )

    if not os.path.exists(curr_dir / name):
        os.makedirs(curr_dir / name)
        args.save_path = str(curr_dir / name)
    else:
        args.save_path = str(
            increment_path(curr_dir / name, exist_ok=args.exist_ok, mkdir=True)
        )

    if args.cuda and torch.cuda.is_available():
        args.cuda = True
    else:
        args.cuda = False
    set_logger(args)

    ####################### Create DataLoader#################################
    train_path = os.path.join(args.data_path, "train.txt")
    test_path = os.path.join(args.data_path, "test.txt")
    stat_path = os.path.join(args.data_path, "stat.txt")
    valid_path = os.path.join(args.data_path, "valid.txt")

    baseData = baseDataset(train_path, test_path, stat_path, valid_path)

    trainDataset = QuadruplesDataset(baseData.trainQuadruples, baseData.num_r)
    train_dataloader = DataLoader(
        trainDataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    validDataset = QuadruplesDataset(baseData.validQuadruples, baseData.num_r)
    valid_dataloader = DataLoader(
        validDataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    testDataset = QuadruplesDataset(baseData.testQuadruples, baseData.num_r)
    test_dataloader = DataLoader(
        testDataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    ###################### Creat the agent and the environment###########################
    config = get_model_config(args, baseData.num_e, baseData.num_r)
    logging.info(config)
    logging.info(args)

    # creat the agent
    if args.ac:
        agent = AgentAC(config)
    elif args.use_confidence:
        agent = AgentCT(config)
    else:
        agent = Agent(config)

    # creat the environment
    state_actions_path = os.path.join(args.data_path, args.state_actions_path)
    if not os.path.exists(state_actions_path):
        state_action_space = None
    else:
        state_action_space = pickle.load(
            open(os.path.join(args.data_path, args.state_actions_path), "rb")
        )
        
    env = Env(baseData.allQuadruples, config, state_action_space)

    # Create episode controller
    if args.ac:
        episode = EpisodeAC(env, agent, config)
    else:
        episode = Episode(env, agent, config)
        
    if args.pretrain:
        episode.load_pretrain(args.data_path + '/pretrain/entity_embedding.npy',
                              args.data_path + '/pretrain/relation_embedding.npy')
        
    if args.cuda:
        episode = episode.cuda()

    # Policy Gradient
    if args.ac:
        pg = ACPG(config)
    else:
        pg = PG(config)  

    if args.optim == "Adam":
        # optimizer = torch.optim.Adam(episode.parameters(), lr=args.lr, weight_decay=0.00001)
        optimizer = torch.optim.Adam(
            episode.parameters(),
            lr=args.lr,
            betas=(args.beta1, 0.999),
            weight_decay=args.weight_decay,
        )
         
         
    elif args.optim == "RMSProp":
        optimizer = torch.optim.RMSprop(
            episode.parameters(), lr=args.lr, momentum=args.beta1
        )
    elif args.optim == "Adadelta":
        optimizer = torch.optim.Adadelta(
            episode.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optim == "Lion":
        optimizer = Lion(
            episode.parameters(),
            lr=args.lr,
            betas=(args.beta1, 0.999),
            weight_decay=args.weight_decay,
        )
    elif args.optim == "Adanorm":
        optimizer = AdamNorm(
            episode.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {args.optim} not implemented.")

    # Load the model parameters
    if os.path.isfile(args.load_model_path):
        params = torch.load(args.load_model_path)
        episode.load_state_dict(params["model_state_dict"])
        optimizer.load_state_dict(params["optimizer_state_dict"])
        logging.info("Load pretrain model: {}".format(args.load_model_path))

    ###################### Training and Testing ###########################
    if args.reward_shaping:
        alphas = pickle.load(open(os.path.join(args.data_path, args.alphas_pkl), "rb"))
        distributions = Dirichlet(alphas, args.k)
    else:
        distributions = None

    if args.ac:
        trainer = ACTrainer(episode, pg, optimizer, args, distributions)
    else:
        trainer = Trainer(episode, pg, optimizer, args, distributions)
        
    tester = Tester(episode, args, baseData.train_entities, baseData.RelEntCooccurrence)

    if args.do_train:
        logging.info("Start Training......")
        for i in range(args.max_epochs):
            loss, reward = trainer.train_epoch(train_dataloader, trainDataset.__len__())
            logging.info(
                "Epoch {}/{} Loss: {}, reward: {}".format(
                    i, args.max_epochs, loss, reward
                )
            )

            if i % args.save_epoch == 0 and i != 0:
                trainer.save_model(args.save_path + "/checkpoint_{}.pth".format(i))
                logging.info("Save Model in {}".format(args.save_path))

            if i % args.valid_epoch == 0 and i != 0:
                logging.info("Start Val......")
                metrics = tester.test(
                    valid_dataloader,
                    validDataset.__len__(),
                    baseData.skip_dict,
                    config["num_ent"],
                )
                for mode in metrics.keys():
                    logging.info("{} at epoch {}: {}".format(mode, i, metrics[mode]))

        trainer.save_model()
        logging.info("Save Model in {}".format(args.save_path))

    if args.do_test:
        args.save_path = args.load_model_path
        logging.info("Start Testing......")
        metrics = tester.test(
            test_dataloader,
            testDataset.__len__(),
            baseData.skip_dict,
            config["num_ent"],
        )
        for mode in metrics.keys():
            logging.info("Test {} : {}".format(mode, metrics[mode]))


if __name__ == "__main__":
    args = parse_args()
    main(args)
