import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Training and Testing Temporal Knowledge Graph Forecasting Models",
        usage="main.py [<args>] [-h | --help]",
    )

    parser.add_argument("--name", type=str, default="TimeTraveler", help="Model name.")
    parser.add_argument("--model_id", type=int, default=0)
    parser.add_argument("--cuda", action="store_true", help="whether to use GPU or not.")
    parser.add_argument("--data-path", type=str, default="data/ICEWS14", help="Path to data.")
    parser.add_argument("--do-train", action="store_true", help="whether to train.")
    parser.add_argument("--do-test", action="store_true", help="whether to test.")
    parser.add_argument("--save-path", default="./runs", type=str, help="log and model save path.")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment.")
    parser.add_argument("--load-model-path", default="logs", type=str, help="trained model checkpoint path.",)
    parser.add_argument('--history-encoder', default="lstm", type=str, help='how to learn path embeddings')
    # parser.add_argument("--adaptive-sample", action="store_true", help='whether to use time-adaptive sample')
    # parser.add_argument('--random-sample', action="store_true", help='whether to use random sample')
    # parser.add_argument('--timeaware-sample', action="store_true", help='whether to use random sample')
    parser.add_argument('--random-seed', default=2024, type=int, help='init random seed')

    # Train Params
    parser.add_argument("--batch-size", default=512, type=int, help="training batch size.")
    parser.add_argument("--max-epochs", default=400, type=int, help="max training epochs.")
    parser.add_argument("--num-workers", default=8, type=int, help="workers number used for dataloader.")
    parser.add_argument("--valid-epoch", default=30, type=int, help="validation frequency.")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate.")
    parser.add_argument("--save-epoch", default=30, type=int, help="model saving frequency.")
    parser.add_argument("--clip-gradient", default=10.0, type=float, help="for gradient crop.")
    parser.add_argument('--pretrain', action='store_true', help='whether to use pretrain')

    # Test Params
    parser.add_argument("--test-batch-size", default=1, type=int, help="test batch size, it needs to be set to 1 when using IM module.",)
    parser.add_argument("--beam-size", default=100, type=int, help="the beam number of the beam search.")
    parser.add_argument("--test-inductive", action="store_true", help="whether to verify inductive inference performance.",)
    parser.add_argument("--IM", action="store_true", help="whether to use IM module.")
    parser.add_argument("--mu", default=0.1, type=float, help="the hyperparameter of IM module.")

    # Agent Params
    parser.add_argument("--ent-dim", default=100, type=int, help="Embedding dimension of the entities.")
    parser.add_argument("--rel-dim", default=100, type=int, help="Embedding dimension of the relations.")
    parser.add_argument("--state-dim", default=100, type=int, help="dimension of the LSTM hidden state.")
    parser.add_argument("--hidden-dim", default=100, type=int, help="dimension of the MLP hidden layer.")
    parser.add_argument("--time-dim", default=20, type=int, help="Embedding dimension of the timestamps.")
    parser.add_argument("--entities-embeds-method", default="dynamic", type=str, help="Representation method of the entities, dynamic or static",)
    parser.add_argument("--use-confidence", "-conf", action="store_true", help='Whether to calculate confidence.',)
    parser.add_argument("--confidence-mode", "-conf-mode", default="lowfer", type=str, help='Confidence mode.',)
    parser.add_argument("--lowfer-k", default=30, type=int, help="K for LowFER factorization model.")
    parser.add_argument("--droprate", default=0.2, type=float, help="Dropout rate of policy MLP.",)

    # Environment Params
    parser.add_argument("--state-actions-path", default="state_actions_space.pkl", type=str, help="The file stores preprocessed candidate action array.",)

    # Episode Params
    parser.add_argument("--path-length", default=3, type=int, help="The agent search path length.")
    parser.add_argument("--max-action-num", default=50, type=int, help="The max candidate actions number. 50 for ICEWS14, ICEWS18, 60 for WIKI and 30 for YAGO.",)
    parser.add_argument("--path_weight", default=1, type=float, help="The weight for current path length which is taken.")

    # Policy Gradient Params
    parser.add_argument("--Lambda", default=0.0, type=float, help="The update rate of baseline.")
    parser.add_argument("--Gamma", default=0.95, type=float, help="The discount factor of Bellman Eq.")
    parser.add_argument("--Ita", default=0.01, type=float, help="The regular proportionality constant.")
    parser.add_argument("--Zita",default=0.9, type=float, help="The attenuation factor of entropy regular term.",)
    parser.add_argument("--policy-network", default="MLP", type=str, help="Policy network type. Default: MLP")
    parser.add_argument("--spline-type", default="B-Spline", type=str, help="Spline method for KAN policy network.")
    

    # Reward shaping params
    parser.add_argument("--reward-shaping", action="store_true", help="whether to use reward shaping.")
    parser.add_argument("--time-span", default=24, type=int, help="24 for ICEWS, 1 for WIKI and YAGO.")
    parser.add_argument("--alphas-pkl", default="dirchlet_alphas.pkl", type=str, help="The file storing the alpha parameters of the Dirichlet distribution.",)
    parser.add_argument("--k", default=300, type=int, help="Statistics recent K historical snapshots.")
    parser.add_argument("--multi-reward", action="store_true", help="Use multi-reward function")
    
    parser.add_argument("--rule", action="store_true", help="Use mined rule.")
    parser.add_argument("--rules-path", default="ER.json", type=str, help="Rule json file path.")
    parser.add_argument('--r-weight', default=0.05, type=float, help='Base reward for rule based reward')
    parser.add_argument("--alpha_r", default=0.6, type=int, help="Control factor for rule-based reward")
    
    parser.add_argument("--freq", action="store_true", help="Use frequency-based reward")
    parser.add_argument("--alpha_f", default=0.2, type=int, help="Control factor for frequency-based reward")
    
    parser.add_argument("--eff", action="store_true", help="Use efficiency-based reward")
    parser.add_argument("--alpha_e", default=0.3, type=int, help="Control factor for efficiency-based reward")
    
    # Optimization
    parser.add_argument("--optim", default="Adam", type=str, help="Optimization method.")
    parser.add_argument("--beta1", default=0.937, type=float, help="Momentum for optimization method.") # momentum: 0.937  # SGD momentum/Adam beta1
    parser.add_argument("--weight-decay", default=0.00001, type=float, help="Weight decay for optimization method.")  # optimizer weight decay 5e-4
    
    return parser.parse_args(args)

def get_model_config(args, num_ent, num_rel):
    config = {
        "data_path": args.data_path,
        "cuda": args.cuda,  # whether to use GPU or not.
        "batch_size": args.batch_size,  # training batch size.
        "num_ent": num_ent,  # number of entities
        "num_rel": num_rel,  # number of relations
        "ent_dim": args.ent_dim,  # Embedding dimension of the entities
        "rel_dim": args.rel_dim,  # Embedding dimension of the relations
        "time_dim": args.time_dim,  # Embedding dimension of the timestamps
        "state_dim": args.state_dim,  # dimension of the LSTM hidden state
        "action_dim": args.ent_dim + args.rel_dim,  # dimension of the actions
        "mlp_input_dim": args.ent_dim + args.rel_dim + args.state_dim, # dimension of the input of the MLP
        "mlp_hidden_dim": args.hidden_dim,  # dimension of the MLP hidden layer
        "droprate": args.droprate, # dropout rate
        "path_length": args.path_length,  # agent search path length
        "max_action_num": args.max_action_num,  # max candidate action number
        "lambda": args.Lambda,  # update rate of baseline
        "gamma": args.Gamma,  # discount factor of Bellman Eq.
        "ita": args.Ita,  # regular proportionality constant
        "zita": args.Zita,  # attenuation factor of entropy regular term
        "phi": args.phi,  # attenuation factor of entropy regular term
        "ac": args.ac, # Use Actor-Critic Training
        "critic_weight": args.critic_weight, # weight for Actor-Critic
        "beam_size": args.beam_size,  # beam size for beam search
        "entities_embeds_method": args.entities_embeds_method, # default: 'dynamic', otherwise static encoder will be used
        "history_encoder": args.history_encoder, # how to learn path embeddings
        "use_confidence": args.use_confidence, # whether to calculate confidence
        "confidence_mode": args.confidence_mode, # confidence mode
        "policy_network": args.policy_network, # policy-network type
        "spline_type": args.spline_type, # spline type
        "lowfer_k": args.lowfer_k, # lowfer k for confidence mode with LowFER
        "rule": args.rule, # Uuse rule
        "rules_path": args.rules_path, # Rule path
        "r_weight": args.r_weight, 
        "multi_reward": args.multi_reward, # Use multi-reward
        "path_weight": args.path_weight
    }
    return config