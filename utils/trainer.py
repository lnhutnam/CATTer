import torch
import json
import os
import tqdm


class Trainer(object):
    def __init__(self, model, pg, optimizer, args, distribution=None):
        self.model = model
        self.pg = pg
        self.optimizer = optimizer
        self.args = args
        self.distribution = distribution
        
    def adjust_control(self, prev_metrics, curr_metrics):
        curr_mrr = curr_metrics["MRR"]
        prev_mrr = prev_metrics["MRR"]
        
        if curr_mrr < prev_mrr:
            self.args.alpha_r += 0.1
            self.args.alpha_f += 0.1
            self.args.alpha_e += 0.1

    def train_epoch(self, dataloader, ntriple):
        self.model.train()
        total_loss = 0.0
        total_reward = 0.0
        counter = 0
        with tqdm.tqdm(total=ntriple, unit="ex") as bar:
            bar.set_description("Train")
            for src_batch, rel_batch, dst_batch, time_batch in dataloader:
                if self.args.cuda:
                    src_batch = src_batch.cuda()
                    rel_batch = rel_batch.cuda()
                    dst_batch = dst_batch.cuda()
                    time_batch = time_batch.cuda()

                # all_loss, all_logits, _, current_entities, current_time = self.model(src_batch, time_batch, rel_batch)
                all_loss, all_logits, all_relations, current_entities, current_time, current_entites_lst = self.model(src_batch, time_batch, rel_batch)
                # reward = self.pg.get_reward(current_entities, dst_batch)
                if self.args.rule:
                    global_reward = self.pg.get_global_reward(current_entities, dst_batch)
                    rule_reward = self.pg.get_rule_reward(all_relations).cuda()
                    alpha_r = self.args.alpha_r
                    reward = (1 + alpha_r * rule_reward) * global_reward
                    
                elif self.args.freq:
                    global_reward = self.pg.get_global_reward(current_entities, dst_batch)
                    frequency_reward = self.pg.get_frequency_reward(current_entites_lst, dst_batch, current_time, time_batch)
                    alpha_f = self.args.alpha_f
                    reward = (1 + alpha_f * frequency_reward) * global_reward
                    
                elif self.args.eff:
                    global_reward = self.pg.get_global_reward(current_entities, dst_batch)
                    path_length_reward = self.pg.get_pathlength_reward(current_entites_lst, dst_batch)
                    alpha_e = self.args.alpha_e
                    reward = (1 + alpha_e * path_length_reward) * global_reward
                    
                elif self.args.multi_reward:
                    global_reward = self.pg.get_global_reward(current_entities, dst_batch) # torch.Size([512])
                    frequency_reward = self.pg.get_frequency_reward(current_entites_lst, dst_batch, current_time, time_batch)
                    path_length_reward = self.pg.get_pathlength_reward(current_entites_lst, dst_batch)
                    rule_reward = self.pg.get_rule_reward(all_relations).cuda()
                    
                    alpha_r = self.args.alpha_r
                    alpha_f = self.args.alpha_f
                    alpha_e = self.args.alpha_e
                
                    reward = (1 + alpha_f * frequency_reward) * (1 + alpha_r * rule_reward) * (global_reward + alpha_e * path_length_reward)
                else:
                    reward = self.pg.get_reward(current_entities, dst_batch)
                        
                if self.args.reward_shaping:
                    # reward shaping
                    delta_time = time_batch - current_time
                    p_dt = []

                    for i in range(rel_batch.shape[0]):
                        rel = rel_batch[i].item()
                        dt = delta_time[i].item()
                        p_dt.append(self.distribution(rel, dt // self.args.time_span))

                    p_dt = torch.tensor(p_dt)
                    if self.args.cuda:
                        p_dt = p_dt.cuda()
                    shaped_reward = (1 + p_dt) * reward
                    cum_discounted_reward = self.pg.calc_cum_discounted_reward(shaped_reward)
                else:
                    cum_discounted_reward = self.pg.calc_cum_discounted_reward(reward)

                reinfore_loss = self.pg.calc_reinforce_loss(all_loss, all_logits, cum_discounted_reward)
                self.pg.baseline.update(torch.mean(cum_discounted_reward))
                self.pg.now_epoch += 1

                self.optimizer.zero_grad()
                reinfore_loss.backward()
                if self.args.clip_gradient:
                    total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_gradient)
                self.optimizer.step()

                total_loss += reinfore_loss
                total_reward += torch.mean(reward)
                counter += 1
                bar.update(self.args.batch_size)
                bar.set_postfix(loss="%.4f" % reinfore_loss, reward="%.4f" % torch.mean(reward).item(), )
        return total_loss / counter, total_reward / counter

    def save_model(self, checkpoint_path="checkpoint.pth"):
        """Save the parameters of the model and the optimizer,"""
        argparse_dict = vars(self.args)
        
        with open(os.path.join(self.args.save_path, "config.json"), "w") as fjson:
            json.dump(argparse_dict, fjson)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            os.path.join(self.args.save_path, checkpoint_path),
        )


class ACTrainer(object):
    def __init__(self, model, pg, optimizer, args, distribution=None):
        self.model = model
        self.pg = pg
        self.optimizer = optimizer
        self.args = args
        self.distribution = distribution

    def train_epoch(self, dataloader, ntriple):
        self.model.train()
        total_loss = 0.0
        total_reward = 0.0
        counter = 0
        with tqdm.tqdm(total=ntriple, unit='ex') as bar:
            bar.set_description('Train')
            for src_batch, rel_batch, dst_batch, time_batch in dataloader:
                if self.args.cuda:
                    src_batch = src_batch.cuda()
                    rel_batch = rel_batch.cuda()
                    dst_batch = dst_batch.cuda()
                    time_batch = time_batch.cuda()

                all_loss, all_logits, all_relations, all_critic_values, current_entities, current_time, current_entites_lst = self.model(src_batch, time_batch, rel_batch)
                if self.args.rule:
                    global_reward = self.pg.get_global_reward(current_entities, dst_batch)
                    rule_reward = self.pg.get_rule_reward(all_relations).cuda()
                    alpha_r = self.args.alpha_r
                    reward = (1 + alpha_r * rule_reward) * global_reward
                    
                elif self.args.freq:
                    global_reward = self.pg.get_global_reward(current_entities, dst_batch)
                    frequency_reward = self.pg.get_frequency_reward(current_entites_lst, dst_batch, current_time, time_batch)
                    alpha_f = self.args.alpha_f
                    reward = (1 + alpha_f * frequency_reward) * global_reward
                    
                elif self.args.eff:
                    global_reward = self.pg.get_global_reward(current_entities, dst_batch)
                    path_length_reward = self.pg.get_pathlength_reward(current_entites_lst, dst_batch)
                    alpha_e = self.args.alpha_e
                    reward = (1 + alpha_e * path_length_reward) * global_reward
                    
                elif self.args.multi_reward:
                    global_reward = self.pg.get_global_reward(current_entities, dst_batch) # torch.Size([512])
                    frequency_reward = self.pg.get_frequency_reward(current_entites_lst, dst_batch, current_time, time_batch)
                    path_length_reward = self.pg.get_pathlength_reward(current_entites_lst, dst_batch)
                    rule_reward = self.pg.get_rule_reward(all_relations).cuda()
                    
                    alpha_r = self.args.alpha_r
                    alpha_f = self.args.alpha_f
                    alpha_e = self.args.alpha_e
                
                    reward = (1 + alpha_f * frequency_reward) * (1 + alpha_r * rule_reward) * (global_reward + alpha_e * path_length_reward)
                else:
                    reward = self.pg.get_reward(current_entities, dst_batch)
                
                if self.args.reward_shaping:
                    # reward shaping
                    delta_time = time_batch - current_time
                    p_dt = []

                    for i in range(rel_batch.shape[0]):
                        rel = rel_batch[i].item()
                        dt = delta_time[i].item()
                        p_dt.append(self.distribution(
                            rel, dt // self.args.time_span))

                    p_dt = torch.tensor(p_dt)
                    if self.args.cuda:
                        p_dt = p_dt.cuda()
                    shaped_reward = (1 + p_dt) * reward
                    cum_discounted_reward = self.pg.calc_cum_discounted_reward(shaped_reward, all_critic_values)
                else:
                    cum_discounted_reward = self.pg.calc_cum_discounted_reward(reward, all_critic_values)
                reinfore_loss, final_reward = self.pg.calc_reinforce_loss(all_loss, all_logits, all_critic_values, cum_discounted_reward)
                
                # to be done:
                # critic_loss = MSE(all_critic_values, tagets)
                # actor_critic_loss = reinfore_loss + critic_loss
                # criterion = torch.nn.MSELoss()
                critic_loss = final_reward.pow(2).mean()
                actor_critic_loss = reinfore_loss + self.args.critic_weight * critic_loss
                # self.pg.baseline.update(torch.mean(cum_discounted_reward))
                self.pg.now_epoch += 1

                self.optimizer.zero_grad()
                # reinfore_loss.backward()
                actor_critic_loss.backward()
                if self.args.clip_gradient:
                    total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_gradient)
                self.optimizer.step()

                # total_loss += reinfore_loss
                total_loss += actor_critic_loss
                total_reward += torch.mean(reward)
                counter += 1
                bar.update(self.args.batch_size)
                bar.set_postfix(loss='%.4f' % reinfore_loss,
                                reward='%.4f' % torch.mean(reward).item())
        return total_loss / counter, total_reward / counter

    def save_model(self, checkpoint_path='checkpoint.pth'):
        """Save the parameters of the model and the optimizer,"""
        argparse_dict = vars(self.args)
        with open(os.path.join(self.args.save_path, 'config.json'), 'w') as fjson:
            json.dump(argparse_dict, fjson)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()},
            os.path.join(self.args.save_path, checkpoint_path)
        )
