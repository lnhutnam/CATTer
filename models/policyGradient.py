import math
import json

import torch
import torch.nn.functional as F
from models.baseline import ReactiveBaseline

import numpy as np


class PG(object):
    def __init__(self, config):
        self.config = config
        self.positive_reward = 1.0
        self.negative_reward = 0.0
        self.baseline = ReactiveBaseline(config, config["lambda"])
        self.now_epoch = 0

        self.agent_path_length_prev = torch.zeros(
            config["num_ent"], dtype=torch.float32
        ).cuda()
        self.agent_path_length_curr = torch.zeros(
            config["num_ent"], dtype=torch.float32
        ).cuda()

        self.path_weight = config["path_weight"]

        if config["rule"]:
            self.weight = config["r_weight"]
            with open(config["rules_path"], "r", encoding="utf-8") as f:
                self.preknown_rules = json.load(f)

    def get_reward(self, current_entities, answers):
        positive = (
            torch.ones_like(current_entities, dtype=torch.float32)
            * self.positive_reward
        )
        negative = (
            torch.ones_like(current_entities, dtype=torch.float32)
            * self.negative_reward
        )
        reward = torch.where(current_entities == answers, positive, negative)
        return reward

    def get_reward_continuous(self, current_ent_embs, answer_embs):
        reward = torch.sigmoid(
            5
            - torch.norm(
                abs(current_ent_embs - answer_embs), p=2, dim=1, dtype=torch.float32
            )
        )
        return reward

    def get_global_reward(self, current_entities, answers):
        positive = (
            torch.ones_like(current_entities, dtype=torch.float32)
            * self.positive_reward
        )
        negative = (
            torch.ones_like(current_entities, dtype=torch.float32)
            * self.negative_reward
        )
        r_g = torch.where(current_entities == answers, positive, negative)
        return r_g

    def get_frequency_reward(
        self, current_entites_lst, answers, current_time, time_batch
    ):
        masks = []

        for idx in range(len(current_entites_lst)):
            positive = (
                torch.ones_like(current_entites_lst[idx], dtype=torch.float32)
                * self.positive_reward
            )
            negative = (
                torch.ones_like(current_entites_lst[idx], dtype=torch.float32)
                * self.negative_reward
            )
            mask = torch.where(current_entites_lst[idx] == answers, positive, negative)
            masks.append(mask)

        r_f = []

        for idx in range(len(current_entites_lst)):
            if max(current_time) < max(time_batch):
                current_entities = current_entites_lst[idx]
                selected_indices = torch.nonzero(masks[idx], as_tuple=False).squeeze()
                selected_elements = current_entities[selected_indices]
                output, inverse_indices = torch.unique(
                    selected_elements, return_inverse=True
                )
                counts = torch.bincount(inverse_indices)
                output = counts[inverse_indices]
                max_target = torch.max(counts)
                min_target = torch.min(counts)
                output = output / (max_target - min_target)
                mask[selected_indices] = output
                r_f.append(mask)
            else:
                r_f.append(mask)

        r_f = torch.stack(r_f, dim=0).sum(dim=0) + 1

        return r_f

    def get_pathlength_reward(self, current_entites_lst, answers):
        masks = []

        for idx in range(len(current_entites_lst)):
            positive = (
                torch.ones_like(current_entites_lst[idx], dtype=torch.float32)
                * self.positive_reward
            )
            negative = (
                torch.ones_like(current_entites_lst[idx], dtype=torch.float32)
                * self.negative_reward
            )
            mask = torch.where(current_entites_lst[idx] == answers, positive, negative)
            masks.append(mask)

        mask = torch.stack(masks, dim=0).sum(dim=0)
        mask = torch.abs(torch.ones_like(mask, dtype=torch.float32) - mask) + 1
        r_p = self.path_weight / (mask - 1)
        return r_p

    def get_rule_reward(self, record):
        reward = torch.zeros(len(record[0]))
        for idx in range(len(record[0])):
            if (record[0][idx], record[1][idx]) in self.preknown_rules:
                reward[idx] = self.weight
        return reward

    def calc_cum_discounted_reward(self, rewards):
        running_add = torch.zeros([rewards.shape[0]])
        cum_disc_reward = torch.zeros([rewards.shape[0], self.config["path_length"]])
        if self.config["cuda"]:
            running_add = running_add.cuda()
            cum_disc_reward = cum_disc_reward.cuda()

        cum_disc_reward[:, self.config["path_length"] - 1] = rewards
        for t in reversed(range(self.config["path_length"])):
            running_add = self.config["gamma"] * running_add + cum_disc_reward[:, t]
            cum_disc_reward[:, t] = running_add
        return cum_disc_reward

    def entropy_reg_loss(self, all_logits):
        all_logits = torch.stack(all_logits, dim=2)
        entropy_loss = -torch.mean(
            torch.sum(torch.mul(torch.exp(all_logits), all_logits), dim=1)
        )
        return entropy_loss

    def calc_reinforce_loss(self, all_loss, all_logits, cum_discounted_reward):
        loss = torch.stack(all_loss, dim=1)
        base_value = self.baseline.get_baseline_value()
        final_reward = cum_discounted_reward - base_value

        reward_mean = torch.mean(final_reward)
        reward_std = torch.std(final_reward) + 1e-6
        final_reward = torch.div(final_reward - reward_mean, reward_std)

        loss = torch.mul(loss, final_reward)
        entropy_loss = (
            self.config["ita"]
            * math.pow(self.config["zita"], self.now_epoch)
            * self.entropy_reg_loss(all_logits)
        )

        total_loss = torch.mean(loss) - entropy_loss
        return total_loss


class ACPG(object):
    def __init__(self, config):
        self.config = config
        self.positive_reward = 1.0
        self.negative_reward = 0.0
        self.baseline = ReactiveBaseline(config, config["lambda"])
        self.now_epoch = 0

        self.agent_path_length_prev = torch.zeros(
            config["num_ent"], dtype=torch.float32
        ).cuda()
        self.agent_path_length_curr = torch.zeros(
            config["num_ent"], dtype=torch.float32
        ).cuda()

        if config["rule"]:
            self.weight = config["r_weight"]
            with open(config["rules_path"], "r", encoding="utf-8") as f:
                self.preknown_rules = json.load(f)

    def get_reward(self, current_entities, answers):
        positive = (
            torch.ones_like(current_entities, dtype=torch.float32)
            * self.positive_reward
        )
        negative = (
            torch.ones_like(current_entities, dtype=torch.float32)
            * self.negative_reward
        )
        reward = torch.where(current_entities == answers, positive, negative)
        return reward

    def get_reward_continuous(self, current_ent_embs, answer_embs):
        reward = torch.sigmoid(
            5
            - torch.norm(
                abs(current_ent_embs - answer_embs), p=2, dim=1, dtype=torch.float32
            )
        )
        return reward

    def get_global_reward(self, current_entities, answers):
        positive = (
            torch.ones_like(current_entities, dtype=torch.float32)
            * self.positive_reward
        )
        negative = (
            torch.ones_like(current_entities, dtype=torch.float32)
            * self.negative_reward
        )
        r_g = torch.where(current_entities == answers, positive, negative)
        return r_g

    def get_frequency_reward(
        self, current_entites_lst, answers, current_time, time_batch
    ):
        masks = []

        for idx in range(len(current_entites_lst)):
            positive = (
                torch.ones_like(current_entites_lst[idx], dtype=torch.float32)
                * self.positive_reward
            )
            negative = (
                torch.ones_like(current_entites_lst[idx], dtype=torch.float32)
                * self.negative_reward
            )
            mask = torch.where(current_entites_lst[idx] == answers, positive, negative)
            masks.append(mask)

        r_f = []

        for idx in range(len(current_entites_lst)):
            if max(current_time) < max(time_batch):
                current_entities = current_entites_lst[idx]
                selected_indices = torch.nonzero(masks[idx], as_tuple=False).squeeze()
                selected_elements = current_entities[selected_indices]
                output, inverse_indices = torch.unique(
                    selected_elements, return_inverse=True
                )
                counts = torch.bincount(inverse_indices)
                output = counts[inverse_indices]
                max_target = torch.max(counts)
                output = output / max_target
                mask[selected_indices] = output
                r_f.append(mask)
            else:
                r_f.append(mask)

        r_f = torch.stack(r_f, dim=0).sum(dim=0) + 1

        return r_f

    def get_pathlength_reward(self, current_entites_lst, answers):
        masks = []

        for idx in range(len(current_entites_lst)):
            positive = (
                torch.ones_like(current_entites_lst[idx], dtype=torch.float32)
                * self.positive_reward
            )
            negative = (
                torch.ones_like(current_entites_lst[idx], dtype=torch.float32)
                * self.negative_reward
            )
            mask = torch.where(current_entites_lst[idx] == answers, positive, negative)
            masks.append(mask)

        mask = torch.stack(masks, dim=0).sum(dim=0)
        mask = torch.abs(torch.ones_like(mask, dtype=torch.float32) - mask) + 1
        r_p = 1 / mask
        return r_p

    def get_rule_reward(self, record):
        reward = torch.zeros(len(record[0]))
        for idx in range(len(record[0])):
            if (record[0][idx], record[1][idx]) in self.preknown_rules:
                reward[idx] = self.weight
        return reward

    def calc_cum_discounted_reward(self, rewards, all_critic_values):
        # running_add = torch.zeros([rewards.shape[0]])
        running_add = (
            torch.stack(all_critic_values, dim=-1).squeeze(dim=1)[:, -1]
            * self.config["phi"]
        )
        cum_disc_reward = torch.zeros(
            [rewards.shape[0], self.config["path_length"]]
        )  # shape: [batchsize, path_length]
        if self.config["cuda"]:
            running_add = running_add.cuda()
            cum_disc_reward = cum_disc_reward.cuda()

        cum_disc_reward[:, self.config["path_length"] - 1] = rewards
        for t in reversed(range(self.config["path_length"])):
            running_add = self.config["gamma"] * running_add + cum_disc_reward[:, t]
            # running_add = self.config['gamma'] * running_add + cum_disc_reward[:, t]
            # all_critic_values.shape: [batchsize, pathlength + 1]
            cum_disc_reward[:, t] = running_add
        return cum_disc_reward  # shape: [batchsize, pathlength]

    def entropy_reg_loss(self, all_logits):
        all_logits = torch.stack(all_logits, dim=2)
        entropy_loss = -torch.mean(
            torch.sum(torch.mul(torch.exp(all_logits), all_logits), dim=1)
        )
        return entropy_loss

    def calc_reinforce_loss(
        self, all_loss, all_logits, all_critic_values, cum_discounted_reward
    ):
        loss = torch.stack(all_loss, dim=1)
        base_value = self.baseline.get_baseline_value()
        critic_values = torch.stack(all_critic_values, dim=-1).squeeze(dim=1)
        final_reward = (
            cum_discounted_reward
            - base_value
            - critic_values[:, :-1] * self.config["phi"]
        )
        # cum_discounted_reward.shape: [batchsize, pathlength]
        # critic_values.shape: [batchsize, pathlength + 1]

        reward_mean = torch.mean(final_reward)
        reward_std = torch.std(final_reward) + 1e-6
        final_reward = torch.div(final_reward - reward_mean, reward_std)

        loss = torch.mul(loss, final_reward)
        entropy_loss = (
            self.config["ita"]
            * math.pow(self.config["zita"], self.now_epoch)
            * self.entropy_reg_loss(all_logits)
        )

        total_loss = torch.mean(loss) - entropy_loss
        return total_loss, final_reward
