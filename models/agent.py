import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.xkan import KANLinear
from models.xkan import (
    BesselKANLayer,
    BesselKANLayerWithNorm,
    ChebyKANLayer,
    ChebyKANLayerWithNorm,
    FibonacciKANLayer,
    FibonacciKANLayerWithNorm,
    FourierKANLayer,
    GegenbauerKANLayer,
    GegenbauerKANLayerWithNorm,
    HermiteKANLayer,
    HermiteKANLayerWithNorm,
    JacobiKANLayer,
    JacobiKANLayerWithNorm,
    LaguerreKANLayer,
    LaguerreKANLayerNorm,
    LegendreKANLayer,
    LegendreKANLayerWithNorm,
    LucasKANLayer,
    LucasKANLayerWithNorm,
)


class HistoryEncoder(nn.Module):
    def __init__(self, config):
        super(HistoryEncoder, self).__init__()
        self.config = config
        if self.config["history_encoder"] == "lstm":
            self.history_encoder = torch.nn.LSTMCell(
                input_size=config["action_dim"], hidden_size=config["state_dim"]
            )
        elif self.config["history_encoder"] == "gru":
            self.history_encoder = torch.nn.GRUCell(
                input_size=config["action_dim"], hidden_size=config["state_dim"]
            )
        else:
            raise NotImplementedError(f"History encoder is not implemented.")

    def set_hiddenx(self, batch_size):
        """Set hidden layer parameters. Initialize to 0"""
        if self.config["cuda"]:
            self.hx = torch.zeros(batch_size, self.config["state_dim"], device="cuda")
            if self.config["history_encoder"] == "lstm":
                self.cx = torch.zeros(
                    batch_size, self.config["state_dim"], device="cuda"
                )
        else:
            self.hx = torch.zeros(batch_size, self.config["state_dim"])
            if self.config["history_encoder"] == "lstm":
                self.cx = torch.zeros(batch_size, self.config["state_dim"])

    def forward(self, prev_action, mask):
        """mask: True if NO_OP. ON_OP does not affect history coding results"""
        if self.config["history_encoder"] == "lstm":
            self.hx_, self.cx_ = self.history_encoder(prev_action, (self.hx, self.cx))
            self.hx = torch.where(mask, self.hx, self.hx_)
            self.cx = torch.where(mask, self.cx, self.cx_)
            return self.hx
        elif self.config["history_encoder"] == "gru":
            self.hx_ = self.history_encoder(prev_action, self.hx)
            self.hx = torch.where(mask, self.hx, self.hx_)
            return self.hx


class PolicyKAN(nn.Module):
    def __init__(self, config):
        super(PolicyKAN, self).__init__()
        self.config = config
        if config["ac"]:
            self.ac = True
        else:
            self.ac = False

        if config["spline_type"] == "B-Spline":
            self.kan_l1 = KANLinear(config["mlp_input_dim"], config["mlp_hidden_dim"])
            self.kan_l2 = KANLinear(config["mlp_hidden_dim"], config["action_dim"])
            if self.config["ac"]:
                self.kan_l3 = KANLinear(config["mlp_hidden_dim"], 1)

        elif config["spline_type"] == "Bessel":
            self.kan_l1 = BesselKANLayer(
                config["mlp_input_dim"], config["mlp_hidden_dim"]
            )
            self.kan_l2 = BesselKANLayer(config["mlp_hidden_dim"], config["action_dim"])
            if self.config["ac"]:
                self.kan_l3 = BesselKANLayer(config["mlp_hidden_dim"], 1)

        elif config["spline_type"] == "Bessel_Norm":
            self.kan_l1 = BesselKANLayerWithNorm(
                config["mlp_input_dim"], config["mlp_hidden_dim"]
            )
            self.kan_l2 = BesselKANLayerWithNorm(
                config["mlp_hidden_dim"], config["action_dim"]
            )
            if self.config["ac"]:
                self.kan_l3 = BesselKANLayerWithNorm(config["mlp_hidden_dim"], 1)

        elif config["spline_type"] == "Chebyshev":
            self.kan_l1 = ChebyKANLayer(
                config["mlp_input_dim"], config["mlp_hidden_dim"]
            )
            self.kan_l2 = ChebyKANLayer(config["mlp_hidden_dim"], config["action_dim"])
            if self.config["ac"]:
                self.kan_l3 = ChebyKANLayer(config["mlp_hidden_dim"], 1)

        elif config["spline_type"] == "Chebyshev_Norm":
            self.kan_l1 = ChebyKANLayerWithNorm(
                config["mlp_input_dim"], config["mlp_hidden_dim"]
            )
            self.kan_l2 = ChebyKANLayerWithNorm(
                config["mlp_hidden_dim"], config["action_dim"]
            )
            if self.config["ac"]:
                self.kan_l3 = ChebyKANLayerWithNorm(config["mlp_hidden_dim"], 1)

        elif config["spline_type"] == "Fibonacci":
            self.kan_l1 = FibonacciKANLayer(
                config["mlp_input_dim"], config["mlp_hidden_dim"]
            )
            self.kan_l2 = FibonacciKANLayer(
                config["mlp_hidden_dim"], config["action_dim"]
            )
            if self.config["ac"]:
                self.kan_l3 = FibonacciKANLayer(config["mlp_hidden_dim"], 1)

        elif config["spline_type"] == "Fibonacci_Norm":
            self.kan_l1 = FibonacciKANLayerWithNorm(
                config["mlp_input_dim"], config["mlp_hidden_dim"]
            )
            self.kan_l2 = FibonacciKANLayerWithNorm(
                config["mlp_hidden_dim"], config["action_dim"]
            )
            if self.config["ac"]:
                self.kan_l3 = FibonacciKANLayerWithNorm(config["mlp_hidden_dim"], 1)

        elif config["spline_type"] == "Fourier":
            self.kan_l1 = FourierKANLayer(
                config["mlp_input_dim"], config["mlp_hidden_dim"]
            )
            self.kan_l2 = FourierKANLayer(
                config["mlp_hidden_dim"], config["action_dim"]
            )
            if self.config["ac"]:
                self.kan_l3 = FourierKANLayer(config["mlp_hidden_dim"], 1)

        elif config["spline_type"] == "Gegenbauer":
            self.kan_l1 = GegenbauerKANLayer(
                config["mlp_input_dim"], config["mlp_hidden_dim"]
            )
            self.kan_l2 = GegenbauerKANLayer(
                config["mlp_hidden_dim"], config["action_dim"]
            )
            if self.config["ac"]:
                self.kan_l3 = GegenbauerKANLayer(config["mlp_hidden_dim"], 1)

        elif config["spline_type"] == "Gegenbauer_Norm":
            self.kan_l1 = GegenbauerKANLayerWithNorm(
                config["mlp_input_dim"], config["mlp_hidden_dim"]
            )
            self.kan_l2 = GegenbauerKANLayerWithNorm(
                config["mlp_hidden_dim"], config["action_dim"]
            )
            if self.config["ac"]:
                self.kan_l3 = GegenbauerKANLayerWithNorm(config["mlp_hidden_dim"], 1)

        elif config["spline_type"] == "Hermite":
            self.kan_l1 = HermiteKANLayer(
                config["mlp_input_dim"], config["mlp_hidden_dim"]
            )
            self.kan_l2 = HermiteKANLayer(
                config["mlp_hidden_dim"], config["action_dim"]
            )
            if self.config["ac"]:
                self.kan_l3 = HermiteKANLayer(config["mlp_hidden_dim"], 1)

        elif config["spline_type"] == "Hermite_Norm":
            self.kan_l1 = HermiteKANLayerWithNorm(
                config["mlp_input_dim"], config["mlp_hidden_dim"]
            )
            self.kan_l2 = HermiteKANLayerWithNorm(
                config["mlp_hidden_dim"], config["action_dim"]
            )
            if self.config["ac"]:
                self.kan_l3 = HermiteKANLayerWithNorm(config["mlp_hidden_dim"], 1)

        elif config["spline_type"] == "Jacobi":
            self.kan_l1 = JacobiKANLayer(
                config["mlp_input_dim"], config["mlp_hidden_dim"]
            )
            self.kan_l2 = JacobiKANLayer(config["mlp_hidden_dim"], config["action_dim"])
            if self.config["ac"]:
                self.kan_l3 = JacobiKANLayer(config["mlp_hidden_dim"], 1)

        elif config["spline_type"] == "Jacobi_Norm":
            self.kan_l1 = JacobiKANLayerWithNorm(
                config["mlp_input_dim"], config["mlp_hidden_dim"]
            )
            self.kan_l2 = JacobiKANLayerWithNorm(
                config["mlp_hidden_dim"], config["action_dim"]
            )
            if self.config["ac"]:
                self.kan_l3 = JacobiKANLayerWithNorm(config["mlp_hidden_dim"], 1)

        elif config["spline_type"] == "Laguerre":
            self.kan_l1 = LaguerreKANLayer(
                config["mlp_input_dim"], config["mlp_hidden_dim"]
            )
            self.kan_l2 = LaguerreKANLayer(
                config["mlp_hidden_dim"], config["action_dim"]
            )
            if self.config["ac"]:
                self.kan_l3 = LaguerreKANLayer(config["mlp_hidden_dim"], 1)

        elif config["spline_type"] == "Laguerre_Norm":
            self.kan_l1 = LaguerreKANLayerNorm(
                config["mlp_input_dim"], config["mlp_hidden_dim"]
            )
            self.kan_l2 = LaguerreKANLayerNorm(
                config["mlp_hidden_dim"], config["action_dim"]
            )
            if self.config["ac"]:
                self.kan_l3 = LaguerreKANLayerNorm(config["mlp_hidden_dim"], 1)

        elif config["spline_type"] == "Legendre":
            self.kan_l1 = LegendreKANLayer(
                config["mlp_input_dim"], config["mlp_hidden_dim"]
            )
            self.kan_l2 = LegendreKANLayer(
                config["mlp_hidden_dim"], config["action_dim"]
            )
            if self.config["ac"]:
                self.kan_l3 = LegendreKANLayer(config["mlp_hidden_dim"], 1)

        elif config["spline_type"] == "Legendre_Norm":
            self.kan_l1 = LegendreKANLayerWithNorm(
                config["mlp_input_dim"], config["mlp_hidden_dim"]
            )
            self.kan_l2 = LegendreKANLayerWithNorm(
                config["mlp_hidden_dim"], config["action_dim"]
            )
            if self.config["ac"]:
                self.kan_l3 = LegendreKANLayerWithNorm(config["mlp_hidden_dim"], 1)

        elif config["spline_type"] == "Lucas":
            self.kan_l1 = LucasKANLayer(
                config["mlp_input_dim"], config["mlp_hidden_dim"]
            )
            self.kan_l2 = LucasKANLayer(config["mlp_hidden_dim"], config["action_dim"])
            if self.config["ac"]:
                self.kan_l3 = LucasKANLayer(config["mlp_hidden_dim"], 1)

        elif config["spline_type"] == "Lucas_Norm":
            self.kan_l1 = LucasKANLayerWithNorm(
                config["mlp_input_dim"], config["mlp_hidden_dim"]
            )
            self.kan_l2 = LucasKANLayerWithNorm(
                config["mlp_hidden_dim"], config["action_dim"]
            )
            if self.config["ac"]:
                self.kan_l3 = LucasKANLayerWithNorm(config["mlp_hidden_dim"], 1)

    def forward(self, state_query):
        hidden = torch.relu(self.kan_l1(state_query))
        output = self.kan_l2(hidden).unsqueeze(1)
        if self.ac:
            critic_values = self.kan_l3(hidden)
            return output, critic_values
        else:
            return output


class PolicyMLP(nn.Module):
    def __init__(self, config):
        super(PolicyMLP, self).__init__()
        self.mlp_l1 = nn.Linear(
            config["mlp_input_dim"], config["mlp_hidden_dim"], bias=True
        )
        self.mlp_l2 = nn.Linear(
            config["mlp_hidden_dim"], config["action_dim"], bias=True
        )
        if config["ac"]:
            self.ac = True
            self.mlp_l3 = nn.Linear(config["mlp_hidden_dim"], 1)
        else:
            self.ac = False

    def forward(self, state_query):
        hidden = torch.relu(self.mlp_l1(state_query))
        output = self.mlp_l2(hidden).unsqueeze(1)
        if self.ac:
            critic_values = self.mlp_l3(hidden)
            return output, critic_values
        else:
            return output


class DynamicEmbedding(nn.Module):
    def __init__(self, n_ent, dim_ent, dim_t):
        super(DynamicEmbedding, self).__init__()
        self.ent_embs = nn.Embedding(n_ent, dim_ent - dim_t)
        self.w = torch.nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim_t))).float()
        )
        self.b = torch.nn.Parameter(torch.zeros(dim_t).float())

    def forward(self, entities, dt):
        dt = dt.unsqueeze(-1)
        batch_size = dt.size(0)
        seq_len = dt.size(1)

        dt = dt.view(batch_size, seq_len, 1)
        t = torch.cos(self.w.view(1, 1, -1) * dt + self.b.view(1, 1, -1))
        t = t.squeeze(1)  # [batch_size, time_dim]

        e = self.ent_embs(entities)
        return torch.cat((e, t), -1)


class StaticEmbedding(nn.Module):
    def __init__(self, n_ent, dim_ent):
        super(StaticEmbedding, self).__init__()
        self.ent_embs = nn.Embedding(n_ent, dim_ent)

    def forward(self, entities, timestamps=None):
        return self.ent_embs(entities)


class Agent(nn.Module):
    def __init__(self, config):
        super(Agent, self).__init__()
        self.num_rel = config["num_rel"] * 2 + 2
        self.config = config

        # [0, num_rel) -> normal relations; num_rel -> stay in place，(num_rel, num_rel * 2] reversed relations.
        self.NO_OP = self.num_rel  # Stay in place; No Operation
        self.ePAD = config["num_ent"]  # Padding entity
        self.rPAD = config["num_rel"] * 2 + 1  # Padding relation
        self.tPAD = 0  # Padding time

        if self.config["entities_embeds_method"] == "dynamic":
            self.ent_embs = DynamicEmbedding(
                config["num_ent"] + 1, config["ent_dim"], config["time_dim"]
            )
        else:
            self.ent_embs = StaticEmbedding(config["num_ent"] + 1, config["ent_dim"])

        self.rel_embs = nn.Embedding(config["num_ent"], config["rel_dim"])

        self.policy_step = HistoryEncoder(config)

        if self.config["policy_network"] == "MLP":
            self.policy_mlp = PolicyMLP(config)
        else:
            self.policy_mlp = PolicyKAN(config)

        self.score_weighted_fc = nn.Linear(
            self.config["ent_dim"] * 2
            + self.config["rel_dim"] * 2
            + self.config["state_dim"],
            1,
            bias=True,
        )

    def forward(
        self,
        prev_relation,
        current_entities,
        current_timestamps,
        query_relation,
        query_entity,
        query_timestamps,
        action_space,
    ):
        """
        Args:
            prev_relation: [batch_size]
            current_entities: [batch_size]
            current_timestamps: [batch_size]
            query_relation: embeddings of query relation, [batch_size, rel_dim]
            query_entity: embeddings of query entity, [batch_size, ent_dim]
            query_timestamps: [batch_size]
            action_space: [batch_size, max_actions_num, 3] (relations, entities, timestamps)
        """
        # embeddings
        current_delta_time = query_timestamps - current_timestamps
        current_embds = self.ent_embs(
            current_entities, current_delta_time
        )  # [batch_size, ent_dim] #dynamic embedding
        prev_relation_embds = self.rel_embs(prev_relation)  # [batch_size, rel_dim]

        # Pad Mask
        pad_mask = (
            torch.ones_like(action_space[:, :, 0]) * self.rPAD
        )  # [batch_size, action_number]
        pad_mask = torch.eq(
            action_space[:, :, 0], pad_mask
        )  # [batch_size, action_number]

        # History Encode
        NO_OP_mask = torch.eq(
            prev_relation, torch.ones_like(prev_relation) * self.NO_OP
        )  # [batch_size]
        NO_OP_mask = NO_OP_mask.repeat(self.config["state_dim"], 1).transpose(
            1, 0
        )  # [batch_size, state_dim]
        prev_action_embedding = torch.cat(
            [prev_relation_embds, current_embds], dim=-1
        )  # [batch_size, rel_dim + ent_dim]
        lstm_output = self.policy_step(
            prev_action_embedding, NO_OP_mask
        )  # [batch_size, state_dim] (5) Path encoding

        # Neighbor/condidate_actions embeddings
        action_num = action_space.size(1)
        neighbors_delta_time = (
            query_timestamps.unsqueeze(-1).repeat(1, action_num) - action_space[:, :, 2]
        )
        neighbors_entities = self.ent_embs(
            action_space[:, :, 1], neighbors_delta_time
        )  # [batch_size, action_num, ent_dim]
        neighbors_relations = self.rel_embs(
            action_space[:, :, 0]
        )  # [batch_size, action_num, rel_dim]

        # agent state representation
        agent_state = torch.cat(
            [lstm_output, query_entity, query_relation], dim=-1
        )  # [batch_size, state_dim + ent_dim + rel_dim]
        output = self.policy_mlp(
            agent_state
        )  # [batch_size, 1, action_dim] action_dim == rel_dim + ent_dim

        # scoring
        entitis_output = output[:, :, self.config["rel_dim"] :]
        relation_ouput = output[:, :, : self.config["rel_dim"]]
        relation_score = torch.sum(
            torch.mul(neighbors_relations, relation_ouput), dim=2
        )
        entities_score = torch.sum(
            torch.mul(neighbors_entities, entitis_output), dim=2
        )  # [batch_size, action_number]

        actions = torch.cat(
            [neighbors_relations, neighbors_entities], dim=-1
        )  # [batch_size, action_number, action_dim]

        agent_state_repeats = agent_state.unsqueeze(1).repeat(1, actions.shape[1], 1)
        score_attention_input = torch.cat([actions, agent_state_repeats], dim=-1)
        a = self.score_weighted_fc(score_attention_input)  # (8)
        a = torch.sigmoid(a).squeeze()  # [batch_size, action_number]   # (8)

        scores = (1 - a) * relation_score + a * entities_score  # (6) a= beta

        # Padding mask
        scores = scores.masked_fill(pad_mask, -1e10)  # [batch_size ,action_number]

        action_prob = torch.softmax(scores, dim=1)
        action_id = torch.multinomial(
            action_prob, 1
        )  # Randomly select an action. [batch_size, 1] # ACTION SELECTION

        logits = torch.nn.functional.log_softmax(
            scores, dim=1
        )  # [batch_size, action_number]
        one_hot = torch.zeros_like(logits).scatter(1, action_id, 1)
        loss = -torch.sum(torch.mul(logits, one_hot), dim=1)
        return loss, logits, action_id

    def get_im_embedding(self, cooccurrence_entities):
        """Get the inductive mean representation of the co-occurrence relation.
        cooccurrence_entities: a list that contains the trained entities with the co-occurrence relation.
        return: torch.tensor, representation of the co-occurrence entities.
        """
        entities = self.ent_embs.ent_embs.weight.data[cooccurrence_entities]
        im = torch.mean(entities, dim=0)
        return im

    def update_entity_embedding(self, entity, ims, mu):
        """Update the entity representation with the co-occurrence relations in the last timestamp.
        entity: int, the entity that needs to be updated.
        ims: torch.tensor, [number of co-occurrence, -1], the IM representations of the co-occurrence relations
        mu: update ratio, the hyperparam.
        """
        self.source_entity = self.ent_embs.ent_embs.weight.data[entity]
        self.ent_embs.ent_embs.weight.data[entity] = mu * self.source_entity + (
            1 - mu
        ) * torch.mean(ims, dim=0)

    def entities_embedding_shift(self, entity, im, mu):
        """Prediction shift."""
        self.source_entity = self.ent_embs.ent_embs.weight.data[entity]
        self.ent_embs.ent_embs.weight.data[entity] = (
            mu * self.source_entity + (1 - mu) * im
        )

    def back_entities_embedding(self, entity):
        """Go back after shift ends."""
        self.ent_embs.ent_embs.weight.data[entity] = self.source_entity


class AgentCT(nn.Module):
    # Agent with confidence
    def __init__(self, config):
        super(AgentCT, self).__init__()
        self.num_rel = config["num_rel"] * 2 + 2
        self.config = config

        # [0, num_rel) -> normal relations; num_rel -> stay in place，(num_rel, num_rel * 2] reversed relations.
        self.NO_OP = self.num_rel  # Stay in place; No Operation
        self.ePAD = config["num_ent"]  # Padding entity
        self.rPAD = config["num_rel"] * 2 + 1  # Padding relation
        self.tPAD = 0  # Padding time

        if self.config["entities_embeds_method"] == "dynamic":
            self.ent_embs = DynamicEmbedding(
                config["num_ent"] + 1, config["ent_dim"], config["time_dim"]
            )
        else:
            self.ent_embs = StaticEmbedding(config["num_ent"] + 1, config["ent_dim"])

        self.rel_embs = nn.Embedding(config["num_ent"], config["rel_dim"])

        self.policy_step = HistoryEncoder(config)

        if self.config["policy_network"] == "KAN":
            self.policy_mlp = PolicyKAN(config)
        elif self.config["policy_network"] == "MLP":
            self.policy_mlp = PolicyMLP(config)
        else:
            raise ModuleNotFoundError(self.config["policy_network"])

        self.score_weighted_fc = nn.Linear(
            self.config["ent_dim"] * 2
            + self.config["rel_dim"] * 2
            + self.config["state_dim"],
            1,
            bias=True,
        )

        if self.config["use_confidence"] and self.config["confidence_mode"] == "tucker":
            self.W_tk = torch.nn.Parameter(
                torch.tensor(
                    np.random.uniform(
                        -1, 1, (config["ent_dim"], config["rel_dim"], config["ent_dim"])
                    ),
                    dtype=torch.float,
                    requires_grad=True,
                )
            )
            self.input_dropout = torch.nn.Dropout(0.1)
            self.hidden_dropout1 = torch.nn.Dropout(0.1)
            self.hidden_dropout2 = torch.nn.Dropout(0.1)

            self.bn0 = torch.nn.BatchNorm1d(config["ent_dim"])
            self.bn1 = torch.nn.BatchNorm1d(config["ent_dim"])

        if self.config["use_confidence"] and self.config["confidence_mode"] == "lowfer":
            if self.config["cuda"] == True:
                dev = "cuda"

            self.k, self.o = self.config["lowfer_k"], config["ent_dim"]
            self.U_tk = torch.nn.Parameter(
                torch.tensor(
                    np.random.uniform(
                        -0.01, 0.01, (config["ent_dim"], self.k * self.o)
                    ),
                    dtype=torch.float,
                    device=dev,
                    requires_grad=True,
                )
            )

            self.V_tk = torch.nn.Parameter(
                torch.tensor(
                    np.random.uniform(
                        -0.01, 0.01, (config["rel_dim"], self.k * self.o)
                    ),
                    dtype=torch.float,
                    device=dev,
                    requires_grad=True,
                )
            )

            self.input_dropout = torch.nn.Dropout(0.1)
            self.hidden_dropout1 = torch.nn.Dropout(0.1)
            self.hidden_dropout2 = torch.nn.Dropout(0.1)

            self.bn0 = torch.nn.BatchNorm1d(config["ent_dim"])
            self.bn1 = torch.nn.BatchNorm1d(config["ent_dim"])

    def forward(
        self,
        prev_relation,
        current_entities,
        current_timestamps,
        query_relation,
        query_entity,
        query_timestamps,
        action_space,
    ):
        """
        Args:
            prev_relation: [batch_size]
            current_entities: [batch_size]
            current_timestamps: [batch_size]
            query_relation: embeddings of query relation, [batch_size, rel_dim]
            query_entity: embeddings of query entity, [batch_size, ent_dim]
            query_timestamps: [batch_size]
            action_space: [batch_size, max_actions_num, 3] (relations, entities, timestamps)
        """
        # embeddings
        current_delta_time = query_timestamps - current_timestamps
        current_embds = self.ent_embs(
            current_entities, current_delta_time
        )  # [batch_size, ent_dim] #dynamic embedding
        prev_relation_embds = self.rel_embs(prev_relation)  # [batch_size, rel_dim]

        # Pad Mask
        pad_mask = (
            torch.ones_like(action_space[:, :, 0]) * self.rPAD
        )  # [batch_size, action_number]
        pad_mask = torch.eq(
            action_space[:, :, 0], pad_mask
        )  # [batch_size, action_number]

        # History Encode
        NO_OP_mask = torch.eq(
            prev_relation, torch.ones_like(prev_relation) * self.NO_OP
        )  # [batch_size]
        NO_OP_mask = NO_OP_mask.repeat(self.config["state_dim"], 1).transpose(
            1, 0
        )  # [batch_size, state_dim]
        prev_action_embedding = torch.cat(
            [prev_relation_embds, current_embds], dim=-1
        )  # [batch_size, rel_dim + ent_dim]
        lstm_output = self.policy_step(
            prev_action_embedding, NO_OP_mask
        )  # [batch_size, state_dim] (5) Path encoding

        # Neighbor/condidate_actions embeddings
        action_num = action_space.size(1)
        neighbors_delta_time = (
            query_timestamps.unsqueeze(-1).repeat(1, action_num) - action_space[:, :, 2]
        )
        neighbors_entities = self.ent_embs(
            action_space[:, :, 1], neighbors_delta_time
        )  # [batch_size, action_num, ent_dim]
        neighbors_relations = self.rel_embs(
            action_space[:, :, 0]
        )  # [batch_size, action_num, rel_dim]

        # agent state representation
        agent_state = torch.cat(
            [lstm_output, query_entity, query_relation], dim=-1
        )  # [batch_size, state_dim + ent_dim + rel_dim]
        output = self.policy_mlp(
            agent_state
        )  # [batch_size, 1, action_dim] action_dim == rel_dim + ent_dim

        # scoring
        entitis_output = output[:, :, self.config["rel_dim"] :]
        relation_ouput = output[:, :, : self.config["rel_dim"]]
        relation_score = torch.sum(
            torch.mul(neighbors_relations, relation_ouput), dim=2
        )
        entities_score = torch.sum(
            torch.mul(neighbors_entities, entitis_output), dim=2
        )  # [batch_size, action_number]

        actions = torch.cat(
            [neighbors_relations, neighbors_entities], dim=-1
        )  # [batch_size, action_number, action_dim]

        agent_state_repeats = agent_state.unsqueeze(1).repeat(1, actions.shape[1], 1)
        score_attention_input = torch.cat([actions, agent_state_repeats], dim=-1)
        a = self.score_weighted_fc(score_attention_input)  # (8)
        a = torch.sigmoid(a).squeeze()  # [batch_size, action_number]   # (8)

        scores = (1 - a) * relation_score + a * entities_score  # (6) a= beta
        conf_scores = self.confidence(
            query_entity,
            query_relation,
            neighbors_entities,
            pad_mask,
            mode=self.config["confidence_mode"],
        )

        scores = scores + conf_scores

        # Padding mask
        scores = scores.masked_fill(pad_mask, -1e10)  # [batch_size ,action_number]

        action_prob = torch.softmax(scores, dim=1)
        action_id = torch.multinomial(
            action_prob, 1
        )  # Randomly select an action. [batch_size, 1] # ACTION SELECTION

        logits = torch.nn.functional.log_softmax(scores, dim=1)
        # logits = torch.log(action_prob)  # [batch_size, action_number]

        one_hot = torch.zeros_like(logits).scatter(1, action_id, 1)
        loss = -torch.sum(torch.mul(logits, one_hot), dim=1)
        return loss, logits, action_id

    def get_im_embedding(self, cooccurrence_entities):
        """Get the inductive mean representation of the co-occurrence relation.
        cooccurrence_entities: a list that contains the trained entities with the co-occurrence relation.
        return: torch.tensor, representation of the co-occurrence entities.
        """
        entities = self.ent_embs.ent_embs.weight.data[cooccurrence_entities]
        im = torch.mean(entities, dim=0)
        return im

    def update_entity_embedding(self, entity, ims, mu):
        """Update the entity representation with the co-occurrence relations in the last timestamp.
        entity: int, the entity that needs to be updated.
        ims: torch.tensor, [number of co-occurrence, -1], the IM representations of the co-occurrence relations
        mu: update ratio, the hyperparam.
        """
        self.source_entity = self.ent_embs.ent_embs.weight.data[entity]
        self.ent_embs.ent_embs.weight.data[entity] = mu * self.source_entity + (
            1 - mu
        ) * torch.mean(ims, dim=0)

    def entities_embedding_shift(self, entity, im, mu):
        """Prediction shift."""
        self.source_entity = self.ent_embs.ent_embs.weight.data[entity]
        self.ent_embs.ent_embs.weight.data[entity] = (
            mu * self.source_entity + (1 - mu) * im
        )

    def back_entities_embedding(self, entity):
        """Go back after shift ends."""
        self.ent_embs.ent_embs.weight.data[entity] = self.source_entity

    def confidence(
        self,
        query_entities,
        query_relations,
        neighbor_entities,
        pad_mask,
        mode="complex",
    ):
        if mode == "complex":
            rank = query_entities.shape[1] // 2
            lhs = query_entities[:, :rank], query_entities[:, rank:]
            rel = query_relations[:, :rank], query_relations[:, rank:]

            right = neighbor_entities
            right = right[:, :, :rank], right[:, :, rank:]
            s = (lhs[0] * rel[0] - lhs[1] * rel[1]).unsqueeze(1) @ right[0].transpose(
                1, 2
            ) + (lhs[0] * rel[1] + lhs[1] * rel[0]).unsqueeze(1) @ right[1].transpose(
                1, 2
            )
            s = s.squeeze(1).masked_fill(pad_mask, -1e10)
            # s = torch.softmax(s, dim=1)
            s = torch.sigmoid(s)

        elif mode == "tucker":
            x = query_entities.view(-1, 1, self.config["ent_dim"])
            W_mat = torch.mm(
                query_relations, self.W_tk.view(self.config["rel_dim"], -1)
            )
            W_mat = W_mat.view(-1, self.config["ent_dim"], self.config["ent_dim"])
            x = torch.bmm(x, W_mat)
            x = x.view(-1, self.config["ent_dim"])
            x = x.unsqueeze(1)
            s = x @ neighbor_entities.transpose(1, 2)
            s = s.squeeze(1).masked_fill(pad_mask, -1e10)
            # s = torch.softmax(s, dim=1)
            s = torch.sigmoid(s)

        elif mode == "lowfer":
            x = torch.mm(query_entities, self.U_tk) * torch.mm(
                query_relations, self.V_tk
            )  # torch.Size([512, 3000])
            x = x.view(-1, self.o, self.k)  # torch.Size([512, 100, 30])
            x = x.sum(-1)  # torch.Size([512, 100])
            x = torch.mul(
                torch.sign(x), torch.sqrt(torch.abs(x) + 1e-12)
            )  # torch.Size([512, 100])
            x = nn.functional.normalize(x, p=2, dim=-1)  # torch.Size([512, 100])
            x = x.view(-1, self.config["ent_dim"])
            x = x.unsqueeze(1)  # torch.Size([512, 1, 100])
            # neighbor_entities: torch.Size([512, 512, 50])
            s = x @ neighbor_entities.transpose(1, 2)
            s = s.squeeze(1).masked_fill(pad_mask, -1e10)
            # s = torch.softmax(s, dim=1)
            s = torch.sigmoid(s)

        return s
