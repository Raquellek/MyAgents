import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class StrategyMessageGRUExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space,
        features_dim=256,
        hidden_dim=256,
        strategy_count=6,
        strategy_dim=32,
        message_dim=32
    ):
        super().__init__(observation_space, features_dim)

        obs_shape = observation_space.shape

        input_dim = 1
        for x in obs_shape:
            input_dim *= x

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.strategy_embedding = nn.Embedding(
            strategy_count,
            strategy_dim
        )

        self.message_head = nn.Sequential(
            nn.Linear(hidden_dim, message_dim),
            nn.Tanh()
        )

        self.final = nn.Sequential(
            nn.Linear(
                hidden_dim + strategy_dim + message_dim,
                features_dim
            ),
            nn.ReLU()
        )

        self.current_strategy_id = 0

    def set_strategy(self, strategy_id):
        self.current_strategy_id = int(strategy_id)

    def forward(self, observations):

        batch_size = observations.shape[0]

        encoded = self.encoder(observations)

        seq = encoded.unsqueeze(1)

        gru_out, _ = self.gru(seq)

        memory = gru_out[:, -1, :]

        message = self.message_head(memory)

        strategy_ids = torch.full(
            (batch_size,),
            self.current_strategy_id,
            dtype=torch.long,
            device=observations.device
        )

        strategy_vec = self.strategy_embedding(strategy_ids)

        combined = torch.cat(
            [memory, strategy_vec, message],
            dim=1
        )

        features = self.final(combined)

        return features


class CoalitionGRUExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        features_dim=512,
        hidden_dim=256,
        message_dim=64,
    ):
        super().__init__(observation_space, features_dim)

        max_agents, agent_feature_size = observation_space.shape
        self.max_agents = max_agents
        self.agent_feature_size = agent_feature_size

        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_feature_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.coalition_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.message_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, message_dim),
            nn.Tanh(),
        )

        self.final = nn.Sequential(
            nn.Linear(hidden_dim * 2 + message_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        encoded_agents = self.agent_encoder(observations)
        gru_out, _ = self.coalition_gru(encoded_agents)

        alive_mask = observations[:, :, -13] > 0.5
        mask = alive_mask.float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (gru_out * mask).sum(dim=1) / denom

        message = self.message_head(pooled)
        return self.final(torch.cat([pooled, message], dim=1))


class CoalitionSlotExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        features_dim=512,
        agent_hidden_dim=32,
        global_hidden_dim=128,
    ):
        super().__init__(observation_space, features_dim)

        max_agents, agent_feature_size = observation_space.shape
        self.max_agents = max_agents
        self.agent_feature_size = agent_feature_size

        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_feature_size, agent_hidden_dim),
            nn.ReLU(),
            nn.Linear(agent_hidden_dim, agent_hidden_dim),
            nn.ReLU(),
        )

        flat_agent_dim = max_agents * agent_hidden_dim
        self.global_head = nn.Sequential(
            nn.Linear(flat_agent_dim, global_hidden_dim),
            nn.ReLU(),
        )

        self.final = nn.Sequential(
            nn.Linear(flat_agent_dim + global_hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        encoded_agents = self.agent_encoder(observations)

        alive_mask = observations[:, :, -13] > 0.5
        encoded_agents = encoded_agents * alive_mask.float().unsqueeze(-1)

        flat_agents = encoded_agents.flatten(start_dim=1)
        global_features = self.global_head(flat_agents)

        return self.final(torch.cat([flat_agents, global_features], dim=1))
