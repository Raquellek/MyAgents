"""
model.py
========
MessageA2CNetwork
-----------------
Architecture:
  ┌─────────┐   ┌──────────┐
  │  obs    │   │ agg_msg  │      ← aggregated neighbour messages
  └────┬────┘   └────┬─────┘
       │              │
       └──── cat ─────┘
              │
          Linear(hidden)
          ReLU
          Linear(hidden)
          ReLU
          ┌─────────┬──────────┬──────────┐
       Actor     Critic    Message
      (logits)  (value)   encoder
                           │
                      new_msg  →  broadcast to neighbours next step
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class MessageA2CNetwork(nn.Module):
    """
    Shared-parameter Actor-Critic with message passing.

    Parameters
    ----------
    obs_shape  : tuple – shape of a *flattened* observation
    n_actions  : int
    msg_dim    : int   – dimension of inter-agent message vectors
    hidden_dim : int   – MLP hidden size
    """

    def __init__(self,
                 obs_shape:  tuple,
                 n_actions:  int,
                 msg_dim:    int = 16,
                 hidden_dim: int = 256):
        super().__init__()
        self.msg_dim = msg_dim

        # Flat obs dimension
        obs_dim = int(np.prod(obs_shape))

        # ── Shared encoder (obs + aggregated message) ─────────────────
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + msg_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # ── Actor head ────────────────────────────────────────────────
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

        # ── Critic head ───────────────────────────────────────────────
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # ── Message encoder head ──────────────────────────────────────
        # Produces the message this agent will send to its neighbours
        self.msg_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, msg_dim),
            nn.Tanh(),          # bounded messages in [-1, 1]
        )

        # ── Weight initialisation ─────────────────────────────────────
        self._init_weights()

    # ─────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # smaller init for output heads
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)
        nn.init.orthogonal_(self.msg_encoder[-2].weight, gain=0.01)

    # ─────────────────────────────────────────
    def forward(self,
                obs: torch.Tensor,
                agg_msg: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        obs     : (N, obs_dim)   – batch of agent observations
        agg_msg : (N, msg_dim)   – aggregated neighbour messages

        Returns
        -------
        logits   : (N, n_actions) – unnormalised action probabilities
        value    : (N, 1)         – state value estimate
        new_msg  : (N, msg_dim)   – message to broadcast to neighbours
        """
        obs_flat = obs.view(obs.shape[0], -1)           # ensure flat
        x        = torch.cat([obs_flat, agg_msg], dim=-1)
        hidden   = self.encoder(x)

        logits  = self.actor_head(hidden)
        value   = self.critic_head(hidden)
        new_msg = self.msg_encoder(hidden)

        return logits, value, new_msg
