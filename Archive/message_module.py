"""
message_module.py
=================
MessageAggregator
-----------------
Implements the communication protocol between agents.

Each agent receives an aggregated message from its k nearest neighbours
(by Euclidean distance in grid space).  Aggregation is mean-pooling.

This is a *learned* communication scheme:
  - Agents output a message vector each step (model.msg_encoder)
  - MessageAggregator routes those vectors to the right recipients
  - The aggregated message is fed back into the model next step

This pattern is inspired by CommNet / TarMAC / QMIX message passing.
"""

import numpy as np
from typing import Dict, List, Tuple


class MessageAggregator:
    """
    Routes messages between agents using k-nearest-neighbour topology.

    Parameters
    ----------
    msg_dim : int  – dimension of message vectors
    k       : int  – number of nearest neighbours to communicate with
    """

    def __init__(self, msg_dim: int = 16, k: int = 4):
        self.msg_dim = msg_dim
        self.k       = k

    # ─────────────────────────────────────────
    def aggregate(self,
                  agent_ids:  List[str],
                  positions:  Dict[str, Tuple[int, int]],
                  msg_state:  Dict[str, np.ndarray],
                  ) -> Dict[str, np.ndarray]:
        """
        For each agent, compute the mean message from its k nearest neighbours.

        Parameters
        ----------
        agent_ids : list of active agent ids
        positions : {agent_id: (x, y)}
        msg_state : {agent_id: np.ndarray(msg_dim)} – messages from last step

        Returns
        -------
        agg_msgs : {agent_id: np.ndarray(msg_dim)}
        """
        n = len(agent_ids)
        agg_msgs: Dict[str, np.ndarray] = {}

        if n <= 1:
            # No neighbours → zero message
            for aid in agent_ids:
                agg_msgs[aid] = np.zeros(self.msg_dim, dtype=np.float32)
            return agg_msgs

        # Build position array (fall back to zeros if missing)
        pos_arr = np.array([positions.get(aid, (0, 0)) for aid in agent_ids],
                           dtype=np.float32)                  # (N, 2)

        # Build message matrix
        msg_arr = np.stack([msg_state.get(aid, np.zeros(self.msg_dim, dtype=np.float32))
                            for aid in agent_ids])             # (N, msg_dim)

        k_eff = min(self.k, n - 1)

        for i, aid in enumerate(agent_ids):
            # Squared Euclidean distances to all other agents
            diffs  = pos_arr - pos_arr[i]                     # (N, 2)
            dists  = np.sum(diffs ** 2, axis=1)               # (N,)
            dists[i] = np.inf                                  # exclude self

            # k nearest neighbour indices
            nbr_idx = np.argpartition(dists, k_eff)[:k_eff]

            # Mean-pool their messages
            agg_msgs[aid] = msg_arr[nbr_idx].mean(axis=0).astype(np.float32)

        return agg_msgs

    # ─────────────────────────────────────────
    def build_comm_graph(self,
                         agent_ids: List[str],
                         positions: Dict[str, Tuple[int, int]],
                         ) -> Dict[str, List[str]]:
        """
        (Optional utility) Build explicit neighbour lists for visualisation.

        Returns
        -------
        graph : {agent_id: [neighbour_id, ...]}
        """
        n = len(agent_ids)
        if n <= 1:
            return {aid: [] for aid in agent_ids}

        pos_arr = np.array([positions.get(aid, (0, 0)) for aid in agent_ids],
                           dtype=np.float32)
        k_eff   = min(self.k, n - 1)
        graph: Dict[str, List[str]] = {}

        for i, aid in enumerate(agent_ids):
            dists    = np.sum((pos_arr - pos_arr[i]) ** 2, axis=1)
            dists[i] = np.inf
            nbr_idx  = np.argpartition(dists, k_eff)[:k_eff]
            graph[aid] = [agent_ids[j] for j in nbr_idx]

        return graph
