from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.workspace.state import WorkspaceState, N_SLOTS, D
from src.memory.episodic import EpisodicMemory


def _default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CrossAttentionRetriever(nn.Module):
    def __init__(
        self,
        d_model: int = D,
        nhead: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(
        self,
        workspace_objects: Tensor,  # [N, D]
        retrieved_objects: Tensor,  # [k*N_r, D]
    ) -> Tensor:                    # [N, D]
        q = self.norm_q(workspace_objects).unsqueeze(0)   # [1, N, D]
        kv = self.norm_kv(retrieved_objects).unsqueeze(0) # [1, k*N_r, D]
        attended, _ = self.attn(q, kv, kv)                # [1, N, D]
        x = workspace_objects + attended.squeeze(0)        # residual
        x = x + self.ff(self.norm_ff(x))                  # FFN residual
        return x


class MemoryRetriever:
    def __init__(
        self,
        episodic: EpisodicMemory,
        device: str | None = None,
    ) -> None:
        self.episodic = episodic
        self.device = device or _default_device()
        self.cross_attn = CrossAttentionRetriever().to(self.device)

    def retrieve(
        self,
        workspace: WorkspaceState,
        goal_embedding: Tensor | None = None,
        k: int = 5,
    ) -> WorkspaceState:
        query_vec = self._build_query_embedding(workspace, goal_embedding)
        results = self.episodic.query(query_vec, n_results=k)

        if not results:
            return workspace.clone()

        retrieved_stacked = torch.cat(
            [r["state"].objects.to(self.device) for r in results], dim=0
        )  # [k*N_r, D]

        updated_objects = self.cross_attn(
            workspace.objects.to(self.device),
            retrieved_stacked,
        )

        new_ws = workspace.clone()
        new_ws.objects = updated_objects
        return new_ws

    def _build_query_embedding(
        self,
        workspace: WorkspaceState,
        goal_embedding: Tensor | None,
    ) -> Tensor:
        objects = workspace.objects.to(self.device)
        if goal_embedding is not None:
            goal = goal_embedding.to(self.device)
            goal_broadcast = goal.unsqueeze(0).expand(N_SLOTS, -1)
            combined = torch.cat([objects, goal_broadcast], dim=0)
            return combined.mean(dim=0)
        return objects.mean(dim=0)

    def save(self, path: str) -> None:
        torch.save(self.cross_attn.state_dict(), path)

    def load(self, path: str) -> None:
        self.cross_attn.load_state_dict(
            torch.load(path, map_location=self.device)
        )
