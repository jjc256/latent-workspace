from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SlotAttention(nn.Module):
    """Locatello et al. (2020) slot attention module.

    Args:
        num_slots: number of output slots (N)
        slot_dim: dimensionality of each slot vector (D)
        input_dim: dimensionality of input features
        num_iters: number of recurrent refinement iterations
        eps: small value for attention normalization stability
    """

    def __init__(
        self,
        num_slots: int = 8,
        slot_dim: int = 256,
        input_dim: int = 256,
        num_iters: int = 5,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iters = num_iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        # Learned slot initialization distribution
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

        self.norm_inputs = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre_ff = nn.LayerNorm(slot_dim)

        self.project_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(input_dim, slot_dim, bias=False)
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, 128),
            nn.ReLU(),
            nn.Linear(128, slot_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.project_k.weight)
        nn.init.xavier_uniform_(self.project_v.weight)
        nn.init.xavier_uniform_(self.project_q.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs: [B, L, D_in] — input feature sequence
        Returns:
            slots: [B, N, slot_dim]
        """
        B, L, _ = inputs.shape
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # [B, L, D]
        v = self.project_v(inputs)  # [B, L, D]

        # Initialize slots from learned Gaussian
        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)  # [B, N, D]

        for _ in range(self.num_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.project_q(slots)  # [B, N, D]

            # Attention weights: normalize over slots (dim=1)
            dots = torch.einsum("bnd,bld->bnl", q, k) * self.scale  # [B, N, L]
            attn = dots.softmax(dim=1) + self.eps  # [B, N, L]
            attn = attn / attn.sum(dim=-1, keepdim=True)  # normalize over inputs

            # Weighted mean of values
            updates = torch.einsum("bnl,bld->bnd", attn, v)  # [B, N, D]

            # GRU update (operates on flat [B*N, D])
            slots = self.gru(
                updates.reshape(B * self.num_slots, self.slot_dim),
                slots_prev.reshape(B * self.num_slots, self.slot_dim),
            ).reshape(B, self.num_slots, self.slot_dim)

            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


class SlotAttentionAutoEncoder(nn.Module):
    """Slot attention with an MLP decoder for reconstruction-based toy training."""

    def __init__(
        self,
        num_slots: int = 8,
        slot_dim: int = 256,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_iters: int = 5,
    ) -> None:
        super().__init__()
        self.encoder = SlotAttention(num_slots, slot_dim, input_dim, num_iters)
        self.decoder = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            inputs: [B, L, D_in]
        Returns:
            slots: [B, N, slot_dim]
            recon: [B, N, D_in] — per-slot reconstructions (mixture-style)
        """
        slots = self.encoder(inputs)           # [B, N, D]
        recon = self.decoder(slots)            # [B, N, D_in]
        return slots, recon
