"""Train WorkspaceUpdateTransformer on synthetic object-absorption sequences.

Task: present T random unit-vector "objects" one at a time; after T steps the
workspace should contain each object in at least one slot.

Saves weights to data/checkpoints/workspace_update.pt (~1-2 hours on MPS).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.workspace.state import D, K_MEM, N_SLOTS
from src.workspace.update import WorkspaceUpdateTransformer


def _default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Batched forward (bypasses WorkspaceState for training)
# ---------------------------------------------------------------------------

def _batched_update(
    model: WorkspaceUpdateTransformer,
    objects: Tensor,       # [B, N, D]
    memory: Tensor,        # [B, K, D]
    input_tokens: Tensor,  # [B, L, D]
) -> tuple[Tensor, Tensor]:
    """Return updated (objects [B,N,D], memory [B,K,D])."""
    B, N, _ = objects.shape
    device = input_tokens.device
    slot_ids = torch.arange(N, device=device)
    mem_ids = torch.arange(K_MEM, device=device)
    ws_tokens = objects + model.slot_embeddings(slot_ids)    # [B, N, D]
    mem_tokens = memory + model.mem_embeddings(mem_ids)      # [B, K, D]
    z = model.input_proj(input_tokens)                       # [B, L, D]
    combined = torch.cat([ws_tokens, mem_tokens, z], dim=1)  # [B, N+K+L, D]
    out = model.transformer(combined)                        # [B, N+K+L, D]
    return out[:, :N, :], out[:, N : N + K_MEM, :]


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _absorption_loss(workspace_objects: Tensor, targets: Tensor) -> Tensor:
    """For each target object, penalize distance to its nearest workspace slot."""
    # workspace_objects: [B, N, D]  targets: [B, T, D]
    dists = torch.cdist(targets, workspace_objects)  # [B, T, N]
    return dists.min(dim=2).values.mean()


def _diversity_loss(workspace_objects: Tensor) -> Tensor:
    """Penalize high pairwise cosine similarity between slots."""
    norms = workspace_objects.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normed = workspace_objects / norms                                   # [B, N, D]
    cosine = torch.bmm(normed, normed.transpose(1, 2))                  # [B, N, N]
    eye = torch.eye(N_SLOTS, device=workspace_objects.device).unsqueeze(0)
    return cosine.pow(2).mul(1.0 - eye).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    steps: int = 50_000,
    batch_size: int = 64,
    seq_len: int = 4,
    lr: float = 1e-3,
    div_weight: float = 0.1,
    persist_weight: float = 0.5,
    log_every: int = 100,
    save_every: int = 5_000,
    ckpt_dir: str = "data/checkpoints",
) -> None:
    device = _default_device()
    print(f"Training workspace update transformer on {device}")
    print(f"  steps={steps}  batch={batch_size}  seq_len={seq_len}")

    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    model = WorkspaceUpdateTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    t0 = time.time()
    running = {"abs": 0.0, "div": 0.0, "per": 0.0, "total": 0.0}

    for step in range(1, steps + 1):
        # --- generate a batch of T-object sequences ---
        targets = F.normalize(torch.randn(batch_size, seq_len, D, device=device), dim=-1)

        # --- run T absorption steps from blank workspace ---
        objects = torch.zeros(batch_size, N_SLOTS, D, device=device)
        memory = torch.zeros(batch_size, K_MEM, D, device=device)

        for t in range(seq_len):
            tok = targets[:, t : t + 1, :]  # [B, 1, D]
            objects, memory = _batched_update(model, objects, memory, tok)

        loss_abs = _absorption_loss(objects, targets)
        loss_div = _diversity_loss(objects)

        # --- persistence: one blank step should preserve absorbed objects ---
        zero_tok = torch.zeros(batch_size, 1, D, device=device)
        objects_p, _ = _batched_update(model, objects.detach(), memory.detach(), zero_tok)
        loss_per = _absorption_loss(objects_p, targets)

        loss = loss_abs + div_weight * loss_div + persist_weight * loss_per

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running["abs"] += loss_abs.item()
        running["div"] += loss_div.item()
        running["per"] += loss_per.item()
        running["total"] += loss.item()

        if step % log_every == 0:
            elapsed = time.time() - t0
            n = log_every
            print(
                f"step {step:6d}/{steps}  "
                f"total={running['total']/n:.4f}  "
                f"abs={running['abs']/n:.4f}  "
                f"div={running['div']/n:.4f}  "
                f"per={running['per']/n:.4f}  "
                f"elapsed={elapsed:.0f}s"
            )
            for k in running:
                running[k] = 0.0

        if step % save_every == 0:
            ckpt = Path(ckpt_dir) / f"workspace_update_step{step}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"  → saved {ckpt}")

    final_ckpt = Path(ckpt_dir) / "workspace_update.pt"
    torch.save(model.state_dict(), final_ckpt)
    print(f"Training complete. Saved to {final_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--div-weight", type=float, default=0.1)
    parser.add_argument("--persist-weight", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--ckpt-dir", default="data/checkpoints")
    args = parser.parse_args()
    train(
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        div_weight=args.div_weight,
        persist_weight=args.persist_weight,
        log_every=args.log_every,
        save_every=args.save_every,
        ckpt_dir=args.ckpt_dir,
    )
