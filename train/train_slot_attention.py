"""Train SlotAttention on synthetic colored-shape images.

Saves weights to data/checkpoints/slot_attention.pt (~2 hours on MPS).
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.workspace.slot_attention import SlotAttention

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

_COLORS = [
    (220, 50, 50),
    (50, 180, 50),
    (50, 80, 220),
    (220, 200, 50),
    (200, 50, 200),
    (50, 200, 200),
    (220, 130, 50),
    (130, 50, 220),
]


def _generate_image(img_size: int = 64, rng: np.random.RandomState | None = None) -> Image.Image:
    if rng is None:
        rng = np.random.RandomState()
    num_shapes = rng.randint(3, 6)
    img = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    color_indices = rng.choice(len(_COLORS), num_shapes, replace=False)
    placed: list[list[int]] = []
    min_sz, max_sz = 8, 20
    for ci in color_indices:
        for _ in range(50):
            sz = int(rng.randint(min_sz, max_sz + 1))
            x = int(rng.randint(0, img_size - sz))
            y = int(rng.randint(0, img_size - sz))
            box = [x, y, x + sz, y + sz]
            if all(
                box[2] <= pb[0] or box[0] >= pb[2] or box[3] <= pb[1] or box[1] >= pb[3]
                for pb in placed
            ):
                color = _COLORS[ci]
                if rng.rand() > 0.5:
                    draw.ellipse(box, fill=color)
                else:
                    draw.rectangle(box, fill=color)
                placed.append(box)
                break
    return img


class ShapeDataset(Dataset):
    def __init__(self, size: int = 50_000, img_size: int = 64, seed: int = 0) -> None:
        self.size = size
        self.img_size = img_size
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, _idx: int) -> Tensor:
        img = _generate_image(self.img_size, self.rng)
        arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]
        return torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]


# ---------------------------------------------------------------------------
# Encoder: [B, 3, 64, 64] → [B, 64, 256]
# ---------------------------------------------------------------------------

class SceneEncoder(nn.Module):
    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),   # → [B, 32, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),  # → [B, 64, 16, 16]
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2), # → [B, 128, 8, 8]
            nn.ReLU(),
            nn.Conv2d(128, out_dim, 1),                  # → [B, 256, 8, 8]
        )
        self.pos_emb = nn.Embedding(64, out_dim)  # 8×8 = 64 positions

    def forward(self, x: Tensor) -> Tensor:  # [B, 3, 64, 64] → [B, 64, 256]
        feats = self.conv(x)                     # [B, 256, 8, 8]
        B, C, H, W = feats.shape
        feats = feats.reshape(B, C, H * W).permute(0, 2, 1)  # [B, 64, 256]
        pos_ids = torch.arange(H * W, device=x.device)
        feats = feats + self.pos_emb(pos_ids)
        return feats


# ---------------------------------------------------------------------------
# Broadcast decoder: [B, N, D] → [B, 64, D], masks [B, N, 64]
# ---------------------------------------------------------------------------

class BroadcastDecoder(nn.Module):
    def __init__(self, slot_dim: int = 256, out_dim: int = 256, n_positions: int = 64, pos_dim: int = 32) -> None:
        super().__init__()
        self.n_positions = n_positions
        self.pos_emb = nn.Embedding(n_positions, pos_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim + pos_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim + 1),  # +1 for alpha mask
        )

    def forward(self, slots: Tensor) -> tuple[Tensor, Tensor]:  # [B,N,D] → ([B,L,D],[B,N,L])
        B, N, D = slots.shape
        L = self.n_positions
        pos_ids = torch.arange(L, device=slots.device)
        pos_enc = self.pos_emb(pos_ids)  # [L, pos_dim]

        slots_exp = slots.unsqueeze(2).expand(-1, -1, L, -1)                         # [B, N, L, D]
        pos_exp = pos_enc.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)             # [B, N, L, pos_dim]
        x = torch.cat([slots_exp, pos_exp], dim=-1)                                  # [B, N, L, D+pos]

        out = self.mlp(x.reshape(B * N * L, -1)).reshape(B, N, L, -1)               # [B, N, L, D+1]
        recon = out[..., :-1]          # [B, N, L, D]
        alpha = out[..., -1:]          # [B, N, L, 1]
        masks = alpha.softmax(dim=1)   # softmax over N slots → [B, N, L, 1]

        final_recon = (masks * recon).sum(dim=1)  # [B, L, D]
        return final_recon, masks.squeeze(-1)      # [B, L, D], [B, N, L]


# ---------------------------------------------------------------------------
# Full training model
# ---------------------------------------------------------------------------

class SlotAttentionTrainer(nn.Module):
    def __init__(self, num_slots: int = 8, slot_dim: int = 256) -> None:
        super().__init__()
        self.encoder = SceneEncoder(out_dim=slot_dim)
        self.slot_attn = SlotAttention(num_slots=num_slots, slot_dim=slot_dim, input_dim=slot_dim)
        self.decoder = BroadcastDecoder(slot_dim=slot_dim, out_dim=slot_dim)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.encoder(images)               # [B, 64, 256]
        slots = self.slot_attn(features)              # [B, N, 256]
        recon, masks = self.decoder(slots)            # [B, 64, 256], [B, N, 64]
        return recon, features, masks


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(
    steps: int = 50_000,
    batch_size: int = 32,
    lr: float = 4e-4,
    log_every: int = 100,
    save_every: int = 5_000,
    ckpt_dir: str = "data/checkpoints",
    num_workers: int = 0,
) -> None:
    device = _default_device()
    print(f"Training slot attention on {device}")

    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    dataset = ShapeDataset(size=max(steps * batch_size, 50_000))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    model = SlotAttentionTrainer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    data_iter = iter(loader)
    t0 = time.time()
    running_loss = 0.0

    for step in range(1, steps + 1):
        try:
            images = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images = next(data_iter)

        images = images.to(device)
        recon, features, _masks = model(images)

        loss = F.mse_loss(recon, features.detach())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        if step % log_every == 0:
            elapsed = time.time() - t0
            avg = running_loss / log_every
            print(f"step {step:6d}/{steps}  loss={avg:.4f}  elapsed={elapsed:.0f}s")
            running_loss = 0.0

        if step % save_every == 0:
            ckpt = Path(ckpt_dir) / f"slot_attention_step{step}.pt"
            torch.save(model.slot_attn.state_dict(), ckpt)
            print(f"  → saved {ckpt}")

    final_ckpt = Path(ckpt_dir) / "slot_attention.pt"
    torch.save(model.slot_attn.state_dict(), final_ckpt)
    print(f"Training complete. Saved to {final_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--ckpt-dir", default="data/checkpoints")
    args = parser.parse_args()
    train(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        log_every=args.log_every,
        save_every=args.save_every,
        ckpt_dir=args.ckpt_dir,
    )
