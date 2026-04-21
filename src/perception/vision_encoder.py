from __future__ import annotations

import threading
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from workspace.slot_attention import SlotAttention


class PatchProjection(nn.Module):
    def __init__(self, in_dim: int = 1024, out_dim: int = 256) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class VisionEncoder:
    MODEL_NAME = "ViT-L-14"
    PRETRAINED = "openai"
    PATCH_DIM = 1024
    N_OBJECT_SLOTS = 8

    def __init__(self, device: Optional[str] = None) -> None:
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device

        self._clip_model = None
        self._preprocess = None
        self._hook_handle = None
        self._patch_features: Optional[Tensor] = None
        self._lock = threading.Lock()

        # Lightweight modules — no download required
        self.slot_attn = SlotAttention(
            num_slots=self.N_OBJECT_SLOTS,
            slot_dim=self.PATCH_DIM,
            input_dim=self.PATCH_DIM,
        ).to(device)
        self.proj = PatchProjection(self.PATCH_DIM, 256).to(device)

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._clip_model is not None:
                return
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                self.MODEL_NAME, pretrained=self.PRETRAINED
            )
            model = model.to(self.device)
            model.eval()
            self._clip_model = model
            self._preprocess = preprocess
            self._register_hook()

    def _register_hook(self) -> None:
        resblocks = self._clip_model.visual.transformer.resblocks
        penultimate = resblocks[-2]

        def hook_fn(module, input, output):
            # open_clip uses sequence-first layout: [L+1, B, D]
            # Strip CLS token and transpose to [B, L, D]
            self._patch_features = output[1:, :, :].permute(1, 0, 2).contiguous()

        self._hook_handle = penultimate.register_forward_hook(hook_fn)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, image) -> Tensor:
        """Return [B, 8, 256] object slot tensors from an image."""
        self._ensure_loaded()
        patches = self._extract_patch_features(image)  # [B, 196, 1024]
        slots = self.slot_attn(patches)                  # [B, 8, 1024]
        return self.proj(slots)                          # [B, 8, 256]

    def encode_raw(self, image) -> Tensor:
        """Return [B, 8, 1024] pre-projection slots (used by SemanticGate)."""
        self._ensure_loaded()
        patches = self._extract_patch_features(image)  # [B, 196, 1024]
        return self.slot_attn(patches)                  # [B, 8, 1024]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_patch_features(self, image) -> Tensor:
        """Run CLIP forward pass; return [B, 196, 1024] via registered hook."""
        import PIL.Image

        if isinstance(image, PIL.Image.Image):
            tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        else:
            tensor = image.to(self.device)
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)

        self._patch_features = None
        with torch.no_grad():
            self._clip_model.encode_image(tensor)

        assert self._patch_features is not None, "Hook did not fire — check resblocks layout"
        return self._patch_features
