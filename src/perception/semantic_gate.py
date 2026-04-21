from __future__ import annotations

from enum import IntEnum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GateLabel(IntEnum):
    VALID = 0
    LOW_CONFIDENCE = 1
    INVALID = 2


# ~200 common English nouns used as CLIP anchor concepts
_ANCHOR_CONCEPTS = [
    "cat", "dog", "car", "tree", "house", "person", "table", "chair",
    "book", "phone", "computer", "window", "door", "floor", "wall",
    "sky", "cloud", "sun", "moon", "star", "water", "fire", "earth",
    "bird", "fish", "horse", "cow", "sheep", "pig", "chicken",
    "apple", "banana", "orange", "grape", "strawberry", "bread",
    "cup", "plate", "spoon", "fork", "knife", "bottle", "bag",
    "shirt", "pants", "shoe", "hat", "coat", "dress", "glasses",
    "flower", "grass", "rock", "mountain", "river", "ocean", "beach",
    "road", "bridge", "building", "city", "village", "park", "garden",
    "ball", "bicycle", "bus", "train", "airplane", "boat", "truck",
    "key", "lock", "lamp", "clock", "mirror", "camera", "umbrella",
    "box", "basket", "bucket", "shelf", "drawer", "cabinet", "bed",
    "pillow", "blanket", "towel", "soap", "toothbrush", "brush",
    "pencil", "pen", "paper", "notebook", "envelope", "stamp",
    "coin", "money", "card", "ticket", "flag", "sign", "light",
    "wire", "pipe", "rope", "chain", "nail", "screw", "bolt",
    "hammer", "saw", "drill", "wrench", "shovel", "broom", "mop",
    "pot", "pan", "oven", "refrigerator", "microwave", "sink",
    "toilet", "shower", "bathtub", "mirror", "curtain", "carpet",
    "painting", "statue", "vase", "candle", "trophy", "medal",
    "guitar", "piano", "drum", "violin", "trumpet", "microphone",
    "television", "radio", "speaker", "headphones", "keyboard",
    "mouse", "monitor", "printer", "scanner", "charger", "battery",
    "engine", "wheel", "seat", "helmet", "glove", "mask", "belt",
    "wallet", "watch", "ring", "necklace", "earring", "bracelet",
    "tent", "backpack", "suitcase", "map", "compass", "telescope",
    "microscope", "thermometer", "scale", "ruler", "calculator",
    "scissors", "tape", "glue", "needle", "thread", "button",
    "ladder", "staircase", "elevator", "fence", "gate", "bench",
    "swing", "slide", "fountain", "statue", "column", "arch",
    "lantern", "torch", "canteen", "fork", "skillet", "wok",
]


class _GateHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)
        with torch.no_grad():
            self.linear.weight.zero_()
            # Default bias: VALID > LOW_CONFIDENCE > INVALID without training
            self.linear.bias.copy_(torch.tensor([1.0, 0.0, -1.0]))

    def forward(self, signals: Tensor) -> Tensor:
        return self.linear(signals)


class SemanticGate:
    """Lightweight 3-class validity filter over perception outputs.

    Works out of the box with default biases. Call fit() with labeled data to
    improve accuracy.
    """

    def __init__(
        self,
        text_encoder=None,
        vision_encoder=None,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self._text_encoder = text_encoder
        self._vision_encoder = vision_encoder

        self._head = _GateHead().to(device)

        # Normalization stats (updated on fit)
        self._signal_mean = torch.zeros(4, device=device)
        self._signal_std = torch.ones(4, device=device)

        # Cached anchor embeddings [N_anchors, 256], built lazily
        self._anchor_embeddings: Optional[Tensor] = None
        # Projects CLIP text 512-dim to workspace 256-dim
        self._anchor_proj = nn.Linear(512, 256).to(device)
        nn.init.xavier_uniform_(self._anchor_proj.weight)

    # ------------------------------------------------------------------
    # Public prediction API
    # ------------------------------------------------------------------

    def predict(
        self,
        text: Optional[str] = None,
        slots: Optional[Tensor] = None,
    ) -> GateLabel:
        proba = self.predict_proba(text, slots)
        return GateLabel(int(proba.argmax().item()))

    def predict_proba(
        self,
        text: Optional[str] = None,
        slots: Optional[Tensor] = None,
    ) -> Tensor:
        """Return [3] softmax probabilities."""
        features = self._build_feature_vector(text, slots)
        normed = (features - self._signal_mean) / (self._signal_std + 1e-8)
        with torch.no_grad():
            logits = self._head(normed.to(self.device))
        return F.softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _build_feature_vector(
        self,
        text: Optional[str],
        slots: Optional[Tensor],
    ) -> Tensor:
        s1 = self._compute_perplexity(text) if text else -1.0
        s2 = self._compute_clip_confidence(slots) if slots is not None else 0.5
        s3 = self._compute_lang_id_confidence(text) if text else 1.0
        s4 = self._compute_embedding_entropy(slots) if slots is not None else 0.5
        return torch.tensor([s1, s2, s3, s4], dtype=torch.float32)

    def _compute_perplexity(self, text: str) -> float:
        """LLM cross-entropy loss → perplexity. Falls back to -1.0."""
        if self._text_encoder is None or self._text_encoder._model is None:
            return -1.0
        try:
            enc = self._text_encoder._tokenizer(text, return_tensors="pt").to(self.device)
            input_ids = enc["input_ids"]
            with torch.no_grad():
                out = self._text_encoder._model(**enc, labels=input_ids)
            return float(out.loss.exp().item())
        except Exception:
            return -1.0

    def _compute_clip_confidence(self, slots: Tensor) -> float:
        """Max cosine similarity of slot vectors to anchor concept embeddings."""
        if self._vision_encoder is None or self._vision_encoder._clip_model is None:
            return 0.5
        try:
            anchors = self._get_anchor_embeddings()  # [N, 256]
            # slots: [B, 8, 256] or [8, 256]
            s = slots.to(self.device).float()
            if s.dim() == 3:
                s = s[0]  # take first batch item: [8, 256]
            s = F.normalize(s, dim=-1)            # [8, 256]
            a = F.normalize(anchors, dim=-1)      # [N, 256]
            sims = torch.einsum("nd,md->nm", s, a)  # [8, N]
            return float(sims.max().item())
        except Exception:
            return 0.5

    def _get_anchor_embeddings(self) -> Tensor:
        """Lazily compute and cache anchor concept embeddings at 256-dim."""
        if self._anchor_embeddings is not None:
            return self._anchor_embeddings
        import open_clip

        tokenizer = open_clip.get_tokenizer(self._vision_encoder.MODEL_NAME)
        tokens = tokenizer(_ANCHOR_CONCEPTS).to(self.device)
        with torch.no_grad():
            text_feats = self._vision_encoder._clip_model.encode_text(tokens)  # [N, 512]
            projected = self._anchor_proj(text_feats.float())  # [N, 256]
        self._anchor_embeddings = projected
        return self._anchor_embeddings

    def _compute_lang_id_confidence(self, text: str) -> float:
        """P(English) from langdetect. Falls back to 1.0 if not installed."""
        try:
            from langdetect import detect_langs

            results = detect_langs(text)
            for r in results:
                if r.lang == "en":
                    return float(r.prob)
            return 0.5
        except ImportError:
            return 1.0
        except Exception:
            return 0.5

    def _compute_embedding_entropy(self, slots: Tensor) -> float:
        """Shannon entropy of softmax over slot norms. High = diffuse/noisy input."""
        s = slots.to(self.device).float()
        if s.dim() == 3:
            s = s[0]  # [N, D]
        norms = s.norm(dim=-1)          # [N]
        weights = F.softmax(norms, dim=0)
        entropy = -(weights * (weights + 1e-8).log()).sum()
        return float(entropy.item())

    # ------------------------------------------------------------------
    # Training API
    # ------------------------------------------------------------------

    def fit(self, features: list[dict], labels: list[int]) -> None:
        """Train the gate head on labeled examples.

        Each feature dict must have keys: perplexity, clip_confidence,
        lang_confidence, embedding_entropy.
        """
        xs = torch.tensor(
            [
                [
                    f["perplexity"],
                    f["clip_confidence"],
                    f["lang_confidence"],
                    f["embedding_entropy"],
                ]
                for f in features
            ],
            dtype=torch.float32,
        )
        ys = torch.tensor(labels, dtype=torch.long)

        # Fit normalization stats
        self._signal_mean = xs.mean(dim=0).to(self.device)
        self._signal_std = xs.std(dim=0).to(self.device)

        xs_norm = (xs - self._signal_mean.cpu()) / (self._signal_std.cpu() + 1e-8)
        xs_norm = xs_norm.to(self.device)
        ys = ys.to(self.device)

        optimizer = torch.optim.Adam(self._head.parameters(), lr=1e-2)
        self._head.train()
        for _ in range(200):
            optimizer.zero_grad()
            logits = self._head(xs_norm)
            loss = F.cross_entropy(logits, ys)
            loss.backward()
            optimizer.step()
        self._head.eval()

    def save(self, path: str) -> None:
        torch.save(
            {
                "head": self._head.state_dict(),
                "mean": self._signal_mean,
                "std": self._signal_std,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self._head.load_state_dict(ckpt["head"])
        self._signal_mean = ckpt["mean"].to(self.device)
        self._signal_std = ckpt["std"].to(self.device)
