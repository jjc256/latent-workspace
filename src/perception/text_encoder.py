from __future__ import annotations

import json
import threading
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class ModelNotAvailableError(RuntimeError):
    """Raised when Qwen2.5-3B weights cannot be loaded."""


class TextProjection(nn.Module):
    def __init__(self, in_dim: int = 2048, out_dim: int = 256) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


_ENTITY_PROMPT = """\
<|im_start|>system
You are an entity extractor. Respond ONLY with valid JSON, no prose.
<|im_end|>
<|im_start|>user
Extract named entities and the overall intent from the text below.
Text: {text}
Output format: {{"entities": ["entity1", "entity2"], "intent": "one sentence"}}
<|im_end|>
<|im_start|>assistant
"""


class TextEncoder:
    MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
    HIDDEN_DIM = 2048

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: Optional[str] = None,
        stub_mode: bool = False,
    ) -> None:
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self._model_id = model_id
        self._stub_mode = stub_mode

        self._model = None
        self._tokenizer = None
        self._proj: Optional[TextProjection] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            if self._stub_mode:
                self._proj = TextProjection(self.HIDDEN_DIM, 256).to(self.device)
                return
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._model_id, trust_remote_code=True
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._model_id,
                    torch_dtype=torch.float16,
                    device_map=self.device,
                    trust_remote_code=True,
                )
                self._model.eval()
            except (OSError, EnvironmentError) as e:
                raise ModelNotAvailableError(
                    f"Could not load {self._model_id}. "
                    f"Download it with: huggingface-cli download {self._model_id}"
                ) from e
            self._proj = TextProjection(self.HIDDEN_DIM, 256).to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> Tensor:
        """Return [num_entities, 256] workspace-compatible vectors."""
        self._ensure_loaded()

        if self._stub_mode:
            # Return random projections without a real model
            dummy = torch.randn(1, self.HIDDEN_DIM, device=self.device)
            return self._proj(dummy)  # [1, 256]

        entities = self._extract_entities(text)
        return self._embed_entities(text, entities)

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _extract_entities(self, text: str) -> list[str]:
        """Call LLM to extract entities; fall back to [text] on any failure."""
        prompt = _ENTITY_PROMPT.format(text=text)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens
        new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        raw = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        try:
            parsed = json.loads(raw)
            entities = parsed.get("entities", [])
            if isinstance(entities, list) and entities:
                return [str(e) for e in entities]
        except (json.JSONDecodeError, AttributeError):
            pass
        return [text]

    def _embed_entities(self, text: str, entities: list[str]) -> Tensor:
        """Return [num_entities, 256] by extracting hidden states at entity spans."""
        enc = self._tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = enc["input_ids"]  # [1, L]
        with torch.no_grad():
            out = self._model(
                **enc,
                output_hidden_states=True,
            )
        # last_hidden_state: [1, L, H]
        hidden = out.hidden_states[-1][0]  # [L, H]

        spans = self._find_entity_token_spans(
            input_ids[0].tolist(), entities
        )
        vecs = []
        for start, end in spans:
            vec = hidden[start:end].mean(dim=0)  # [H]
            vecs.append(vec)

        if not vecs:
            vecs.append(hidden.mean(dim=0))

        stacked = torch.stack(vecs, dim=0)  # [E, H]
        return self._proj(stacked.to(torch.float32))  # [E, 256]

    def _find_entity_token_spans(
        self, input_ids: list[int], entities: list[str]
    ) -> list[tuple[int, int]]:
        """Sliding-window search for entity token subsequences."""
        spans = []
        for entity in entities:
            entity_ids = self._tokenizer.encode(entity, add_special_tokens=False)
            n = len(entity_ids)
            found = False
            for i in range(len(input_ids) - n + 1):
                if input_ids[i : i + n] == entity_ids:
                    spans.append((i, i + n))
                    found = True
                    break
            if not found:
                # Fall back to the full sequence span
                spans.append((0, len(input_ids)))
        return spans
