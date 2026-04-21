from __future__ import annotations

import sys
import os
import importlib
from unittest.mock import MagicMock, patch

import pytest
import torch

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from perception.text_encoder import TextEncoder, TextProjection, ModelNotAvailableError
from perception.vision_encoder import VisionEncoder, PatchProjection
from perception.semantic_gate import SemanticGate, GateLabel, _GateHead
from workspace.slot_attention import SlotAttention


# ===========================================================================
# TextEncoder tests
# ===========================================================================


def test_text_projection_shape():
    proj = TextProjection(3584, 256)
    x = torch.randn(5, 3584)
    out = proj(x)
    assert out.shape == (5, 256)


def test_text_encoder_lazy_init():
    enc = TextEncoder(device="cpu")
    assert enc._model is None
    assert enc._tokenizer is None
    assert enc._proj is None


def test_text_encoder_raises_when_unavailable():
    with patch(
        "perception.text_encoder.AutoModelForCausalLM" if False else
        "transformers.AutoModelForCausalLM.from_pretrained",
        side_effect=OSError("not found"),
    ):
        enc = TextEncoder(device="cpu")
        # Patch inside _ensure_loaded
        import transformers
        original = transformers.AutoModelForCausalLM.from_pretrained

        def raise_os(*args, **kwargs):
            raise OSError("not found")

        transformers.AutoModelForCausalLM.from_pretrained = raise_os
        try:
            with pytest.raises(ModelNotAvailableError):
                enc._ensure_loaded()
        finally:
            transformers.AutoModelForCausalLM.from_pretrained = original


def test_entity_span_finding():
    enc = TextEncoder(device="cpu", stub_mode=True)
    enc._ensure_loaded()  # loads proj only

    # Build a simple mock tokenizer that maps chars to ints
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda s, **kw: [ord(c) for c in s]
    enc._tokenizer = mock_tokenizer

    input_ids = [ord(c) for c in "hello world"]
    entities = ["world"]
    spans = enc._find_entity_token_spans(input_ids, entities)
    assert len(spans) == 1
    start, end = spans[0]
    assert input_ids[start:end] == [ord(c) for c in "world"]


def test_json_fallback():
    enc = TextEncoder(device="cpu", stub_mode=True)
    enc._ensure_loaded()

    bad_json = "not valid json {{"

    mock_model = MagicMock()
    mock_model.generate.return_value = torch.zeros(1, 6, dtype=torch.long)

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.decode.return_value = bad_json
    # Make tokenizer(prompt, ...) return a batch-encoding-like object with .to()
    fake_encoding = MagicMock()
    fake_encoding.__getitem__ = lambda self, k: torch.zeros(1, 3, dtype=torch.long)
    fake_encoding.to.return_value = fake_encoding
    mock_tokenizer.return_value = fake_encoding

    enc._model = mock_model
    enc._tokenizer = mock_tokenizer

    result = enc._extract_entities("test text")
    assert isinstance(result, list)
    assert len(result) == 1


# ===========================================================================
# VisionEncoder tests
# ===========================================================================


def test_patch_projection_shape():
    proj = PatchProjection(1024, 256)
    x = torch.randn(2, 8, 1024)
    out = proj(x)
    assert out.shape == (2, 8, 256)


def test_vision_encoder_lazy_init():
    enc = VisionEncoder(device="cpu")
    assert enc._clip_model is None


def test_slot_attention_with_patch_dims():
    attn = SlotAttention(num_slots=8, slot_dim=1024, input_dim=1024)
    x = torch.randn(1, 196, 1024)
    with torch.no_grad():
        out = attn(x)
    assert out.shape == (1, 8, 1024)


def test_vision_encoder_encode_with_mock():
    enc = VisionEncoder(device="cpu")

    # Build a fake CLIP model whose encode_image triggers the hook
    fake_clip = MagicMock()

    # Simulate the hook firing: encode_image sets _patch_features
    def fake_encode_image(tensor):
        enc._patch_features = torch.randn(tensor.shape[0], 196, 1024)
        return torch.randn(tensor.shape[0], 512)

    fake_clip.encode_image.side_effect = fake_encode_image

    fake_preprocess = MagicMock()
    fake_preprocess.return_value = torch.randn(3, 224, 224)

    enc._clip_model = fake_clip
    enc._preprocess = fake_preprocess
    # No hook registration needed — fake_encode_image sets _patch_features directly

    import PIL.Image
    img = PIL.Image.new("RGB", (224, 224))
    with torch.no_grad():
        out = enc.encode(img)

    assert out.shape == (1, 8, 256)


# ===========================================================================
# SemanticGate tests
# ===========================================================================


def test_gate_label_enum():
    assert GateLabel.VALID == 0
    assert GateLabel.LOW_CONFIDENCE == 1
    assert GateLabel.INVALID == 2


def test_gate_default_predicts_valid():
    gate = SemanticGate(device="cpu")
    slots = torch.randn(1, 8, 256)
    result = gate.predict(text="hello", slots=slots)
    assert result == GateLabel.VALID


def test_embedding_entropy_varies():
    gate = SemanticGate(device="cpu")

    # Uniform norms → high entropy
    uniform_slots = torch.ones(1, 8, 256)
    entropy_uniform = gate._compute_embedding_entropy(uniform_slots)

    # Peaked: one slot dominates → low entropy
    peaked_slots = torch.zeros(1, 8, 256)
    peaked_slots[0, 0, :] = 100.0
    entropy_peaked = gate._compute_embedding_entropy(peaked_slots)

    assert entropy_uniform > entropy_peaked


def test_lang_id_fallback():
    gate = SemanticGate(device="cpu")

    # Remove langdetect from sys.modules to simulate ImportError
    saved = sys.modules.pop("langdetect", None)
    try:
        # Reload the module so the import path is hit fresh
        import importlib
        import perception.semantic_gate as sg_mod
        importlib.reload(sg_mod)
        gate2 = sg_mod.SemanticGate(device="cpu")
        conf = gate2._compute_lang_id_confidence("hello world")
        assert conf == 1.0
    finally:
        if saved is not None:
            sys.modules["langdetect"] = saved


def test_gate_fit_and_predict():
    gate = SemanticGate(device="cpu")

    features = [
        {"perplexity": 10.0, "clip_confidence": 0.8, "lang_confidence": 0.95, "embedding_entropy": 1.5},
        {"perplexity": 500.0, "clip_confidence": 0.1, "lang_confidence": 0.2, "embedding_entropy": 0.1},
        {"perplexity": 50.0, "clip_confidence": 0.5, "lang_confidence": 0.7, "embedding_entropy": 1.0},
        {"perplexity": 10.0, "clip_confidence": 0.9, "lang_confidence": 0.99, "embedding_entropy": 1.8},
        {"perplexity": 800.0, "clip_confidence": 0.05, "lang_confidence": 0.1, "embedding_entropy": 0.05},
        {"perplexity": 20.0, "clip_confidence": 0.7, "lang_confidence": 0.9, "embedding_entropy": 1.6},
        {"perplexity": 300.0, "clip_confidence": 0.3, "lang_confidence": 0.4, "embedding_entropy": 0.3},
        {"perplexity": 15.0, "clip_confidence": 0.85, "lang_confidence": 0.92, "embedding_entropy": 1.7},
        {"perplexity": 600.0, "clip_confidence": 0.08, "lang_confidence": 0.15, "embedding_entropy": 0.08},
        {"perplexity": 25.0, "clip_confidence": 0.6, "lang_confidence": 0.8, "embedding_entropy": 1.4},
    ]
    labels = [0, 2, 1, 0, 2, 0, 1, 0, 2, 0]

    gate.fit(features, labels)
    proba = gate.predict_proba(text="test", slots=torch.randn(1, 8, 256))
    assert proba.shape == (3,)
    assert abs(proba.sum().item() - 1.0) < 1e-5


def test_gate_save_load(tmp_path):
    gate = SemanticGate(device="cpu")

    features = [
        {"perplexity": 10.0, "clip_confidence": 0.8, "lang_confidence": 0.9, "embedding_entropy": 1.5},
        {"perplexity": 500.0, "clip_confidence": 0.1, "lang_confidence": 0.2, "embedding_entropy": 0.1},
    ]
    labels = [0, 2]
    gate.fit(features, labels)

    save_path = str(tmp_path / "gate.pt")
    gate.save(save_path)

    gate2 = SemanticGate(device="cpu")
    gate2.load(save_path)

    # Weights should match
    for p1, p2 in zip(gate._head.parameters(), gate2._head.parameters()):
        assert torch.allclose(p1, p2)
    assert torch.allclose(gate._signal_mean, gate2._signal_mean)
    assert torch.allclose(gate._signal_std, gate2._signal_std)
