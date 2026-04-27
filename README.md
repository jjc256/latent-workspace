# Latent Workspace Intelligence

A local AI system that maintains a persistent structured **workspace** — a continuous latent state of object slots, relations, and goals — rather than thinking purely in token space. Language becomes a read/write interface to this internal state.

Runs entirely on Apple Silicon (MPS backend, 24 GB target). No cloud compute.

> **Status**: work in progress. Phases 0–3 are implemented; Phases 1–2 modules still need training; Phases 4–6 are not started. See [Implementation status](#implementation-status) at the bottom for the full picture.
>
> The deeper architectural rationale is in [`report.md`](report.md).

---

## Architecture overview

```
                    ┌─────────────────────────────────┐
  text/image ──────▶│  Perception Layer               │
                    │  TextEncoder · VisionEncoder    │
                    │  SemanticGate                   │
                    └───────────────┬─────────────────┘
                                    │ object tokens [N, 256]
                                    ▼
                    ┌─────────────────────────────────┐
                    │  Workspace  W_t                 │
                    │  objects [32, 256]              │◀──── EpisodicMemory
                    │  relations · goals · uncertainty│      (ChromaDB)
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │  Cognitive Modules (Phase 4)    │
                    │  Planner · WorldModel           │
                    │  SpatialReasoner · Consistency  │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │  Streaming Controller (Phase 5) │
                    │  H_t hidden state               │
                    │  cognitive loop (20 Hz async)   │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │  Language Output (Phase 5)      │
                    │  Qwen2.5-3B via MLX             │
                    └─────────────────────────────────┘
```

**Key idea:** the workspace persists across turns. The transformer `W_{t+1} = f(W_t, Z_t)` updates it incrementally rather than reconstructing context from a prompt each time.

---

## Memory budget (24 GB Apple Silicon)

| Component | ~VRAM |
|---|---|
| Qwen2.5-3B (4-bit MLX) | 2.5 GB |
| CLIP ViT-L/14 | 1.5 GB |
| Slot attention + workspace transformer | 0.5 GB |
| World model + planner (Phase 4) | 0.2 GB |
| Working tensors | 2 GB |
| OS + other | 8 GB |
| **Total** | **~14.5 GB** |

On a 36 GB machine you can swap Qwen2.5-3B for Qwen2.5-7B (~4 GB at 4-bit) for better Phase 5 output quality.

---

## Setup

```bash
bash setup.sh       # creates conda env 'lwi', installs deps, verifies MPS
conda activate lwi  # all subsequent commands assume this env is active

# One-time: download Qwen2.5-3B (4-bit quantized, ~2.5 GB)
python -m mlx_lm.convert --hf-path Qwen/Qwen2.5-3B-Instruct -q
```

---

## Training (required before Phase 4)

Phases 1 and 2 include modules that need to be trained before the full pipeline is useful. Run these before starting Phase 4. They can run in parallel in separate terminals.

### Slot attention (~2 hours on MPS)

Trains `SlotAttention` on synthetic 64×64 images of 3–5 colored shapes. A CNN encoder produces patch features; a broadcast decoder reconstructs them slot-by-slot with learned alpha masks. MSE reconstruction loss.

```bash
python -m train.train_slot_attention
# options: --steps 50000  --batch-size 32  --lr 4e-4
```

Saves to `data/checkpoints/slot_attention.pt`. Intermediate checkpoints every 5,000 steps.

### Workspace update transformer (~1–2 hours on MPS)

Trains `WorkspaceUpdateTransformer` on synthetic object-absorption sequences. Each sequence presents T=4 random unit vectors one at a time; the loss rewards absorbing each into a distinct slot and keeping the slots diverse across a blank persistence step.

```bash
python -m train.train_workspace_update
# options: --steps 50000  --batch-size 64  --seq-len 4  --lr 1e-3
```

Saves to `data/checkpoints/workspace_update.pt`. Intermediate checkpoints every 5,000 steps.

### Semantic gate (Phase 2)

`SemanticGate` works out of the box with default biases but needs ~500 hand-labeled examples to filter inputs reliably. Three steps:

```bash
# 1. Source 500 (text, image) examples — Flickr30k captions for the valid/low-conf
#    buckets, synthetic gibberish/non-English for invalid. Writes
#    data/gate_data/unlabeled.jsonl with label=null on every record.
python -m train.seed_gate_data

# 2. Hand-label them via a local web UI (~30 min). Opens http://127.0.0.1:8000;
#    press 1/2/3 to label and arrow keys to navigate. Writes through to the JSONL
#    on every label change.
python -m train.label_gate_ui
```

**TODO (not yet implemented):** `train/fit_gate.py` — load the labeled JSONL, run each example through `TextEncoder` + `VisionEncoder` to compute the 4 feature signals (perplexity, CLIP confidence, language-ID, embedding entropy), then call `gate.fit(features, labels)` and `gate.save("data/checkpoints/semantic_gate.pt")`.

Not required for Phase 4 testing — the gate's default biases let inputs through with sensible class probabilities — but needed for the full pipeline.

---

## Running tests

```bash
python -m pytest tests/ -v
```

| Test file | What it covers |
|---|---|
| `tests/test_workspace.py` | WorkspaceState, SlotAttention, WorkspaceUpdateTransformer |
| `tests/test_perception.py` | TextEncoder, VisionEncoder, SemanticGate |
| `tests/test_memory.py` | EpisodicMemory, MemoryRetriever, 1,000-episode milestone |

---

## File structure

```
latent-workspace/
├── src/
│   ├── workspace/
│   │   ├── state.py          # WorkspaceState dataclass (objects, relations, goals, uncertainty)
│   │   ├── slot_attention.py # Locatello et al. slot attention + autoencoder wrapper
│   │   └── update.py         # WorkspaceUpdateTransformer (4-layer, ~20M params)
│   ├── perception/
│   │   ├── text_encoder.py   # Qwen2.5-3B entity extraction → [N, 256]
│   │   ├── vision_encoder.py # CLIP ViT-L/14 patches → slot attention → [8, 256]
│   │   └── semantic_gate.py  # 4-signal validity filter (perplexity, CLIP, langid, entropy)
│   ├── memory/
│   │   ├── episodic.py       # ChromaDB episode store (write/query, 100k cap, eviction)
│   │   └── retrieval.py      # MemoryRetriever with cross-attention integration (~0.8M params)
│   ├── cognition/            # Phase 4 (not yet implemented)
│   ├── controller/           # Phase 5 (not yet implemented)
│   └── output/               # Phase 5 (not yet implemented)
├── train/
│   ├── train_slot_attention.py    # Toy shape-image reconstruction training
│   ├── train_workspace_update.py  # Synthetic object-absorption training
│   ├── seed_gate_data.py          # Source 500 (text, image) examples for the gate
│   ├── label_gate_ui.py           # Local web UI for hand-labeling gate examples
│   └── fit_gate.py                # TODO: compute features + call gate.fit()
├── tests/
├── data/
│   ├── episodic_db/     # ChromaDB persistent store
│   ├── synthetic/       # generated training data
│   ├── gate_data/       # SemanticGate labeling: unlabeled.jsonl + images/
│   └── checkpoints/     # saved model weights
└── report.md            # technical report on the LWI architecture
```

---

## Implementation status

| Phase | What | Status |
|---|---|---|
| 0 | Environment setup | Done |
| 1 | WorkspaceState, SlotAttention, UpdateTransformer | Code done, weights untrained |
| 2 | TextEncoder, VisionEncoder, SemanticGate | Code done; gate has data sourcing + labeling tools, fit script TODO |
| 3 | EpisodicMemory, MemoryRetriever | Done |
| 4 | Planner, WorldModel, SpatialReasoner, Consistency | Not started |
| 5 | StreamingController, LanguageDecoder | Not started |
| 6 | CLI, evaluation | Not started |

### Outstanding work to reach a usable Phase 4

1. Train `SlotAttention` (`python -m train.train_slot_attention`).
2. Train `WorkspaceUpdateTransformer` (`python -m train.train_workspace_update`).
3. (Optional but recommended) Source + label gate data, then run the not-yet-written `fit_gate.py`.
4. Begin implementing Phase 4 modules (Planner, WorldModel, SpatialReasoner, Consistency).
