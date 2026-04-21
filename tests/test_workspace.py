import torch
import pytest

from src.workspace.state import WorkspaceState, N_SLOTS, D, K_MEM, R
from src.workspace.slot_attention import SlotAttention, SlotAttentionAutoEncoder
from src.workspace.update import WorkspaceUpdateTransformer


def test_workspace_state_initialization():
    ws = WorkspaceState.zeros(device="cpu")
    assert ws.objects.shape == (N_SLOTS, D)
    assert ws.memory_slots.shape == (K_MEM, D)
    assert ws.uncertainty.shape == (N_SLOTS,)
    assert ws.relations_idx.shape == (2, 0)
    assert ws.relations_val.shape == (0, R)
    assert ws.goals == []
    assert ws.objects.dtype == torch.float32


def test_workspace_clone_is_independent():
    ws = WorkspaceState.zeros(device="cpu")
    ws2 = ws.clone()
    ws2.objects[0, 0] = 99.0
    assert ws.objects[0, 0].item() == 0.0, "clone must be independent (copy-on-write)"


def test_workspace_persists_across_updates():
    model = WorkspaceUpdateTransformer().eval()
    ws = WorkspaceState.zeros(device="cpu")
    original_objects = ws.objects.clone()

    input_tokens = torch.randn(1, 8, D)

    with torch.no_grad():
        for _ in range(10):
            ws = model(ws, input_tokens)

    # State must have changed from zero
    assert not torch.allclose(ws.objects, original_objects), \
        "workspace objects should change after 10 update steps"

    # Original state must not have been mutated
    assert torch.allclose(original_objects, torch.zeros(N_SLOTS, D)), \
        "initial workspace state was mutated (copy-on-write violated)"


def test_object_slots_distinct():
    model = WorkspaceUpdateTransformer().eval()
    ws = WorkspaceState.zeros(device="cpu")

    torch.manual_seed(42)
    input_tokens = torch.randn(1, 8, D)

    with torch.no_grad():
        for _ in range(10):
            ws = model(ws, input_tokens)

    slots = ws.objects  # [N, D]
    # Normalize for cosine similarity
    normed = torch.nn.functional.normalize(slots, dim=-1)
    sim_matrix = normed @ normed.T  # [N, N]

    # Zero out diagonal (self-similarity)
    mask = ~torch.eye(N_SLOTS, dtype=torch.bool)
    off_diag = sim_matrix[mask]

    mean_sim = off_diag.mean().item()
    assert mean_sim < 0.99, \
        f"object slots are too similar (mean off-diagonal cosine sim = {mean_sim:.4f})"


def test_slot_attention_output_shape():
    model = SlotAttention(num_slots=8, slot_dim=64, input_dim=128, num_iters=3)
    x = torch.randn(2, 16, 128)
    with torch.no_grad():
        slots = model(x)
    assert slots.shape == (2, 8, 64)


def test_slot_attention_autoencoder():
    model = SlotAttentionAutoEncoder(num_slots=4, slot_dim=64, input_dim=32, num_iters=3)
    x = torch.randn(1, 10, 32)
    with torch.no_grad():
        slots, recon = model(x)
    assert slots.shape == (1, 4, 64)
    assert recon.shape == (1, 4, 32)


def test_goal_queue():
    ws = WorkspaceState.zeros(device="cpu")
    emb1 = torch.randn(D)
    emb2 = torch.randn(D)
    ws.push_goal(emb1, priority=0.5, description="low priority goal")
    ws.push_goal(emb2, priority=0.9, description="high priority goal")
    top = ws.pop_goal()
    assert top.description == "high priority goal", "heapq should return highest priority first"
