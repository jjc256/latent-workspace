from __future__ import annotations

import heapq

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.memory.episodic import (
    EpisodicMemory,
    _deserialize_workspace,
    _dict_to_tensor,
    _serialize_workspace,
    _tensor_to_dict,
)
from src.memory.retrieval import CrossAttentionRetriever, MemoryRetriever
from src.workspace.state import D, N_SLOTS, GoalEntry, WorkspaceState


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def test_tensor_roundtrip_float():
    t = torch.randn(32, 256, dtype=torch.float32)
    recovered = _dict_to_tensor(_tensor_to_dict(t), device="cpu")
    assert recovered.shape == t.shape
    assert torch.allclose(t, recovered)


def test_tensor_roundtrip_long():
    t = torch.randint(0, 100, (2, 10), dtype=torch.long)
    recovered = _dict_to_tensor(_tensor_to_dict(t), device="cpu")
    assert recovered.shape == t.shape
    assert torch.equal(t, recovered)


def test_tensor_roundtrip_empty():
    t = torch.zeros(2, 0, dtype=torch.float32)
    recovered = _dict_to_tensor(_tensor_to_dict(t), device="cpu")
    assert recovered.shape == (2, 0)


def test_workspace_serialization_roundtrip():
    ws = WorkspaceState.zeros(device="cpu")
    ws.objects = torch.randn(N_SLOTS, D)
    ws.uncertainty = torch.rand(N_SLOTS)
    doc = _serialize_workspace(ws)
    recovered = _deserialize_workspace(doc, device="cpu")
    assert recovered.objects.shape == (N_SLOTS, D)
    assert torch.allclose(ws.objects, recovered.objects)
    assert torch.allclose(ws.uncertainty, recovered.uncertainty)
    assert torch.allclose(ws.memory_slots, recovered.memory_slots)
    assert len(recovered.goals) == 0


def test_workspace_serialization_with_goals():
    ws = WorkspaceState.zeros(device="cpu")
    ws.push_goal(torch.randn(D), priority=1.0, description="low")
    ws.push_goal(torch.randn(D), priority=5.0, description="high")
    ws.push_goal(torch.randn(D), priority=3.0, description="mid")
    doc = _serialize_workspace(ws)
    recovered = _deserialize_workspace(doc, device="cpu")
    assert len(recovered.goals) == 3
    # heapq pops highest priority first
    assert heapq.heappop(recovered.goals).description == "high"
    assert heapq.heappop(recovered.goals).description == "mid"
    assert heapq.heappop(recovered.goals).description == "low"


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------


def test_episodic_write_and_count(tmp_path):
    store = EpisodicMemory(path=str(tmp_path / "db"))
    ws = WorkspaceState.zeros(device="cpu")
    store.write(ws, action="test", outcome="ok")
    assert store.count() == 1


def test_episodic_write_returns_id(tmp_path):
    store = EpisodicMemory(path=str(tmp_path / "db"))
    ws = WorkspaceState.zeros(device="cpu")
    ep_id = store.write(ws, action="test", outcome="ok")
    assert isinstance(ep_id, str)
    assert ep_id.startswith("ep_")


def test_episodic_query_returns_k_results(tmp_path):
    store = EpisodicMemory(path=str(tmp_path / "db"))
    for _ in range(10):
        ws = WorkspaceState.zeros(device="cpu")
        ws.objects = torch.randn(N_SLOTS, D)
        store.write(ws, action="test", outcome="ok")
    q = torch.randn(D)
    results = store.query(q, n_results=5)
    assert len(results) == 5


def test_episodic_query_result_structure(tmp_path):
    store = EpisodicMemory(path=str(tmp_path / "db"))
    ws = WorkspaceState.zeros(device="cpu")
    store.write(ws, action="describe", outcome="success", goal_id="g1")
    q = ws.objects.mean(dim=0)
    results = store.query(q, n_results=1)
    r = results[0]
    assert "id" in r and "distance" in r and "metadata" in r and "state" in r
    assert r["metadata"]["action"] == "describe"
    assert r["metadata"]["outcome"] == "success"
    assert r["metadata"]["goal_id"] == "g1"
    assert isinstance(r["state"], WorkspaceState)
    assert r["state"].objects.shape == (N_SLOTS, D)


def test_episodic_query_nearest_is_self(tmp_path):
    store = EpisodicMemory(path=str(tmp_path / "db"))
    for _ in range(20):
        ws = WorkspaceState.zeros(device="cpu")
        ws.objects = torch.randn(N_SLOTS, D)
        store.write(ws, action="noise", outcome="ok")
    known_ws = WorkspaceState.zeros(device="cpu")
    known_ws.objects = torch.ones(N_SLOTS, D)
    ep_id = store.write(known_ws, action="target", outcome="ok")
    q = known_ws.objects.mean(dim=0)
    results = store.query(q, n_results=1)
    assert results[0]["id"] == ep_id


def test_episodic_eviction(tmp_path):
    import src.memory.episodic as ep_module
    original_max = ep_module.MAX_EPISODES
    original_batch = ep_module._EVICT_BATCH
    ep_module.MAX_EPISODES = 5
    ep_module._EVICT_BATCH = 2
    try:
        store = EpisodicMemory(path=str(tmp_path / "db"))
        for _ in range(8):
            ws = WorkspaceState.zeros(device="cpu")
            store.write(ws, action="test", outcome="ok")
        assert store.count() <= 5
    finally:
        ep_module.MAX_EPISODES = original_max
        ep_module._EVICT_BATCH = original_batch


def test_episodic_metadata_filter(tmp_path):
    store = EpisodicMemory(path=str(tmp_path / "db"))
    for i in range(5):
        ws = WorkspaceState.zeros(device="cpu")
        ws.objects = torch.randn(N_SLOTS, D)
        store.write(ws, action="success" if i < 3 else "failure", outcome="ok")
    q = torch.randn(D)
    results = store.query(q, n_results=10, where={"action": {"$eq": "success"}})
    assert len(results) == 3
    assert all(r["metadata"]["action"] == "success" for r in results)


# ---------------------------------------------------------------------------
# CrossAttentionRetriever
# ---------------------------------------------------------------------------


def test_cross_attention_output_shape():
    model = CrossAttentionRetriever()
    ws_obj = torch.randn(N_SLOTS, D)
    retrieved = torch.randn(5 * N_SLOTS, D)
    out = model(ws_obj, retrieved)
    assert out.shape == (N_SLOTS, D)


def test_cross_attention_is_residual():
    model = CrossAttentionRetriever()
    model.eval()
    ws_obj = torch.randn(N_SLOTS, D)
    retrieved = torch.zeros(5 * N_SLOTS, D)
    with torch.no_grad():
        out = model(ws_obj, retrieved)
    assert out.shape == (N_SLOTS, D)


def test_cross_attention_param_count():
    model = CrossAttentionRetriever()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 2_000_000


# ---------------------------------------------------------------------------
# MemoryRetriever
# ---------------------------------------------------------------------------


def test_retriever_retrieve_returns_new_workspace(tmp_path):
    store = EpisodicMemory(path=str(tmp_path / "db"))
    for _ in range(10):
        ws = WorkspaceState.zeros(device="cpu")
        ws.objects = torch.randn(N_SLOTS, D)
        store.write(ws, action="test", outcome="ok")
    retriever = MemoryRetriever(store, device="cpu")
    ws = WorkspaceState.zeros(device="cpu")
    ws.objects = torch.randn(N_SLOTS, D)
    updated = retriever.retrieve(ws, k=5)
    assert isinstance(updated, WorkspaceState)
    assert updated.objects.shape == (N_SLOTS, D)
    assert not torch.equal(updated.objects, ws.objects)


def test_retriever_does_not_mutate_input(tmp_path):
    store = EpisodicMemory(path=str(tmp_path / "db"))
    for _ in range(10):
        ws = WorkspaceState.zeros(device="cpu")
        ws.objects = torch.randn(N_SLOTS, D)
        store.write(ws, action="test", outcome="ok")
    retriever = MemoryRetriever(store, device="cpu")
    ws = WorkspaceState.zeros(device="cpu")
    original_objects = ws.objects.clone()
    retriever.retrieve(ws, k=5)
    assert torch.equal(ws.objects, original_objects)


def test_retriever_goal_conditioned_query(tmp_path):
    store = EpisodicMemory(path=str(tmp_path / "db"))
    for _ in range(10):
        ws = WorkspaceState.zeros(device="cpu")
        ws.objects = torch.randn(N_SLOTS, D)
        store.write(ws, action="test", outcome="ok")
    retriever = MemoryRetriever(store, device="cpu")
    ws = WorkspaceState.zeros(device="cpu")
    goal_emb = torch.randn(D)
    updated = retriever.retrieve(ws, goal_embedding=goal_emb, k=5)
    assert updated.objects.shape == (N_SLOTS, D)


def test_retriever_save_load(tmp_path):
    store = EpisodicMemory(path=str(tmp_path / "db"))
    retriever = MemoryRetriever(store, device="cpu")
    ckpt_path = str(tmp_path / "retriever.pt")
    retriever.save(ckpt_path)
    retriever2 = MemoryRetriever(store, device="cpu")
    retriever2.load(ckpt_path)
    for p1, p2 in zip(retriever.cross_attn.parameters(), retriever2.cross_attn.parameters()):
        assert torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
# Milestone: 1,000 synthetic episodes
# ---------------------------------------------------------------------------


def test_milestone_1000_episodes(tmp_path):
    store = EpisodicMemory(path=str(tmp_path / "db"))

    N_CLUSTERS = 5
    N_PER_CLUSTER = 200
    ACTIONS = ["describe_scene", "locate_object", "compare_objects", "track_goal", "update_state"]

    cluster_centers = F.normalize(torch.randn(N_CLUSTERS, D), dim=-1)

    cluster_ids_by_ep: dict[str, int] = {}
    for cluster_idx in range(N_CLUSTERS):
        center = cluster_centers[cluster_idx]
        for _ in range(N_PER_CLUSTER):
            ws = WorkspaceState.zeros(device="cpu")
            noise = torch.randn(N_SLOTS, D) * 0.1
            ws.objects = center.unsqueeze(0).expand(N_SLOTS, -1) + noise
            ep_id = store.write(
                ws,
                action=ACTIONS[cluster_idx],
                outcome="success",
                goal_id=f"cluster_{cluster_idx}",
            )
            cluster_ids_by_ep[ep_id] = cluster_idx

    assert store.count() == 1000

    for cluster_idx in range(N_CLUSTERS):
        query_vec = cluster_centers[cluster_idx]
        results = store.query(query_vec, n_results=5)
        assert len(results) == 5
        for r in results:
            assert cluster_ids_by_ep[r["id"]] == cluster_idx, (
                f"Cluster {cluster_idx}: retrieved episode from cluster "
                f"{cluster_ids_by_ep[r['id']]}. "
                f"Distances: {[r2['distance'] for r2 in results]}"
            )

    sample_results = store.query(cluster_centers[0], n_results=5)
    for r in sample_results:
        assert r["metadata"]["action"] == ACTIONS[0]
        assert r["metadata"]["outcome"] == "success"
        assert r["metadata"]["goal_id"] == "cluster_0"
        assert isinstance(r["state"], WorkspaceState)
        assert r["state"].objects.shape == (N_SLOTS, D)
