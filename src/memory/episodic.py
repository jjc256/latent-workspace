from __future__ import annotations

import base64
import heapq
import json
import time
import uuid
import zlib
from typing import Any

import chromadb
import numpy as np
import torch
from torch import Tensor

from src.workspace.state import GoalEntry, WorkspaceState

MAX_EPISODES: int = 100_000
_EVICT_BATCH: int = 1_000
_COLLECTION_NAME: str = "episodes"


def _default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _tensor_to_dict(t: Tensor) -> dict[str, Any]:
    arr = t.detach().cpu().numpy()
    return {
        "z": base64.b64encode(zlib.compress(arr.tobytes(), level=6)).decode("ascii"),
        "s": list(arr.shape),
        "d": str(arr.dtype),
    }


def _dict_to_tensor(d: dict[str, Any], device: str) -> Tensor:
    arr = (
        np.frombuffer(zlib.decompress(base64.b64decode(d["z"])), dtype=np.dtype(d["d"]))
        .reshape(d["s"])
        .copy()
    )
    return torch.from_numpy(arr).to(device)


def _serialize_workspace(state: WorkspaceState) -> str:
    goals_data = [
        {
            "embedding": _tensor_to_dict(g.embedding),
            "priority": g.priority,
            "description": g.description,
        }
        for g in state.goals
    ]
    snap = {
        "objects": _tensor_to_dict(state.objects),
        "relations_idx": _tensor_to_dict(state.relations_idx),
        "relations_val": _tensor_to_dict(state.relations_val),
        "uncertainty": _tensor_to_dict(state.uncertainty),
        "memory_slots": _tensor_to_dict(state.memory_slots),
        "goals": goals_data,
    }
    return json.dumps(snap)


def _deserialize_workspace(doc: str, device: str) -> WorkspaceState:
    snap = json.loads(doc)
    goals: list[GoalEntry] = []
    for g in snap["goals"]:
        entry = GoalEntry(
            embedding=_dict_to_tensor(g["embedding"], device),
            priority=g["priority"],
            description=g["description"],
        )
        heapq.heappush(goals, entry)
    return WorkspaceState(
        objects=_dict_to_tensor(snap["objects"], device),
        relations_idx=_dict_to_tensor(snap["relations_idx"], device),
        relations_val=_dict_to_tensor(snap["relations_val"], device),
        uncertainty=_dict_to_tensor(snap["uncertainty"], device),
        memory_slots=_dict_to_tensor(snap["memory_slots"], device),
        goals=goals,
    )


class EpisodicMemory:
    def __init__(self, path: str = "./data/episodic_db") -> None:
        self._client = chromadb.PersistentClient(path=path)
        self._col = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def write(
        self,
        state: WorkspaceState,
        action: str,
        outcome: str,
        goal_id: str = "",
        timestamp: float | None = None,
    ) -> str:
        if self.count() >= MAX_EPISODES:
            self._evict_oldest(_EVICT_BATCH)

        ts = timestamp if timestamp is not None else time.time()
        ep_id = f"ep_{int(ts * 1000)}_{uuid.uuid4().hex[:8]}"
        embedding = self._make_embedding(state)
        doc = _serialize_workspace(state)
        metadata = {
            "action": action,
            "outcome": outcome,
            "timestamp": ts,
            "goal_id": goal_id,
        }
        self._col.add(
            embeddings=[embedding.tolist()],
            documents=[doc],
            metadatas=[metadata],
            ids=[ep_id],
        )
        return ep_id

    def query(
        self,
        query_embedding: Tensor,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        qvec = query_embedding.detach().cpu().numpy().astype(np.float32)
        kwargs: dict[str, Any] = {
            "query_embeddings": [qvec.tolist()],
            "n_results": min(n_results, self.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where

        raw = self._col.query(**kwargs)
        device = _default_device()
        results = []
        for i in range(len(raw["ids"][0])):
            results.append(
                {
                    "id": raw["ids"][0][i],
                    "distance": raw["distances"][0][i],
                    "metadata": raw["metadatas"][0][i],
                    "state": _deserialize_workspace(raw["documents"][0][i], device),
                }
            )
        return results

    def count(self) -> int:
        return self._col.count()

    def delete(self, episode_ids: list[str]) -> None:
        self._col.delete(ids=episode_ids)

    def _make_embedding(self, state: WorkspaceState) -> np.ndarray:
        return state.objects.detach().cpu().mean(dim=0).numpy().astype(np.float32)

    def _evict_oldest(self, n: int = _EVICT_BATCH) -> None:
        result = self._col.get(include=["metadatas"])
        ids = result["ids"]
        timestamps = [m["timestamp"] for m in result["metadatas"]]
        sorted_ids = [id_ for id_, _ in sorted(zip(ids, timestamps), key=lambda x: x[1])]
        self._col.delete(ids=sorted_ids[:n])
