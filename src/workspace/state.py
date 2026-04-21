from __future__ import annotations

import copy
import heapq
from dataclasses import dataclass, field
from typing import List

import torch
from torch import Tensor

N_SLOTS = 32
D = 256
R = 32
K_MEM = 16


def _default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class GoalEntry:
    embedding: Tensor
    priority: float
    description: str

    def __lt__(self, other: "GoalEntry") -> bool:
        # heapq is a min-heap; higher priority = lower heap key
        return self.priority > other.priority


@dataclass
class WorkspaceState:
    objects: Tensor          # [N, D]
    relations_idx: Tensor    # [2, E] — edge index (source, target)
    relations_val: Tensor    # [E, R] — edge feature vectors
    goals: List[GoalEntry]   # heapq
    uncertainty: Tensor      # [N]
    memory_slots: Tensor     # [K, D]

    @classmethod
    def zeros(cls, device: str | None = None) -> "WorkspaceState":
        dev = device or _default_device()
        return cls(
            objects=torch.zeros(N_SLOTS, D, dtype=torch.float32, device=dev),
            relations_idx=torch.zeros(2, 0, dtype=torch.long, device=dev),
            relations_val=torch.zeros(0, R, dtype=torch.float32, device=dev),
            goals=[],
            uncertainty=torch.zeros(N_SLOTS, dtype=torch.float32, device=dev),
            memory_slots=torch.zeros(K_MEM, D, dtype=torch.float32, device=dev),
        )

    def clone(self) -> "WorkspaceState":
        return WorkspaceState(
            objects=self.objects.clone(),
            relations_idx=self.relations_idx.clone(),
            relations_val=self.relations_val.clone(),
            goals=copy.deepcopy(self.goals),
            uncertainty=self.uncertainty.clone(),
            memory_slots=self.memory_slots.clone(),
        )

    def push_goal(self, embedding: Tensor, priority: float, description: str) -> None:
        heapq.heappush(self.goals, GoalEntry(embedding, priority, description))

    def pop_goal(self) -> GoalEntry | None:
        return heapq.heappop(self.goals) if self.goals else None
