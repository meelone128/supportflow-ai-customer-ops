"""Stable assignment for opt-in prompt experiments."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptExperiment:
    experiment_id: str

    @classmethod
    def from_environment(cls) -> "PromptExperiment | None":
        experiment_id = os.getenv("SUPPORTFLOW_PROMPT_EXPERIMENT_ID", "").strip()
        return cls(experiment_id) if experiment_id else None

    def assign(self, ticket_id: str) -> str:
        digest = hashlib.sha256(f"{self.experiment_id}:{ticket_id}".encode()).digest()
        return "treatment" if digest[0] % 2 else "control"
