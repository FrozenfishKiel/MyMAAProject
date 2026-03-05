from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class NextSteps:
    tasks: List[Dict[str, Any]]


class HybridDecision:
    def __init__(self, mode: str) -> None:
        self._mode = mode

    def decide(self, context: Dict[str, Any]) -> NextSteps:
        if self._mode == "fixed":
            tasks = list(context.get("tasks") or [])
            return NextSteps(tasks=tasks)
        if self._mode == "rl":
            raise NotImplementedError("RL-driven task synthesis not wired")
        raise ValueError(f"Unsupported decision mode: {self._mode}")

