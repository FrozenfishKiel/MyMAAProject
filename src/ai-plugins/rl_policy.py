from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class Decision:
    action: str
    params: Dict[str, Any]


class RlPolicy:
    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._model: Any | None = None

    def load(self) -> None:
        from stable_baselines3 import PPO

        self._model = PPO.load(self._model_path)

    def act(self, observation: Any) -> Decision:
        if self._model is None:
            raise RuntimeError("RL model not loaded")
        action, _state = self._model.predict(observation, deterministic=True)
        return Decision(action=str(action), params={})

