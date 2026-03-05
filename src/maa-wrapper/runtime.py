from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class StepResult:
    name: str
    ok: bool
    started_at: float
    ended_at: float
    detail: Dict[str, Any]


class BotRuntime:
    def __init__(self, adapter: "BaseAdapter") -> None:
        self._adapter = adapter

    def run_tasks(self, tasks: List[Dict[str, Any]]) -> List[StepResult]:
        results: List[StepResult] = []
        for idx, step in enumerate(tasks):
            step_type = str(step.get("type") or "")
            started = time.time()
            ok = True
            detail: Dict[str, Any] = {"index": idx, "type": step_type}
            try:
                self._run_step(step)
            except Exception as e:
                ok = False
                detail["error"] = str(e)
            ended = time.time()
            results.append(
                StepResult(
                    name=f"step_{idx}",
                    ok=ok,
                    started_at=started,
                    ended_at=ended,
                    detail=detail,
                )
            )
            if not ok:
                break
        return results

    def _run_step(self, step: Dict[str, Any]) -> None:
        step_type = str(step.get("type") or "")
        if step_type == "wait":
            seconds = float(step.get("seconds") or 0)
            self._adapter.sleep(seconds)
            return
        if step_type == "tap":
            x = int(step["x"])
            y = int(step["y"])
            self._adapter.tap(x, y)
            return
        if step_type == "assert":
            condition = str(step.get("condition") or "")
            if condition == "always_true":
                return
            raise ValueError(f"Unsupported assert condition: {condition}")
        raise ValueError(f"Unsupported step type: {step_type}")


class BaseAdapter:
    def tap(self, x: int, y: int) -> None:
        raise NotImplementedError

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class DryRunAdapter(BaseAdapter):
    def __init__(self) -> None:
        self.actions: List[Dict[str, Any]] = []

    def tap(self, x: int, y: int) -> None:
        self.actions.append({"type": "tap", "x": x, "y": y})


class MaaFwAdapter(BaseAdapter):
    def __init__(self, device_config: Dict[str, Any]) -> None:
        self._device_config = device_config
        self._client: Optional[Any] = None
        self._controller: Optional[Any] = None

    def connect(self) -> None:
        import importlib

        maa = importlib.import_module("maa")
        try:
            version = maa.library.Library.version()
        except Exception as e:
            raise RuntimeError(f"MaaFramework Python runtime not available: {e}")
        controller_type = str(self._device_config.get("type") or "").lower()
        if not controller_type:
            raise ValueError("device.type is required (e.g. win32, adb)")

        from maa.controller import AdbController, Win32Controller

        if controller_type == "win32":
            hwnd = self._device_config.get("hwnd")
            if hwnd is None:
                raise ValueError("device.hwnd is required for win32")
            self._controller = Win32Controller(int(hwnd))
        elif controller_type == "adb":
            adb_path = self._device_config.get("adb_path")
            address = self._device_config.get("address")
            if not adb_path or not address:
                raise ValueError("device.adb_path and device.address are required for adb")
            
            # 获取截图方式和配置
            screencap_methods = self._device_config.get("screencap_methods")
            config = self._device_config.get("config", {})
            
            # 创建ADB控制器
            if screencap_methods is not None:
                self._controller = AdbController(
                    adb_path=str(adb_path), 
                    address=str(address),
                    screencap_methods=screencap_methods,
                    config=config
                )
            else:
                self._controller = AdbController(adb_path=str(adb_path), address=str(address))
        else:
            raise ValueError(f"Unsupported device.type: {controller_type}")

        job = self._controller.post_connection().wait()
        if not job.succeeded:
            raise RuntimeError("Maa controller connection failed")

        self._client = maa

    def tap(self, x: int, y: int) -> None:
        if self._client is None or self._controller is None:
            raise RuntimeError("MaaFwAdapter not connected")
        job = self._controller.post_click(int(x), int(y)).wait()
        if not job.succeeded:
            raise RuntimeError("Maa click failed")
