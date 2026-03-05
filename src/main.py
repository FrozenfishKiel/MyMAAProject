from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parent
DEPS_ROOT = ROOT.parent / "maa-deps" / "maafw-5.2.6-win_amd64"
if DEPS_ROOT.exists():
    sys.path.insert(0, str(DEPS_ROOT))
sys.path.insert(0, str(ROOT / "maa-wrapper"))
sys.path.insert(0, str(ROOT / "report"))

from runtime import BotRuntime, DryRunAdapter, MaaFwAdapter  # type: ignore
from report_generator import write_reports  # type: ignore


def _load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        return dict(yaml.safe_load(raw) or {})
    if p.suffix.lower() == ".json":
        return dict(json.loads(raw))
    raise ValueError(f"Unsupported config file: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default=str(Path("report") / "out"))
    args = parser.parse_args()

    cfg = _load_config(args.config)
    project = dict(cfg.get("project") or {})
    device_cfg = dict(cfg.get("device") or {})
    tasks = list(cfg.get("tasks") or [])

    if args.dry_run or device_cfg.get("mode") == "dry_run":
        adapter = DryRunAdapter()
    else:
        adapter = MaaFwAdapter(device_cfg)
        adapter.connect()

    runtime = BotRuntime(adapter)
    results = runtime.run_tasks(tasks)

    run_meta = {
        "project": project.get("name") or "run",
        "device": device_cfg,
    }
    outputs = write_reports(run_meta=run_meta, step_results=results, out_dir=args.out)
    print(json.dumps({"ok": all(r.ok for r in results), "outputs": outputs}, ensure_ascii=False))
    return 0 if all(r.ok for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
