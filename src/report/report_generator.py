from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from openpyxl import Workbook


def write_reports(
    *,
    run_meta: Dict[str, Any],
    step_results: List[Any],
    out_dir: str,
) -> Dict[str, str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "run.json"
    html_path = out_path / "report.html"
    xlsx_path = out_path / "summary.xlsx"

    payload = {
        "meta": run_meta,
        "steps": [asdict(r) for r in step_results],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_html(payload, html_path)
    _write_xlsx(payload, xlsx_path)

    return {
        "json": str(json_path),
        "html": str(html_path),
        "xlsx": str(xlsx_path),
    }


def _write_html(payload: Dict[str, Any], path: Path) -> None:
    meta = payload.get("meta") or {}
    steps = payload.get("steps") or []
    title = str(meta.get("project") or "run")
    generated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    rows = []
    for s in steps:
        ok = bool(s.get("ok"))
        started = float(s.get("started_at") or 0)
        ended = float(s.get("ended_at") or 0)
        duration_ms = int(max(0.0, ended - started) * 1000)
        detail = s.get("detail") or {}
        err = str(detail.get("error") or "")
        rows.append(
            f"<tr><td>{s.get('name')}</td><td>{detail.get('type')}</td><td>{'PASS' if ok else 'FAIL'}</td><td>{duration_ms}</td><td>{err}</td></tr>"
        )

    html = "".join(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'>",
            f"<title>{title}</title>",
            "<style>body{font-family:Arial,Helvetica,sans-serif}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px}th{background:#f4f4f4}</style>",
            "</head><body>",
            f"<h1>{title}</h1>",
            f"<div>generated_at: {generated_at}</div>",
            "<h2>Steps</h2>",
            "<table><thead><tr><th>Name</th><th>Type</th><th>Status</th><th>Duration(ms)</th><th>Error</th></tr></thead><tbody>",
            "".join(rows),
            "</tbody></table>",
            "</body></html>",
        ]
    )
    path.write_text(html, encoding="utf-8")


def _write_xlsx(payload: Dict[str, Any], path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "summary"

    ws.append(["name", "type", "ok", "duration_ms", "error"])
    for s in payload.get("steps") or []:
        detail = s.get("detail") or {}
        started = float(s.get("started_at") or 0)
        ended = float(s.get("ended_at") or 0)
        duration_ms = int(max(0.0, ended - started) * 1000)
        ws.append(
            [
                s.get("name"),
                detail.get("type"),
                bool(s.get("ok")),
                duration_ms,
                detail.get("error"),
            ]
        )
    wb.save(path)

