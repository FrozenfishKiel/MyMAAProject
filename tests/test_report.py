import tempfile
import unittest

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "maa-wrapper"))
sys.path.insert(0, str(ROOT / "src" / "report"))

from runtime import StepResult  # type: ignore
from report_generator import write_reports  # type: ignore


class ReportTests(unittest.TestCase):
    def test_reports_written(self) -> None:
        steps = [
            StepResult(
                name="step_0",
                ok=True,
                started_at=1.0,
                ended_at=2.0,
                detail={"type": "wait"},
            )
        ]
        with tempfile.TemporaryDirectory() as td:
            outputs = write_reports(run_meta={"project": "t"}, step_results=steps, out_dir=td)
            self.assertTrue(Path(outputs["json"]).exists())
            self.assertTrue(Path(outputs["html"]).exists())
            self.assertTrue(Path(outputs["xlsx"]).exists())


if __name__ == "__main__":
    unittest.main()

