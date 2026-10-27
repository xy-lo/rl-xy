from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rl_xy.reporting import deterministic_tree_hash, phase_metrics, write_tables
from rl_xy.types import AstraEvent


def event(round_id: int, start: int, end: int) -> AstraEvent:
    return AstraEvent(
        rank=0,
        node_id=round_id + 1,
        op_id=f"r{round_id:02d}.op",
        round_id=round_id,
        phase="rollout",
        node_type=4,
        issue_tick=start,
        callback_tick=end,
        duration_ticks=end - start,
        issue_ns=float(start),
        callback_ns=float(end),
        duration_ns=float(end - start),
    )


class ReportingTests(unittest.TestCase):
    def test_source_hash_excludes_only_generated_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            manual = docs / "ARCHITECTURE.md"
            manual.write_text("architecture", encoding="utf-8")
            initial = deterministic_tree_hash(root)
            for name in ("工程文件与完整流程.md", "完整流程预览.html", "资源版工程说明与完整流程.md", "资源版完整流程预览.html"):
                (docs / name).write_text("derived report", encoding="utf-8")
            self.assertEqual(initial, deterministic_tree_hash(root))
            manual.write_text("updated architecture", encoding="utf-8")
            self.assertNotEqual(initial, deterministic_tree_hash(root))

    def test_phase_spans_do_not_merge_rounds(self) -> None:
        metrics = phase_metrics([event(0, 0, 10), event(1, 100, 120)])
        self.assertEqual(metrics["r00.rollout"]["span_ns"], 10)
        self.assertEqual(metrics["r01.rollout"]["span_ns"], 20)
        self.assertNotIn("rollout", metrics)

    def test_empty_fixture_parameter_table_still_has_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_tables(
                output,
                operations=(),
                events=(),
                summaries=(),
                objects=(),
                snapshots=(),
                transfers=(),
                checks=(),
            )
            header = (output / "parameters.csv").read_text(encoding="utf-8")
            self.assertEqual(
                header.strip(),
                "parameter_id,unique_elements,local_elements_per_die,replicated,layout",
            )


if __name__ == "__main__":
    unittest.main()
