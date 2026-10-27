from __future__ import annotations

import unittest
from pathlib import Path

from rl_xy.fixture import FixtureError, build_transfer_fixture, validate_63b_negative


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "configs/fixture_transfer/case.json"


class TransferFixtureTest(unittest.TestCase):
    def test_static_schedule_matches_hand_calculation(self) -> None:
        fixture = build_transfer_fixture(CASE)
        compute_ns = sum(op.duration_ns for op in fixture.operations if op.kind == "compute")
        self.assertEqual(compute_ns, 16)
        p2p = [op for op in fixture.operations if op.kind == "p2p"]
        self.assertEqual(len(p2p), 1)
        self.assertEqual(p2p[0].payload_bytes, 32)
        self.assertEqual(fixture.operations[0].participant_ranks, (0,))
        self.assertEqual(fixture.operations[4].dependencies_by_rank, {0: (4,), 1: ()})
        self.assertEqual(fixture.operations[-1].participant_ranks, (1,))
        self.assertEqual(max(row.allocated_bytes for row in fixture.memory_snapshots), 64)

    def test_63_bytes_is_rejected_not_in_place(self) -> None:
        with self.assertRaises(FixtureError):
            build_transfer_fixture(CASE, sram_override=63)
        self.assertTrue(validate_63b_negative(CASE).passed)


if __name__ == "__main__":
    unittest.main()
