from __future__ import annotations

import unittest
from pathlib import Path

from rl_xy.config import load_case
from rl_xy.ledger import build_ledger
from rl_xy.model import parameter_totals
from rl_xy.workflow import build_workflow


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "configs" / "toy_grpo_2die" / "case.json"


class LedgerTests(unittest.TestCase):
    def test_parameter_and_kv_golden(self) -> None:
        case = load_case(CASE)
        operations = build_workflow(case)
        ledger = build_ledger(case, operations)
        totals = parameter_totals(case)

        self.assertEqual(totals["unique_parameter_elements"], 1576)
        self.assertEqual(totals["local_parameter_elements_per_die"], 808)
        self.assertEqual(totals["long_term_bytes_per_die"], 12928)
        self.assertEqual(totals["gradient_buffer_bytes_per_die"], 3232)
        self.assertEqual(totals["resident_with_gradient_bytes_per_die"], 16160)

        persistent_d0 = [
            item for item in ledger.objects if item.persistent and item.owner_rank == 0
        ]
        self.assertEqual(sum(item.nbytes for item in persistent_d0), 16160)

        d0_view = sorted(
            (
                item
                for item in ledger.kv_snapshots
                if item.phase == "r00.rollout" and item.location == "d0.dram.kv_view"
            ),
            key=lambda item: item.step,
        )
        self.assertEqual(d0_view[0].allocated_bytes, 0)
        self.assertEqual(d0_view[0].unmaterialized_bytes, 256)
        self.assertEqual([item.valid_bytes for item in d0_view[1:]], [128, 192, 96, 128, 0])
        self.assertEqual(ledger.kv_bytes_written_per_die["r0.d0"], 224)
        self.assertEqual(ledger.kv_bytes_written_per_die["r0.d1"], 224)
        nic_peaks = {
            item.location: item.allocated_bytes
            for item in ledger.memory_snapshots
            if ".nic." in item.location
        }
        self.assertEqual(nic_peaks["d0.nic.tx"], 8)
        self.assertEqual(nic_peaks["d0.nic.rx"], 32)
        self.assertEqual(nic_peaks["d1.nic.tx"], 32)
        self.assertEqual(nic_peaks["d1.nic.rx"], 8)

        kv_d0 = [
            item
            for item in ledger.objects
            if item.kind == "kv" and item.owner_rank == 0
        ]
        self.assertEqual(sum(item.nbytes for item in kv_d0), 224)
        self.assertTrue(all(item.released_at is not None for item in kv_d0))

    def test_two_rounds_reuse_persistent_storage_and_leave_no_kv(self) -> None:
        case = load_case(CASE, rounds_override=2)
        ledger = build_ledger(case, build_workflow(case))

        # Persistent storage is represented once, not duplicated per round.
        persistent = [item for item in ledger.objects if item.persistent]
        self.assertEqual(sum(item.nbytes for item in persistent if item.owner_rank == 0), 16160)
        self.assertEqual(sum(item.nbytes for item in persistent if item.owner_rank == 1), 16160)

        for round_id in (0, 1):
            for rank in (0, 1):
                final = [
                    item
                    for item in ledger.memory_snapshots
                    if item.phase == f"r{round_id:02d}.rollout"
                    and item.location == f"d{rank}.dram"
                    and item.reason == "after_release_A"
                ]
                self.assertEqual(len(final), 1)
                self.assertEqual(final[0].allocated_bytes, 16160)
                self.assertEqual(final[0].unmaterialized_bytes, 0)
                self.assertEqual(
                    ledger.kv_bytes_written_per_die[f"r{round_id}.d{rank}"], 224
                )


if __name__ == "__main__":
    unittest.main()

