from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rl_xy.astra_adapter.generator import TraceGenerationError, generate_traces
from rl_xy.types import Operation


class GeneratorContractTests(unittest.TestCase):
    def test_p2p_participants_must_be_exact_endpoints(self) -> None:
        operation = Operation(
            1,
            "bad.p2p",
            0,
            "p",
            "bad",
            "p2p",
            payload_bytes=4,
            src_rank=0,
            dst_rank=1,
            participant_ranks=(0,),
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TraceGenerationError):
                generate_traces(
                    [operation], Path(directory) / "trace", ranks=2, astra_time_scale=1000
                )

    def test_p2p_tag_range_is_fail_closed_before_schema_import(self) -> None:
        operation = Operation(
            500_000_000,
            "bad.tag",
            0,
            "p",
            "bad",
            "p2p",
            payload_bytes=4,
            src_rank=0,
            dst_rank=1,
            participant_ranks=(0, 1),
            dependencies_by_rank={0: (), 1: ()},
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TraceGenerationError):
                generate_traces(
                    [operation], Path(directory) / "trace", ranks=2, astra_time_scale=1000
                )

    def test_unsupported_collective_is_rejected(self) -> None:
        operation = Operation(
            1,
            "bad.barrier",
            0,
            "p",
            "bad",
            "collective",
            payload_bytes=4,
            collective="BARRIER",
            participant_ranks=(0, 1),
            dependencies_by_rank={0: (), 1: ()},
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TraceGenerationError):
                generate_traces(
                    [operation], Path(directory) / "trace", ranks=2, astra_time_scale=1000
                )


if __name__ == "__main__":
    unittest.main()
