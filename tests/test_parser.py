from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rl_xy.astra_adapter.parser import AstraParseError, parse_astra_output
from rl_xy.types import AstraRunResult, Operation


class ParserTest(unittest.TestCase):
    def _run(self, directory: Path, *, missing_callback: bool = False):
        stdout = directory / "stdout.log"
        stderr = directory / "stderr.log"
        debug = directory / "log.log"
        stdout.write_text(
            "\n".join(
                [
                    "[x] [I] <workload>: sys[0] finished, 7000 cycles, exposed communication 5000 cycles.",
                    "[x] [I] <statistics>: sys[0], Wall time: 7000",
                    "[x] [I] <statistics>: sys[0], GPU time: 2000",
                    "[x] [I] <statistics>: sys[0], Comm time: 5000",
                    "[x] [I] <workload>: sys[1] finished, 7000 cycles, exposed communication 5000 cycles.",
                    "[x] [I] <statistics>: sys[1], Wall time: 7000",
                    "[x] [I] <statistics>: sys[1], GPU time: 2000",
                    "[x] [I] <statistics>: sys[1], Comm time: 5000",
                ]
            ),
            encoding="utf-8",
        )
        stderr.write_text("", encoding="utf-8")
        lines = []
        for rank in (0, 1):
            lines.extend(
                [
                    f"[x] [D] <workload>: issue,sys->id={rank}, tick=0, node->id=1, node->name=op.compute, node->type=4",
                    f"[x] [D] <workload>: callback,sys->id={rank}, tick=2000, node->id=1, node->name=op.compute, node->type=4",
                    f"[x] [D] <workload>: issue,sys->id={rank}, tick=2000, node->id=2, node->name=op.p2p, node->type={5 if rank == 0 else 6}",
                ]
            )
            if not (missing_callback and rank == 1):
                lines.append(f"[x] [D] <workload>: callback,sys->id={rank}, tick=7000, node->id=2, node->name=op.p2p, node->type={5 if rank == 0 else 6}")
        debug.write_text("\n".join(lines), encoding="utf-8")
        result = AstraRunResult(
            command=("astra",),
            return_code=0,
            started_at_utc="start",
            finished_at_utc="end",
            stdout_path=stdout,
            stderr_path=stderr,
            debug_log_path=debug,
            binary_path=directory / "astra",
            binary_sha256="0" * 64,
            astra_commit="4ce9ecfe746a3ee5fe47559a11d91644d641fa7f",
        )
        operations = [
            Operation(1, "op.compute", 0, "p", "compute", "compute", duration_ns=2),
            Operation(2, "op.p2p", 0, "p", "p2p", "p2p", payload_bytes=32, src_rank=0, dst_rank=1, dependencies=(1,)),
        ]
        return result, operations

    def test_pairs_every_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, operations = self._run(Path(directory))
            events, summaries = parse_astra_output(result, operations, ranks=2, astra_time_scale=1000)
            self.assertEqual(len(events), 4)
            self.assertEqual(events[-1].callback_ns, 7)
            self.assertEqual([row.wall_cycles for row in summaries], [7000, 7000])

    def test_only_explicit_idle_rank_can_omit_zero_communication_stat(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as directory:
            result,operations=self._run(Path(directory))
            operations=[replace(operations[0],participant_ranks=(0,1),metadata={'idle_rank_marker':True})]
            text=result.stdout_path.read_text()
            text='\n'.join(x.replace('7000','2000').replace('5000','0') for x in text.splitlines() if 'Comm time' not in x)
            result.stdout_path.write_text(text)
            result.debug_log_path.write_text('\n'.join(x for x in result.debug_log_path.read_text().splitlines() if 'node->id=1,' in x))
            with self.assertRaises(AstraParseError):parse_astra_output(result,operations,ranks=2,astra_time_scale=1000)
            events,summaries=parse_astra_output(result,operations,ranks=2,astra_time_scale=1000,inactive_ranks=(0,1))
            self.assertEqual([s.communication_cycles for s in summaries],[0,0])
            with self.assertRaises(AstraParseError):parse_astra_output(result,[replace(operations[0],metadata={})],ranks=2,astra_time_scale=1000,inactive_ranks=(0,1))

    def test_missing_callback_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, operations = self._run(Path(directory), missing_callback=True)
            with self.assertRaises(AstraParseError):
                parse_astra_output(result, operations, ranks=2, astra_time_scale=1000)

    def test_rotated_segments_are_all_parsed(self):
        with tempfile.TemporaryDirectory() as d:
            result,operations=self._run(Path(d))
            lines=result.debug_log_path.read_text().splitlines()
            (Path(d)/'log.2.log').write_text('\n'.join(lines[:2]))
            (Path(d)/'log.1.log').write_text('\n'.join(lines[2:5]))
            result.debug_log_path.write_text('\n'.join(lines[5:]))
            events,_=parse_astra_output(result,operations,ranks=2,astra_time_scale=1000)
            self.assertEqual(len(events),4)

    def test_rotated_missing_duplicate_or_error_fails_closed(self):
        for mode in ('gap','duplicate','critical','missing_early'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                result,operations=self._run(Path(d))
                lines=result.debug_log_path.read_text().splitlines()
                if mode=='gap':(Path(d)/'log.2.log').write_text('unused')
                elif mode=='duplicate':(Path(d)/'log.1.log').write_text(lines[0])
                elif mode=='critical':(Path(d)/'log.1.log').write_text('[x] [system] [critical] failed')
                else:result.debug_log_path.write_text('\n'.join(lines[2:]))
                with self.assertRaises(AstraParseError):
                    parse_astra_output(result,operations,ranks=2,astra_time_scale=1000)

    def test_swallowed_native_callback_error_is_not_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, operations = self._run(Path(directory))
            for marker in ('[critical]','[error]','[C]'):
                result.debug_log_path.write_text(result.debug_log_path.read_text()+
                    f'\n[x] [system] {marker} callback failed\n')
                with self.assertRaises(AstraParseError):
                    parse_astra_output(result,operations,ranks=2,astra_time_scale=1000)

    def test_rank_local_nodes_and_posted_receive_are_paired(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            stdout = directory / "stdout.log"
            stderr = directory / "stderr.log"
            debug = directory / "log.log"
            stdout.write_text(
                "\n".join(
                    [
                        "[x] [I] <statistics>: sys[1], Comm time: 7000",
                        "[x] [I] <workload>: sys[0] finished, 7000 cycles, exposed communication 5000 cycles.",
                        "[x] [I] <statistics>: sys[0], GPU time: 2000",
                        "[x] [I] <statistics>: sys[1], Wall time: 9000",
                        "[x] [I] <statistics>: sys[0], Wall time: 7000",
                        "[x] [I] <workload>: sys[1] finished, 9000 cycles, exposed communication 7000 cycles.",
                        "[x] [I] <statistics>: sys[0], Comm time: 5000",
                        "[x] [I] <statistics>: sys[1], GPU time: 2000",
                    ]
                ),
                encoding="utf-8",
            )
            stderr.write_text("", encoding="utf-8")
            # Deliberately shuffled: the parser must use event identity/ticks,
            # never asynchronous log line order.
            debug.write_text(
                "\n".join(
                    [
                        "[x] [D] <workload>: callback,sys->id=1, tick=9000, node->id=3, node->name=op.rank1, node->type=4",
                        "[x] [D] <workload>: issue,sys->id=1, tick=0, node->id=2, node->name=op.p2p, node->type=6",
                        "[x] [D] <workload>: callback,sys->id=0, tick=2000, node->id=1, node->name=op.rank0, node->type=4",
                        "[x] [D] <workload>: callback,sys->id=1, tick=7000, node->id=2, node->name=op.p2p, node->type=6",
                        "[x] [D] <workload>: issue,sys->id=0, tick=2000, node->id=2, node->name=op.p2p, node->type=5",
                        "[x] [D] <workload>: issue,sys->id=0, tick=0, node->id=1, node->name=op.rank0, node->type=4",
                        "[x] [D] <workload>: issue,sys->id=1, tick=7000, node->id=3, node->name=op.rank1, node->type=4",
                        "[x] [D] <workload>: callback,sys->id=0, tick=7000, node->id=2, node->name=op.p2p, node->type=5",
                    ]
                ),
                encoding="utf-8",
            )
            result = AstraRunResult(
                command=("astra",),
                return_code=0,
                started_at_utc="start",
                finished_at_utc="end",
                stdout_path=stdout,
                stderr_path=stderr,
                debug_log_path=debug,
                binary_path=directory / "astra",
                binary_sha256="0" * 64,
                astra_commit="4ce9ecfe746a3ee5fe47559a11d91644d641fa7f",
            )
            operations = [
                Operation(
                    1,
                    "op.rank0",
                    0,
                    "p",
                    "rank0",
                    "compute",
                    duration_ns=2,
                    participant_ranks=(0,),
                    dependencies_by_rank={0: ()},
                ),
                Operation(
                    2,
                    "op.p2p",
                    0,
                    "p",
                    "p2p",
                    "p2p",
                    payload_bytes=32,
                    src_rank=0,
                    dst_rank=1,
                    dependencies=(1,),
                    participant_ranks=(0, 1),
                    dependencies_by_rank={0: (1,), 1: ()},
                ),
                Operation(
                    3,
                    "op.rank1",
                    0,
                    "p",
                    "rank1",
                    "compute",
                    duration_ns=2,
                    dependencies=(2,),
                    participant_ranks=(1,),
                    dependencies_by_rank={1: (2,)},
                ),
            ]
            events, summaries = parse_astra_output(
                result, operations, ranks=2, astra_time_scale=1000
            )
            self.assertEqual([(row.rank, row.node_id) for row in events], [(0, 1), (0, 2), (1, 2), (1, 3)])
            receive = next(row for row in events if row.rank == 1 and row.node_id == 2)
            send = next(row for row in events if row.rank == 0 and row.node_id == 2)
            self.assertEqual(receive.duration_ns, 7)
            self.assertEqual(send.duration_ns, 5)
            self.assertEqual([row.wall_cycles for row in summaries], [7000, 9000])


if __name__ == "__main__":
    unittest.main()
