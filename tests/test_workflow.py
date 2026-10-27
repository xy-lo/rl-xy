from __future__ import annotations

import unittest
from pathlib import Path

from rl_xy.config import load_case
from rl_xy.workflow import REQUIRED_PHASES, build_workflow


ROOT = Path(__file__).resolve().parents[1]


class WorkflowTest(unittest.TestCase):
    def test_full_round_semantics_are_explicit(self) -> None:
        case = load_case(ROOT / "configs/toy_grpo_2die/case.json")
        ops = build_workflow(case)
        self.assertEqual(set(REQUIRED_PHASES), {item.phase for item in ops})
        self.assertEqual([item.node_id for item in ops], list(range(1, len(ops) + 1)))
        last_by_rank = {0: None, 1: None}
        for op in ops:
            for rank in op.participant_ranks:
                expected = () if last_by_rank[rank] is None else (last_by_rank[rank],)
                self.assertEqual(op.dependencies_by_rank[rank], expected)
                last_by_rank[rank] = op.node_id
        names = {item.name for item in ops}
        for phase in ("train_A", "train_B"):
            self.assertTrue(any(name.startswith(f"{phase}_backward") and name.endswith("embedding_backward") for name in names))
            self.assertTrue(any(name.startswith(f"{phase}_backward") and name.endswith("final_rmsnorm_backward") for name in names))
        optimizer = next(item for item in ops if item.name == "adamw_read_modify_write")
        self.assertEqual(optimizer.metadata["reads"], ["gradient", "master", "adam_m", "adam_v"])
        self.assertEqual(optimizer.metadata["writes"], ["master", "adam_m", "adam_v", "actor"])
        self.assertEqual(optimizer.metadata["actor_version_out"], 1)
        rank0_only = next(item for item in ops if item.name == "host_to_rank0_prompt")
        self.assertEqual(rank0_only.participant_ranks, (0,))
        embeddings = [item for item in ops if item.name.endswith(".embedding")]
        self.assertEqual(len(embeddings), 11)
        for item in embeddings:
            following = ops[item.node_id]
            self.assertEqual(following.name, item.name + "_tp_sum")
            self.assertEqual(following.collective, "ALL_REDUCE")
        reset = next(item for item in ops if item.name == "zero_gradient_buffer")
        first_train = next(item for item in ops if item.name == "train_A_forward.embedding")
        self.assertLess(reset.node_id, first_train.node_id)

    def test_two_rounds_publish_v2(self) -> None:
        case = load_case(ROOT / "configs/toy_grpo_2die/case.json", rounds_override=2)
        ops = build_workflow(case)
        published = [item.metadata["published_version"] for item in ops if item.name == "publish_actor_version"]
        self.assertEqual(published, [1, 2])


if __name__ == "__main__":
    unittest.main()
