"""Audit published 54-die stage summaries against retained event evidence."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'experiments/2026-11-02-h0-54'

def read(path):
    return json.loads(path.read_text())

class H0SummaryTests(unittest.TestCase):
    def test_paired_stage_accounting(self):
        summary = read(E / 'results.json')
        self.assertEqual(summary['status'], 'PASS_PAIRED_SIMULATION')
        for row in summary['rows']:
            with self.subTest(model=row['model']):
                d = E / 'evidence' / row['model']
                baseline = read(d / 'baseline-v1/result.json')
                idea = read(d / 'idea-v1/result.json')
                audit = read(d / 'idea-v1/audit-h0.json')
                self.assertEqual(baseline['status'], 'PASS')
                self.assertEqual(audit['status'], 'PASS_LOGICAL_54_DIE_TP2_TP4')
                self.assertEqual(idea['requests'], 1536)
                self.assertEqual(idea['completed'], 1536)
                self.assertEqual(idea['ranks'], 54)
                self.assertEqual(baseline['trajectory_sha256'], idea['canonical_requests_sha256'])
                self.assertEqual(len(baseline['stage_ns']), 9)
                self.assertEqual(len(baseline['native_proof']['training']['per_rank_costs']), 54)
                self.assertAlmostEqual(row['wafer_baseline_s'], baseline['step_model_ns'] / 1e9)
                recomposed = (baseline['step_model_ns'] - baseline['stage_ns']['generation']) / 1e9 + idea['generation_s']
                self.assertAlmostEqual(row['idea_s'], recomposed)
                for phase in baseline['native_proof'].values():
                    self.assertEqual(phase['binary_sha256'], idea['binary_sha256'])

    def test_14b_reference_resharding(self):
        ops = read(E / 'evidence/14B/baseline-v1/reference-operations.json')
        gathers = [x for x in ops if 'restore_reference_shards' in x.get('name', '') and x.get('collective') == 'ALL_GATHER']
        self.assertEqual(len(gathers), 2)
        self.assertTrue(all(len(x['ranks']) == 27 and x['payload_bytes'] == 546990084 for x in gathers))
