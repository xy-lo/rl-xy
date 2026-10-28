import copy
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('validate_gemm_cache_regimes',ROOT/'scripts/validate_gemm_cache_regimes.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
wrap_spec=importlib.util.spec_from_file_location('benchmark_gemm_capped',ROOT/'scripts/benchmark_gemm_capped.py')
wrapper=importlib.util.module_from_spec(wrap_spec);wrap_spec.loader.exec_module(wrapper)


def problem(m,n,k,direction='forward',output_bytes=2):
    return dict(m=m,n=n,k=k,batches=1,a_bytes=2,b_bytes=2,output_bytes=output_bytes,accumulator_bytes=4,direction=direction)


def selection():
    shapes=[problem(1,64,64),problem(512,64,64),problem(64,128,512,'dweight',4)]
    rows={probe.signature(p):p for p in shapes}
    return {'schema_version':'rlxy.gemm_cache_selection/1','shapes':rows,'ordered_sequence':list(rows),
        'sequence_origin':'unit fixture explicitly synthetic shape order'}


class GEMMCacheRegimeTests(unittest.TestCase):
    def test_wrapper_only_caps_fraction_and_never_increases_it(self):
        self.assertEqual(wrapper.capped_fraction(.25),.1)
        self.assertEqual(wrapper.capped_fraction(.1),.1)
        self.assertEqual(wrapper.capped_fraction(.05),.05)
        with self.assertRaises(ValueError):wrapper.capped_fraction(0)

    def test_categories_memory_and_per_occurrence_identity(self):
        source=selection();report=probe.validate_selection(source)
        self.assertFalse(report['missing_requested_categories'])
        self.assertEqual(report['selected_signature_count'],3)
        expected=sum(probe.allocation_bytes(p) for p in source['shapes'].values())
        self.assertEqual(report['sequence_tensor_bytes'],expected)
        source['ordered_sequence'].append(source['ordered_sequence'][0])
        report=probe.validate_selection(source)
        self.assertEqual(report['sequence_tensor_bytes'],expected+probe.allocation_bytes(next(iter(source['shapes'].values()))))

    def test_signature_tamper_and_full_corpus_are_rejected(self):
        source=selection();next(iter(source['shapes'].values()))['m']=2
        with self.assertRaises(ValueError):probe.validate_selection(source)
        source=selection()
        for m in range(20,33):
            p=problem(m,64,64);source['shapes'][probe.signature(p)]=p
        with self.assertRaisesRegex(ValueError,'12'):probe.validate_selection(source)

    def test_sequence_order_and_resident_budget_must_be_explicit(self):
        source=selection();source['ordered_sequence']=source['ordered_sequence'][:1]
        with self.assertRaisesRegex(ValueError,'cover'):probe.validate_selection(source)
        p=problem(65536,65536,65536);key=probe.signature(p)
        source={'schema_version':'rlxy.gemm_cache_selection/1','shapes':{key:p},'ordered_sequence':[key], 'sequence_origin':'synthetic oversized'}
        with self.assertRaisesRegex(ValueError,'6GiB'):probe.validate_selection(source)
        source=selection();source.pop('sequence_origin')
        with self.assertRaisesRegex(ValueError,'source'):probe.validate_selection(source)

    def test_timing_summary_preserves_both_repetitions_and_raw_samples(self):
        a=probe.summarize_samples([1000.,2000.,3000.]);b=probe.summarize_samples([2000.,3000.,4000.])
        result=probe.combine_repeats([a,b])
        self.assertEqual(result['median_of_repeat_medians_ns'],2500.)
        self.assertEqual(result['repeat_relative_difference'],1/3)
        self.assertEqual(result['repeats'][0]['raw_samples_ns'],[1000.,2000.,3000.])
        with self.assertRaises(ValueError):probe.summarize_samples([0])
        with self.assertRaises(ValueError):probe.combine_repeats([a])


if __name__=='__main__':unittest.main()
