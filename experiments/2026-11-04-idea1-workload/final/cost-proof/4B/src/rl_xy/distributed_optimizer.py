"""PP1/no-overlap distributed-optimizer engineering contract.

FP32 optimizer states are DP-sharded. The reduced main-gradient shard is a
VIEW of the full FP32 gradient buffer: it must never be allocated twice.
This models storage/work/native collectives, not a numerical training driver.
"""
from dataclasses import asdict, dataclass
from fractions import Fraction
from math import lcm

from .model import enumerate_parameters
from .types import Operation

MCORE_REFERENCE = ('https://github.com/NVIDIA/Megatron-LM/blob/core_v0.12.2/'
    'megatron/core/distributed/param_and_grad_buffer.py')


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + ' must be a positive integer')


def _pad(value, divisor):
    return ((value + divisor - 1) // divisor) * divisor


@dataclass(frozen=True)
class ParameterSpan:
    parameter_id: str
    start: int
    end: int
    replicated_across_tp: bool

    @property
    def elements(self):
        return self.end - self.start


@dataclass(frozen=True)
class DistributedOptimizerLayout:
    dp_size: int
    tp_size: int
    parameters: tuple[ParameterSpan, ...]
    padded_elements: int
    parameter_alignment: int = 64
    bucket_alignment: int = 128

    @property
    def valid_elements(self):
        return sum(p.elements for p in self.parameters)

    @property
    def padding_elements(self):
        return self.padded_elements - self.valid_elements

    @property
    def shard_elements(self):
        return self.padded_elements // self.dp_size

    def shard_interval(self, dp_rank):
        if type(dp_rank) is not int or not 0 <= dp_rank < self.dp_size:
            raise ValueError('DP rank outside optimizer group')
        return dp_rank * self.shard_elements, (dp_rank + 1) * self.shard_elements

    def intersections(self, dp_rank, *, tp_rank=0, norm_only=False):
        if type(tp_rank) is not int or not 0 <= tp_rank < self.tp_size:
            raise ValueError('TP rank outside tensor group')
        left, right = self.shard_interval(dp_rank)
        rows = []
        for p in self.parameters:
            a, b = max(left, p.start), min(right, p.end)
            if b <= a or (norm_only and p.replicated_across_tp and tp_rank != 0):
                continue
            rows.append({'parameter_id': p.parameter_id, 'buffer_start': a, 'buffer_end': b,
                'parameter_local_start': a - p.start, 'parameter_local_end': b - p.start,
                'shard_local_start': a - left, 'shard_local_end': b - left,
                'elements': b - a, 'replicated_across_tp': p.replicated_across_tp})
        return rows

    def shard_valid_elements(self, dp_rank, *, tp_rank=0, norm_only=False):
        return sum(x['elements'] for x in self.intersections(dp_rank, tp_rank=tp_rank, norm_only=norm_only))

    def state_bytes(self, dp_rank, *, include_reference=True):
        """Physical allocations versus logical views, with one full bucket.

        Actor includes buffer padding; the frozen reference is independent,
        has valid model elements only, and has no optimizer/grad buffer.
        """
        start, end = self.shard_interval(dp_rank)
        owned = self.shard_valid_elements(dp_rank)
        allocated = {'actor_BF16_parameter_buffer': 2 * self.padded_elements,
            'full_FP32_gradient_working_buffer': 4 * self.padded_elements,
            'master_FP32_shard': 4 * owned,
            'adam_m_FP32_shard': 4 * owned,
            'adam_v_FP32_shard': 4 * owned}
        if include_reference:
            allocated['reference_BF16_valid_parameters'] = 2 * self.valid_elements
        return {'allocations': allocated, 'total_allocated_bytes': sum(allocated.values()),
            'main_gradient_shard_view': {'alias_of': 'full_FP32_gradient_working_buffer',
                'byte_start': start * 4, 'byte_end': end * 4, 'logical_bytes': (end - start) * 4,
                'additional_allocated_bytes': 0},
            'parameter_shard_view': {'alias_of': 'actor_BF16_parameter_buffer',
                'byte_start': start * 2, 'byte_end': end * 2, 'additional_allocated_bytes': 0},
            'full_working_gradient_buffer_retained': True,
            'optimizer_state_only_valid_parameter_intersections': True}

    def work(self, dp_rank, *, tp_rank=0, optimizer_ops_per_parameter=15):
        valid = self.shard_valid_elements(dp_rank, tp_rank=tp_rank)
        norm = self.shard_valid_elements(dp_rank, tp_rank=tp_rank, norm_only=True)
        return {'local_pre_RS_scale_elements': self.padded_elements,
            'ring_RS_local_reduction_adds': self.shard_elements * (self.dp_size - 1),
            'gradient_norm_unique_elements': norm, 'gradient_norm_read_bytes': norm * 4,
            'gradient_clip_elements': valid, 'gradient_clip_read_bytes': valid * 4,
            'gradient_clip_write_bytes': valid * 4,
            'adam_valid_elements': valid, 'adam_vector_ops': valid * optimizer_ops_per_parameter,
            'adam_read_bytes': valid * 16, 'adam_write_bytes_including_BF16_actor_shard': valid * 14,
            'padding_is_zero_and_excluded_from_norm_and_Adam': True}

    def collective_sizes(self):
        # Astra convention must be checked with the tiny native fixture:
        # comm_size is the full RS input / PER-RANK AG INPUT shard size.
        return {'reduce_scatter': {'collective': 'REDUCE_SCATTER',
                    'full_input_bytes_per_rank': self.padded_elements * 4,
                    'output_shard_bytes_per_rank': self.shard_elements * 4,
                    'native_full_buffer_payload_bytes': self.padded_elements * 4},
                'parameter_all_gather': {'collective': 'ALL_GATHER',
                    'input_shard_bytes_per_rank': self.shard_elements * 2,
                    'full_output_bytes_per_rank': self.padded_elements * 2,
                    'native_input_shard_payload_bytes': self.shard_elements * 2},
                'ideal_ring_wire_bytes_per_rank': {
                    'gradient_RS': self.shard_elements * (self.dp_size - 1) * 4,
                    'parameter_AG': self.shard_elements * (self.dp_size - 1) * 2},
                'payloads_include_padding': True,
                'network_latency_endpoint_service_and_reduction_work_not_in_wire_byte_formula': True}

    def native_collectives(self, participants, *, group_id, round_id, actor_version, label='do'):
        participants = tuple(participants)
        if len(participants) != self.dp_size or len(set(participants)) != self.dp_size:
            raise ValueError('native optimizer participants must match the exact DP group')
        if type(group_id) is not int or group_id < 1:
            raise ValueError('explicit positive native communicator group ID required')
        common = {'comm_group': group_id, 'distributed_optimizer': True,
            'overlap_grad_reduce': False, 'overlap_param_gather': False,
            'padded_elements': self.padded_elements, 'shard_elements': self.shard_elements,
            'padding_gradient_value': 0}
        rs = Operation(0, f'{label}.gradient_RS', round_id, 'dp_sync', 'do_gradient_reduce_scatter',
            'collective', payload_bytes=self.padded_elements * 4, collective='REDUCE_SCATTER',
            participant_ranks=participants,
            metadata={**common, 'transport_payload_bytes': self.padded_elements * 4,
                'dtype': 'fp32', 'native_reduction': 'SUM',
                'required_pre_RS_gradient_scale_numerator': 1,
                'required_pre_RS_gradient_scale_denominator': self.dp_size,
                'gradient_actor_version': actor_version,
                'output_is_view_of_full_gradient_buffer': True})
        ag = Operation(0, f'{label}.parameter_AG', round_id, 'optimizer_sync', 'do_parameter_all_gather',
            'collective', payload_bytes=self.shard_elements * 2, collective='ALL_GATHER',
            participant_ranks=participants,
            metadata={**common, 'transport_payload_bytes': self.shard_elements * 2,
                'native_payload_caliber': 'per_rank_input_shard',
                'logical_full_output_bytes_per_rank': self.padded_elements * 2,
                'dtype': 'bf16', 'requires_completed_shard_Adam_update': True,
                'parameter_version_in': actor_version, 'parameter_version_out': actor_version + 1,
                'writes_existing_parameter_buffer_not_an_additional_model_copy': True})
        return rs, ag

    def contract(self):
        return {'schema_version': 'rlxy.distributed_optimizer/1', 'dp_size': self.dp_size,
            'tp_size': self.tp_size, 'valid_elements': self.valid_elements,
            'padded_elements': self.padded_elements, 'padding_elements': self.padding_elements,
            'shard_elements': self.shard_elements, 'parameters': [asdict(p) for p in self.parameters],
            'parameter_start_alignment_elements': self.parameter_alignment,
            'bucket_end_divisor_elements': lcm(self.dp_size, self.bucket_alignment),
            'overlap_grad_reduce': False, 'overlap_param_gather': False,
            'offload_param': False, 'offload_optimizer': False,
            'one_full_gradient_bucket': True, 'pad_buckets_for_high_nccl_busbw': False,
            'collective_sizes': self.collective_sizes(), 'reference': MCORE_REFERENCE,
            'scope': 'PP1 dense synchronous engineering model, not a training driver',
            'registration_order_limit': 'reverse coalesced enumerate_parameters rows; not an assertion of exact framework parameter registration or bucket ordering'}


def build_layout(parameter_rows, *, dp_size, tp_size=1, parameter_alignment=64, bucket_alignment=128):
    for value, name in [(dp_size, 'DP'), (tp_size, 'TP'),
            (parameter_alignment, 'parameter alignment'), (bucket_alignment, 'bucket alignment')]:
        _positive(value, name)
    rows = tuple(parameter_rows)
    if not rows or len({p.parameter_id for p in rows}) != len(rows):
        raise ValueError('nonempty unique parameter rows required')
    spans = [];offset = 0
    for p in reversed(rows):
        _positive(p.local_elements_per_die, 'local parameter size')
        offset = _pad(offset, parameter_alignment)
        spans.append(ParameterSpan(p.parameter_id, offset, offset + p.local_elements_per_die, p.replicated))
        offset += p.local_elements_per_die
    padded = _pad(offset, lcm(dp_size, bucket_alignment))
    return DistributedOptimizerLayout(dp_size, tp_size, tuple(spans), padded,
        parameter_alignment, bucket_alignment)


def layout_for_case(case, *, dp_size):
    if case.strategy.pipeline_parallel != 1:
        raise ValueError('this helper supports PP1 only')
    return build_layout(enumerate_parameters(case), dp_size=dp_size, tp_size=case.strategy.tensor_parallel)


def global_loss_scaling(dp_size, valid_response_tokens):
    _positive(dp_size, 'DP');_positive(valid_response_tokens, 'global valid response token denominator')
    return {'local_token_sum_loss_multiplier': str(Fraction(dp_size, valid_response_tokens)),
        'pre_RS_scale_for_native_SUM': str(Fraction(1, dp_size)),
        'net_global_token_sum_multiplier': str(Fraction(1, valid_response_tokens)),
        'reference': 'veRL agg_loss token-mean multiplies local token sum by DP/global_token_count; DP gradient mean cancels DP'}


def validate_publication(*, actor_version, gradient_version, reference_version,
        completed_parameter_all_gather_tp_groups, tp_size):
    if gradient_version != actor_version:
        raise ValueError('gradient behavior/actor version mismatch')
    _positive(tp_size, 'TP')
    completed = tuple(completed_parameter_all_gather_tp_groups)
    if len(completed) != tp_size or set(completed) != set(range(tp_size)):
        raise ValueError('cannot publish before all TP-column DP parameter gathers finish')
    return {'actor_version_in': actor_version, 'actor_version_out': actor_version + 1,
        'reference_version_in': reference_version, 'reference_version_out': reference_version,
        'requires_global_active_rank_publication_fence': True}
