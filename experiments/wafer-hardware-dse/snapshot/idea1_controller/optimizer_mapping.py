"""Map canonical TP2 optimizer operations to physical die IDs.

Only operation ranks are mapped here. Native Plan.append must regenerate
per-rank dependencies and communicator IDs after this translation.
"""
from dataclasses import replace


def optimizer_rank_map(pair_mapping):
    """Logical rank 2*w+lane -> actual physical die, preserving owner lane."""
    return tuple(die for pair in pair_mapping.pairs for die in pair)


def remap_optimizer_operation(operation, pair_mapping):
    lookup=optimizer_rank_map(pair_mapping)
    def convert(rank):
        if type(rank) is not int or not 0<=rank<len(lookup):
            raise ValueError('optimizer logical rank outside paired owners')
        return lookup[rank]
    ranks=tuple(convert(rank) for rank in operation.participant_ranks)
    if len(ranks)!=len(set(ranks)):raise ValueError('duplicate physical rank')
    kwargs={'participant_ranks':ranks,'dependencies_by_rank':{}}
    if operation.kind=='p2p':
        kwargs.update(src_rank=convert(operation.src_rank),dst_rank=convert(operation.dst_rank))
    return replace(operation,**kwargs)
