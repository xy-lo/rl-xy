"""Frozen throughput placement, with task statistics computed once per search."""
import statistics
from idea1_controller.placement import largest_remainder
from idea1_controller.history import quantile


class HistoryPlacer:
    def __init__(self, public, forecasts, catalog):
        self.ids = [str(x['group_id']) for x in public]
        self.index = {g: i for i, g in enumerate(self.ids)}
        self.forecasts = forecasts
        self.catalog = catalog
        self.longs = [g for g in self.ids if forecasts[g]['label'] == 'proxy-long']
        self.long_set = set(self.longs)
        self.means = {g: statistics.mean(sum(v) for v in forecasts[g]['response_scenarios']) for g in self.ids}
        self.tails = {g: quantile([max(v) for v in forecasts[g]['response_scenarios']], .9) for g in self.ids}
        self.long_order = sorted(self.longs, key=lambda g: (-forecasts[g]['score'], self.index[g]))
        self.other_order = sorted((g for g in self.ids if g not in self.long_set),
                                  key=lambda g: (-forecasts[g]['score'], self.index[g]))

    def assign(self, inference):
        groups = {int(r): tuple(ws) for r, ws in inference['groups'].items()}
        roots = sorted(groups)
        tp = {r: 1 if inference.get('paired_tp1') else 2*len(groups[r]) for r in roots}
        caps = {r: int(inference['caps'][str(r)]) for r in roots}
        rate = {r: caps[r]/self.catalog.decode(tp[r], caps[r], 4096) for r in roots}
        minimum = min(rate.values())
        weights = [max(1, round(rate[r]/minimum*1e6)) for r in roots]
        quota = dict(zip(roots, largest_remainder(len(self.ids), weights)))
        longquota = dict(zip(roots, largest_remainder(len(self.longs), weights)))
        longquota = {r: min(longquota[r], quota[r]) for r in roots}
        while sum(longquota.values()) < len(self.longs):
            r = min((r for r in roots if longquota[r] < quota[r]),
                    key=lambda r: ((longquota[r]+1)/rate[r], r))
            longquota[r] += 1
        queues = {r: [] for r in roots}
        work = {r: 0. for r in roots}
        tail = {r: 0. for r in roots}
        left = longquota.copy()
        shortcost = {r: self.catalog.decode(tp[r], min(8, caps[r]), 4096) for r in roots}
        def score(r, g):
            return (work[r]+self.means[g])/rate[r] + .5*max(tail[r], self.tails[g])*shortcost[r]
        def push(r, g):
            queues[r].append(g)
            work[r] += self.means[g]
            tail[r] = max(tail[r], self.tails[g])
        for g in self.long_order:
            r = min((r for r in roots if left[r]), key=lambda r: (score(r, g), r))
            push(r, g)
            left[r] -= 1
        for g in self.other_order:
            r = min((r for r in roots if len(queues[r]) < quota[r]), key=lambda r: (score(r, g), r))
            push(r, g)
        for r in roots:
            queues[r].sort(key=lambda g: (-self.forecasts[g]['score'], self.index[g]))
        assert sorted(g for gs in queues.values() for g in gs) == sorted(self.ids)
        assert all(sum(g in self.long_set for g in queues[r]) == longquota[r] for r in roots)
        return {str(r): gs for r, gs in queues.items()}, dict(
            long_quota={str(r): q for r, q in longquota.items()},
            group_quota={str(r): q for r, q in quota.items()},
            scope='actual inference groups; independent of retained training-home quotas')
