"""Exact event-heap equivalent of the frozen decode planning surrogate.

This changes host bookkeeping only. Operator services, scenarios, admission,
request order, prefill, and the score contract remain unchanged.
"""
import heapq
from collections import deque
from idea1_controller.inference_search import InferenceScenarioScorer


class HeapScenarioScorer(InferenceScenarioScorer):
    def replica(self, gids, public, forecast, scenario, tp, cap):
        key = (tp, cap, tuple((g, public[g]['prompt_length'],
                tuple(forecast[g]['response_scenarios'][scenario % len(forecast[g]['response_scenarios'])])) for g in gids))
        if key in self.replica_cache:
            return self.replica_cache[key]
        queue = deque((public[g]['prompt_length'], int(r)) for g in gids
                      for r in forecast[g]['response_scenarios'][scenario % len(forecast[g]['response_scenarios'])])
        active = []
        decoded = offset_sum = 0
        elapsed = 0.0
        while queue or active:
            if queue and len(active) < cap:
                admitted = []
                while queue and len(active) + len(admitted) < cap:
                    admitted.append(queue.popleft())
                elapsed += self.catalog.phase('prefill', tp, [p for p, _ in admitted])
                for prompt, response in admitted:
                    if response > 1:
                        offset = prompt + 1 - decoded
                        heapq.heappush(active, (decoded + response - 1, offset))
                        offset_sum += offset
            else:
                batch = len(active)
                finish = active[0][0]
                steps = finish - decoded
                context = (batch * decoded + offset_sum) / batch
                intercept, slope = self.catalog.coefficients('decode', tp, batch)
                a = max(1, intercept + slope * context)
                b = max(1, intercept + slope * (context + 1))
                elapsed += steps * a + steps * (steps - 1) / 2 * (b - a)
                decoded = finish
                while active and active[0][0] == decoded:
                    _, offset = heapq.heappop(active)
                    offset_sum -= offset
        self.replica_cache[key] = elapsed
        return elapsed
