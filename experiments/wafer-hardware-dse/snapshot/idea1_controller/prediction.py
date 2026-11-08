"""创新2接口。只读历史快照，不接收当前未完成轨迹真值。均未实现。"""
from typing import Mapping, Protocol, Sequence
from .contracts import AssignmentPlan, HistorySnapshot, ModelContract, PlacementPlan, Prediction, RequestView, RuntimeEvent

class HistoryStore(Protocol):
    def snapshot(self, before_iteration: int, model_signature: str, decoding_signature: str) -> HistorySnapshot: ...
    def observe_completed(self, event: RuntimeEvent) -> None: ...

class WorkPredictor(Protocol):
    def predict(self, requests: Sequence[RequestView], history: HistorySnapshot, model: ModelContract) -> Mapping[str, Prediction]: ...

class QuotaTaskAssigner(Protocol):
    def assign(self, requests: Sequence[RequestView], predictions: Mapping[str, Prediction], placement: PlacementPlan) -> AssignmentPlan: ...
    # MVP按完整GRPO组设home region；区域内排列回答任务，不拆group语义。

class BaselineAssigner(Protocol):
    def assign(self, requests: Sequence[RequestView], placement: PlacementPlan) -> AssignmentPlan: ...
    # 冷启动、低置信度、版本/解码分布漂移的确定性回退。
