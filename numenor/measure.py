from attr import define, field, Factory
from typing import *


@define
class Metric:
    executor: Callable
    method: Optional[str] = None

    def __call__(self, label, prediction, **kwargs):
        self.measurement = self.callable(label, prediction)


@define
class Performance:
    metrics: List[Metric]
