import attr
from typing import *


@attr.s(auto_attribs=True)
class Metric:
    executor: Callable
    method: Optional[str] = None

    def __call__(self, label, prediction, **kwargs):
        self.measurement = self.callable(label, prediction)


@attr.s(auto_attribs=True)
class Performance:
    metrics: List[Metric]
