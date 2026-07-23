from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any


@dataclass(slots=True)
class StageMetric:
    name: str
    duration_seconds: float
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class RunMetrics:
    stages: list[StageMetric] = field(default_factory=list)

    def add(self, name: str, started_at: float, result: Any | None = None) -> None:
        usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
        self.stages.append(
            StageMetric(
                name=name,
                duration_seconds=round(perf_counter() - started_at, 3),
                requests=int(getattr(usage, "requests", 0) or 0),
                input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [asdict(stage) for stage in self.stages],
            "totals": {
                "duration_seconds": round(sum(stage.duration_seconds for stage in self.stages), 3),
                "requests": sum(stage.requests for stage in self.stages),
                "input_tokens": sum(stage.input_tokens for stage in self.stages),
                "output_tokens": sum(stage.output_tokens for stage in self.stages),
                "total_tokens": sum(stage.total_tokens for stage in self.stages),
            },
        }
