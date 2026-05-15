"""Pipeline audit logging — lightweight, ephemeral, per-run only."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class StageEvent:
    stage: str
    started_at: str
    duration_ms: float
    input_rows: int
    output_rows: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineAuditLog:
    run_id: str | None
    started_at: str
    stages: list[StageEvent] = field(default_factory=list)

    def add_stage(self, event: StageEvent) -> None:
        self.stages.append(event)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "stages": [vars(s) for s in self.stages],
        }


class AuditLogger:
    """Context manager that records a single pipeline stage."""

    def __init__(
        self,
        audit_log: PipelineAuditLog | None,
        stage: str,
        input_rows: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._audit_log = audit_log
        self._stage = stage
        self._input_rows = input_rows
        self._metadata = metadata or {}
        self._started_at: str = ""
        self._t0: float = 0.0
        self.output_rows: int = 0

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def __enter__(self) -> "AuditLogger":
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._t0 = time.monotonic()
        return self

    def __exit__(self, *_: object) -> None:
        if self._audit_log is None:
            return
        duration_ms = (time.monotonic() - self._t0) * 1000
        self._audit_log.add_stage(
            StageEvent(
                stage=self._stage,
                started_at=self._started_at,
                duration_ms=round(duration_ms, 2),
                input_rows=self._input_rows,
                output_rows=self.output_rows,
                metadata=self._metadata,
            )
        )
