"""Alert primitives shared by the monitoring modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

PASS, WARNING, FAIL = "PASS", "WARNING", "FAIL"
_RANK = {PASS: 0, WARNING: 1, FAIL: 2}


@dataclass
class Alert:
    check: str
    status: str
    value: object
    threshold: object = None
    detail: str = ""

    def to_dict(self) -> dict:
        return {"check": self.check, "status": self.status, "value": self.value,
                "threshold": self.threshold, "detail": self.detail}


@dataclass
class AlertSet:
    section: str
    alerts: List[Alert] = field(default_factory=list)

    def add(self, check: str, status: str, value, threshold=None, detail: str = ""):
        self.alerts.append(Alert(check, status, value, threshold, detail))

    def gauge(self, check: str, value: float, warn: Optional[float] = None,
              fail: Optional[float] = None, detail: str = "") -> None:
        """Threshold a numeric metric into PASS/WARNING/FAIL."""
        status = PASS
        if warn is not None and value >= warn:
            status = WARNING
        if fail is not None and value >= fail:
            status = FAIL
        self.add(check, status, value, fail if fail is not None else warn, detail)

    @property
    def status(self) -> str:
        if not self.alerts:
            return PASS
        return {0: PASS, 1: WARNING, 2: FAIL}[max(_RANK[a.status] for a in self.alerts)]

    def worse_than(self, other: str) -> bool:
        return _RANK[self.status] > _RANK[other]


def combine(*sets: AlertSet) -> str:
    ranks = [_RANK[s.status] for s in sets if s is not None]
    return {0: PASS, 1: WARNING, 2: FAIL}[max(ranks)] if ranks else PASS
