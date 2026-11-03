"""Independent predicted costs and hard execution/label/token reservations.

Prediction scalars rank purchase actions; they never become actual spend.
Execution adapters must enforce the reserved per-call token/call caps. The
ledger refuses settlement above a reservation and keeps it locked on error.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from numbers import Integral
import math


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class ResourceCost:
    executions: int = 0
    labels: int = 0
    tokens: int = 0

    def __post_init__(self):
        if any(not isinstance(v, Integral) or isinstance(v, bool) or v < 0 for v in (self.executions, self.labels, self.tokens)):
            raise ValueError("Hard resource counts must be nonnegative integers")

    def __add__(self, other):
        return ResourceCost(self.executions + other.executions, self.labels + other.labels, self.tokens + other.tokens)

    def __sub__(self, other):
        if not other.fits_within(self):
            raise BudgetExceeded("Negative resource balance")
        return ResourceCost(self.executions - other.executions, self.labels - other.labels, self.tokens - other.tokens)

    def __mul__(self, count):
        if not isinstance(count, Integral) or count < 0:
            raise ValueError("Multiplier must be a nonnegative integer")
        return ResourceCost(self.executions * count, self.labels * count, self.tokens * count)

    __rmul__ = __mul__

    def fits_within(self, upper):
        return all(a <= b for a, b in zip((self.executions, self.labels, self.tokens), (upper.executions, upper.labels, upper.tokens)))


@dataclass(frozen=True)
class CostPrediction:
    scalar: float
    upper_bound: ResourceCost

    def __post_init__(self):
        if not math.isfinite(self.scalar) or self.scalar <= 0:
            raise ValueError("Predicted incremental scalar cost must be positive")
        if not isinstance(self.upper_bound, ResourceCost):
            raise TypeError("An enforceable hard reservation upper bound is required")


class BudgetLedger:
    def __init__(self, limits):
        if not isinstance(limits, ResourceCost):
            raise TypeError("limits must be a ResourceCost")
        self.limits = limits
        self.spent = ResourceCost()
        self.future_service = ResourceCost()
        self._pending = {}
        self._completed = {}

    @property
    def reserved(self):
        result = ResourceCost()
        for upper in self._pending.values():
            result += upper
        return result

    @property
    def available(self):
        return self.limits - self.spent - self.reserved - self.future_service

    @property
    def remaining(self):
        return self.limits - self.spent

    def reserve_future_service(self, upper_bound):
        """Protect the sum of upper bounds for all future normal services.

        Set before offering audits. Natural main feedback belongs in these
        service bounds; it is not free from the total label budget.
        """
        if not upper_bound.fits_within(self.limits - self.spent - self.reserved):
            raise BudgetExceeded("Cannot protect future normal service")
        self.future_service = upper_bound

    def can_reserve(self, upper_bound):
        return upper_bound.fits_within(self.available)

    def reserve(self, reservation_id, upper_bound):
        if reservation_id in self._pending or reservation_id in self._completed:
            raise ValueError("Reservation IDs cannot be reused")
        if not isinstance(upper_bound, ResourceCost):
            raise TypeError("upper_bound must be a ResourceCost")
        if not self.can_reserve(upper_bound):
            raise BudgetExceeded("Insufficient unprotected execution/label/token budget")
        self._pending[reservation_id] = upper_bound
        return upper_bound

    def reserve_service(self, reservation_id, upper_bound):
        """Move one service's upper bound from protected future to pending."""
        old = self.future_service
        self.future_service = old - upper_bound
        try:
            return self.reserve(reservation_id, upper_bound)
        except Exception:
            self.future_service = old
            raise

    def settle(self, reservation_id, actual_cost):
        upper = self._pending[reservation_id]
        if not isinstance(actual_cost, ResourceCost):
            raise TypeError("Actual spend must be reported separately as ResourceCost")
        if not actual_cost.fits_within(upper):
            raise BudgetExceeded("Actual cost exceeded its hard reservation; execution cap violation")
        self.spent += actual_cost
        del self._pending[reservation_id]
        self._completed[reservation_id] = actual_cost
        return upper - actual_cost

    def cancel(self, reservation_id):
        """Release an unexecuted reservation; never use to erase actual spend."""
        return self._pending.pop(reservation_id)

    def snapshot(self):
        return {"limits": asdict(self.limits), "spent": asdict(self.spent),
                "reserved": asdict(self.reserved), "future_service": asdict(self.future_service),
                "available": asdict(self.available),
                "pending": {k: asdict(v) for k, v in self._pending.items()},
                "completed": {k: asdict(v) for k, v in self._completed.items()}}
