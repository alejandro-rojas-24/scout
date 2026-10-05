"""Exception hierarchy for scout. Each error carries a process exit code."""

from __future__ import annotations


class ScoutError(Exception):
    """Base class for all scout errors."""

    exit_code: int = 1


class WeightReadRefused(ScoutError):
    """A read would have touched weight bytes (Phase 1 is header-only)."""

    exit_code = 5


class ReadThresholdExceeded(ScoutError):
    """A non-weight read would push total read bytes over the configured threshold."""

    exit_code = 5

    def __init__(self, requested: int, total: int, threshold: int, message: str | None = None) -> None:
        self.requested = requested
        self.total = total
        self.threshold = threshold
        if message is None:
            message = (
                f"threshold: requested={requested} total={total} threshold={threshold} "
                f"disk=0 reason=non-weight read budget"
            )
        super().__init__(message)

    def __reduce__(self):  # keep pickling working with the custom signature
        return (type(self), (self.requested, self.total, self.threshold, str(self)))


class StageOrderError(ScoutError):
    """Stages were entered out of order."""


class GatedRepoError(ScoutError):
    exit_code = 3


class RepoNotFoundError(ScoutError):
    exit_code = 3


class RevisionMismatch(ScoutError):
    pass


class NetworkError(ScoutError):
    exit_code = 4


class RangeNotSupported(ScoutError):
    exit_code = 4


class HubAPIError(ScoutError):
    """Unexpected Hub API status or malformed response body."""

    exit_code = 4


class HeaderError(ScoutError):
    pass


class NoSafetensorsError(ScoutError):
    pass


class AmbiguousWeightsError(ScoutError):
    pass


class DuplicateTensorError(ScoutError):
    pass
