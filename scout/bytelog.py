"""Staged byte log: every fetched byte is classified (meta/header/weight) and logged.

No network or file IO happens here. Callers preflight a read (which refuses weight
bytes and enforces the non-weight read budget via reservations) and record the body
bytes actually yielded once the read completes.
"""

from __future__ import annotations

import enum
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable

from scout.errors import ReadThresholdExceeded, StageOrderError, WeightReadRefused


class Stage(enum.IntEnum):
    RESOLVE = 1
    HEADERS = 2
    META = 3
    SAMPLED_READ = 4
    FULL_DOWNLOAD = 5
    COMPUTE = 6
    COMPARE = 7
    REPORT = 8
    PURGE = 9


class ByteClass(str, enum.Enum):
    META = "meta"
    HEADER = "header"
    WEIGHT = "weight"


WEIGHT_EXTENSIONS: tuple[str, ...] = (
    ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf",
    ".h5", ".msgpack", ".onnx", ".pb", ".npz", ".pkl",
)
DEFAULT_THRESHOLD_BYTES: int = 64 * 1024 * 1024
META_MAX_BYTES: int = 16 * 1024 * 1024  # reservation for reads of unknown length
COUNTING: str = "http-body-bytes-yielded"

_RECORD_EVENTS = frozenset({"fetch", "retry", "error"})
_NOTE_EVENTS = frozenset({"error", "refused", "card_written"})


def _zero() -> dict[str, int]:
    return {"meta": 0, "header": 0, "weight": 0}


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def classify_range(path: str, start: int, end_exclusive: int, header_len: int | None) -> dict[str, int]:
    """Split the byte range [start, end_exclusive) of `path` into meta/header/weight counts."""
    if start < 0 or end_exclusive < start:
        raise ValueError(f"invalid range [{start}, {end_exclusive})")
    out = _zero()
    n = end_exclusive - start
    p = path.lower()
    if p.endswith(".safetensors"):
        header_end = 8 + header_len if header_len is not None else 8
        out["header"] = _overlap(start, end_exclusive, 0, header_end)
        out["weight"] = n - out["header"]
    elif p.endswith(WEIGHT_EXTENSIONS):
        out["weight"] = n
    else:
        out["meta"] = n
    return out


@dataclass(frozen=True)
class LogEvent:
    seq: int
    ts: float
    stage: str
    event: str
    source: str | None
    path: str | None
    url: str | None
    range: tuple[int, int] | None
    status: int | None
    bytes: int
    bytes_by_class: Mapping[str, int]  # read-only MappingProxyType over a private copy
    attempt: int
    elapsed_ms: float
    note: str | None
    totals: Mapping[str, int]  # read-only MappingProxyType over a private snapshot

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "stage": self.stage,
            "event": self.event,
            "source": self.source,
            "path": self.path,
            "url": self.url,
            "range": list(self.range) if self.range is not None else None,
            "status": self.status,
            "bytes": self.bytes,
            "bytes_by_class": dict(self.bytes_by_class),
            "attempt": self.attempt,
            "elapsed_ms": self.elapsed_ms,
            "note": self.note,
            "totals": dict(self.totals),
        }


class ByteLog:
    """Thread-safe staged log of every byte fetched."""

    def __init__(
        self,
        threshold_bytes: int = DEFAULT_THRESHOLD_BYTES,
        on_event: Callable[[LogEvent], None] | None = None,
    ) -> None:
        self.threshold_bytes = threshold_bytes
        self._on_event = on_event
        # Lock order is always _emit_lock -> _lock. _lock guards state and is never held
        # while on_event runs. _emit_lock (re-entrant) is held from appending an event
        # until its callback returns, so callbacks see seq 1..n in order and
        # record() called from inside on_event (same thread) does not deadlock.
        self._emit_lock = threading.RLock()
        self._lock = threading.Lock()
        self._seq = 0
        self._stage: Stage | None = None
        self._totals = _zero()
        self._events: list[LogEvent] = []
        self._header_lens: dict[str, int] = {}
        self._reservations: dict[int, int] = {}
        self._next_reservation = 1

    # ------------------------------------------------------------------ views

    @property
    def current_stage(self) -> Stage | None:
        with self._lock:
            return self._stage

    @property
    def totals(self) -> dict[str, int]:
        with self._lock:
            return dict(self._totals)

    @property
    def events(self) -> list[LogEvent]:
        with self._lock:
            return list(self._events)

    def events_since(self, seq: int) -> list[LogEvent]:
        """Events with seq strictly greater than `seq`."""
        with self._lock:
            return [e for e in self._events if e.seq > seq]

    @property
    def reserved(self) -> int:
        with self._lock:
            return sum(self._reservations.values())

    # ------------------------------------------------------------------ internals

    def _append_locked(self, **fields) -> LogEvent:
        """Build and append an event. Caller must hold self._lock."""
        self._seq += 1
        fields["bytes_by_class"] = MappingProxyType(dict(fields["bytes_by_class"]))
        ev = LogEvent(
            seq=self._seq,
            ts=time.time(),
            stage=self._stage.name if self._stage is not None else "NONE",
            totals=MappingProxyType(dict(self._totals)),
            **fields,
        )
        self._events.append(ev)
        return ev

    def _emit(self, ev: LogEvent) -> LogEvent:
        """Run on_event. Caller must hold self._emit_lock (and not self._lock)."""
        if self._on_event is not None:
            self._on_event(ev)
        return ev

    def _zero_event_locked(self, event: str, note: str | None, path: str | None) -> LogEvent:
        return self._append_locked(
            event=event, source=None, path=path, url=None, range=None, status=None,
            bytes=0, bytes_by_class=_zero(), attempt=1, elapsed_ms=0.0, note=note,
        )

    # ------------------------------------------------------------------ stages

    def stage(self, stage: Stage) -> LogEvent:
        stage = Stage(stage)
        with self._emit_lock:
            with self._lock:
                if self._stage is not None and stage < self._stage:
                    raise StageOrderError(
                        f"cannot enter stage {stage.name} after {self._stage.name}"
                    )
                self._stage = stage
                ev = self._zero_event_locked("stage_start", None, None)
            return self._emit(ev)

    # ------------------------------------------------------------------ header lengths

    def set_header_len(self, path: str, header_len: int) -> None:
        """Register the safetensors header length N for `path`.

        Re-registering the same value is a no-op. A negative N, or a different N for a
        path that is already registered, raises ValueError (byte classification for
        already-recorded events must never silently change meaning).
        """
        if header_len < 0:
            raise ValueError(f"header_len must be >= 0, got {header_len} for {path}")
        with self._lock:
            prev = self._header_lens.get(path)
            if prev is not None and prev != header_len:
                raise ValueError(
                    f"header_len for {path} already registered as {prev}, got {header_len}"
                )
            self._header_lens[path] = header_len

    def header_len(self, path: str) -> int | None:
        with self._lock:
            return self._header_lens.get(path)

    # ------------------------------------------------------------------ budget

    def preflight(self, path: str, start: int, end_exclusive: int) -> int:
        c = classify_range(path, start, end_exclusive, self.header_len(path))
        if c["weight"] > 0:
            msg = f"weight bytes {c['weight']} in {path}[{start}:{end_exclusive}]"
            self.note("refused", msg, path=path)
            raise WeightReadRefused(msg)
        requested = end_exclusive - start
        with self._emit_lock:
            with self._lock:
                t = self._totals["meta"] + self._totals["header"] + sum(self._reservations.values())
                thr = self.threshold_bytes
                if t + requested <= thr:
                    rid = self._next_reservation
                    self._next_reservation += 1
                    self._reservations[rid] = requested
                    return rid
                msg = (
                    f"threshold: requested={requested} total={t} threshold={thr} "
                    f"disk=0 reason=non-weight read budget"
                )
                ev = self._zero_event_locked("refused", msg, path)
            self._emit(ev)
        raise ReadThresholdExceeded(requested, t, thr, msg)

    def release(self, reservation: int) -> None:
        with self._lock:
            self._reservations.pop(reservation, None)

    # ------------------------------------------------------------------ recording

    def record(
        self,
        *,
        event: str,
        source: str | None,
        path: str | None,
        url: str | None,
        start: int | None,
        nbytes: int,
        status: int | None,
        attempt: int = 1,
        elapsed_ms: float = 0.0,
        note: str | None = None,
        release: int | None = None,
    ) -> LogEvent:
        if event not in _RECORD_EVENTS:
            raise ValueError(f"record event must be one of {sorted(_RECORD_EVENTS)}, got {event!r}")
        if nbytes < 0:
            raise ValueError(f"nbytes must be >= 0, got {nbytes}")
        s = 0 if start is None else start
        with self._emit_lock:
            with self._lock:
                c = classify_range(path or "", s, s + nbytes, self._header_lens.get(path or ""))
                for k, v in c.items():
                    self._totals[k] += v
                if release is not None:
                    self._reservations.pop(release, None)
                ev = self._append_locked(
                    event=event, source=source, path=path, url=url,
                    range=(start, start + nbytes) if start is not None else None,
                    status=status, bytes=nbytes, bytes_by_class=c, attempt=attempt,
                    elapsed_ms=elapsed_ms, note=note,
                )
            return self._emit(ev)

    def note(self, event: str, note: str, path: str | None = None) -> LogEvent:
        if event not in _NOTE_EVENTS:
            raise ValueError(f"note event must be one of {sorted(_NOTE_EVENTS)}, got {event!r}")
        with self._emit_lock:
            with self._lock:
                ev = self._zero_event_locked(event, note, path)
            return self._emit(ev)

    # ------------------------------------------------------------------ export

    def to_card_dict(self) -> dict:
        with self._lock:
            return {
                "counting": COUNTING,
                "threshold_bytes": self.threshold_bytes,
                "totals": dict(self._totals),
                "events": [e.to_dict() for e in self._events],
            }
