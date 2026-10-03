"""Tests for scout.bytelog and scout.errors (T002)."""

from __future__ import annotations

import json
import threading

import pytest

from scout import errors
from scout.bytelog import (
    COUNTING,
    DEFAULT_THRESHOLD_BYTES,
    META_MAX_BYTES,
    WEIGHT_EXTENSIONS,
    ByteClass,
    ByteLog,
    LogEvent,
    Stage,
    classify_range,
)
from scout.errors import (
    ReadThresholdExceeded,
    ScoutError,
    StageOrderError,
    WeightReadRefused,
)


def c(meta=0, header=0, weight=0):
    return {"meta": meta, "header": header, "weight": weight}


# --------------------------------------------------------------------------- errors


def test_error_hierarchy_and_exit_codes():
    expected = {
        "ScoutError": 1,
        "WeightReadRefused": 5,
        "ReadThresholdExceeded": 5,
        "StageOrderError": 1,
        "GatedRepoError": 3,
        "RepoNotFoundError": 3,
        "RevisionMismatch": 1,
        "NetworkError": 4,
        "RangeNotSupported": 4,
        "HubAPIError": 4,
        "HeaderError": 1,
        "NoSafetensorsError": 1,
        "AmbiguousWeightsError": 1,
        "DuplicateTensorError": 1,
    }
    for name, code in expected.items():
        cls = getattr(errors, name)
        assert issubclass(cls, ScoutError)
        assert issubclass(cls, Exception)
        assert cls.exit_code == code, name


def test_read_threshold_exceeded_attrs():
    e = ReadThresholdExceeded(11, 90, 100)
    assert (e.requested, e.total, e.threshold) == (11, 90, 100)
    assert e.exit_code == 5
    assert "requested=11" in str(e)
    e2 = ReadThresholdExceeded(1, 2, 3, "custom")
    assert str(e2) == "custom"


# --------------------------------------------------------------------------- constants


def test_constants():
    assert DEFAULT_THRESHOLD_BYTES == 64 * 1024 * 1024
    assert META_MAX_BYTES == 16 * 1024 * 1024
    assert COUNTING == "http-body-bytes-yielded"
    assert ".safetensors" in WEIGHT_EXTENSIONS and ".pkl" in WEIGHT_EXTENSIONS
    assert len(WEIGHT_EXTENSIONS) == 12
    assert [s.value for s in Stage] == list(range(1, 10))
    assert Stage.RESOLVE < Stage.PURGE
    assert ByteClass.META == "meta" and ByteClass.HEADER.value == "header"
    assert ByteClass.WEIGHT.value == "weight"


# --------------------------------------------------------------------------- classify_range


@pytest.mark.parametrize(
    "path,start,end,n,expected",
    [
        ("model.safetensors", 0, 8, None, c(header=8)),
        ("model.safetensors", 0, 16, None, c(header=8, weight=8)),
        ("model.safetensors", 8, 108, 100, c(header=100)),
        ("model.safetensors", 100, 120, 100, c(header=8, weight=12)),
        ("model.safetensors", 0, 0, None, c()),
        ("model.safetensors", 4, 6, None, c(header=2)),
        ("model.safetensors", 200, 300, 100, c(weight=100)),
        ("model.bin", 0, 4, None, c(weight=4)),
        ("model.bin", 0, 4, 100, c(weight=4)),
        ("x/y/model.GGUF", 10, 20, None, c(weight=10)),
        ("config.json", 0, 50, None, c(meta=50)),
        ("model.safetensors.index.json", 0, 50, None, c(meta=50)),
        ("@api/revision", 0, 7, None, c(meta=7)),
        ("@error", 0, 7, None, c(meta=7)),
        ("README.md", 0, 3, None, c(meta=3)),
        ("MODEL.SAFETENSORS", 0, 16, None, c(header=8, weight=8)),
        ("Model.SafeTensors", 8, 108, 100, c(header=100)),
    ],
)
def test_classify_range_table(path, start, end, n, expected):
    got = classify_range(path, start, end, n)
    assert got == expected
    assert sum(got.values()) == end - start


@pytest.mark.parametrize("start,end", [(-1, 4), (5, 4)])
def test_classify_range_invalid(start, end):
    with pytest.raises(ValueError):
        classify_range("config.json", start, end, None)


# --------------------------------------------------------------------------- preflight


def test_preflight_weight_refused_before_header_len():
    events = []
    log = ByteLog(on_event=events.append)
    before = log.totals
    with pytest.raises(WeightReadRefused) as ei:
        log.preflight("a.safetensors", 8, 9)
    assert "weight bytes 1 in a.safetensors[8:9]" in str(ei.value)
    assert log.totals == before
    assert log.reserved == 0
    refused = [e for e in log.events if e.event == "refused"]
    assert len(refused) == 1
    assert refused[0].note == "weight bytes 1 in a.safetensors[8:9]"
    assert refused[0].bytes == 0
    assert refused[0].path == "a.safetensors"
    assert events[-1] is refused[0] or events[-1] == refused[0]

    log.set_header_len("a.safetensors", 1)
    assert log.header_len("a.safetensors") == 1
    rid = log.preflight("a.safetensors", 8, 9)
    assert log.reserved == 1
    log.release(rid)
    assert log.reserved == 0
    assert log.totals == before


def test_header_len_unknown_is_none():
    assert ByteLog().header_len("nope.safetensors") is None


def test_threshold():
    log = ByteLog(threshold_bytes=100)
    log.record(event="fetch", source="hub", path="config.json", url=None,
               start=0, nbytes=90, status=200)
    with pytest.raises(ReadThresholdExceeded) as ei:
        log.preflight("x.json", 0, 11)
    e = ei.value
    assert (e.requested, e.total, e.threshold) == (11, 90, 100)
    assert str(e) == (
        "threshold: requested=11 total=90 threshold=100 disk=0 reason=non-weight read budget"
    )
    assert log.events[-1].event == "refused"
    assert log.events[-1].note == str(e)
    assert log.reserved == 0
    rid = log.preflight("x.json", 0, 10)
    assert log.reserved == 10
    log.release(rid)


def test_threshold_counts_header_bytes_not_weight():
    log = ByteLog(threshold_bytes=100)
    # 16 bytes: 8 header + 8 weight (N unknown); weight does not count to budget
    log.record(event="fetch", source=None, path="m.safetensors", url=None,
               start=0, nbytes=16, status=206)
    assert log.totals == c(header=8, weight=8)
    log.preflight("x.json", 0, 92)
    with pytest.raises(ReadThresholdExceeded):
        log.preflight("x.json", 0, 1)


def test_weight_check_precedes_threshold():
    log = ByteLog(threshold_bytes=1)
    with pytest.raises(WeightReadRefused):
        log.preflight("m.bin", 0, 1000)
    assert log.reserved == 0


def test_reservation_race():
    log = ByteLog(threshold_bytes=400)
    barrier = threading.Barrier(8)
    ok: list[int] = []
    bad: list[Exception] = []
    lock = threading.Lock()

    def worker():
        barrier.wait()
        try:
            rid = log.preflight("c.json", 0, 100)
            with lock:
                ok.append(rid)
        except ReadThresholdExceeded as e:
            with lock:
                bad.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ok) == 4
    assert len(bad) == 4
    assert len(set(ok)) == 4
    assert log.reserved == 400
    for rid in ok:
        log.release(rid)
    assert log.reserved == 0
    assert log.totals == c()


def test_record_release_frees_reservation_and_release_idempotent():
    log = ByteLog(threshold_bytes=100)
    rid = log.preflight("config.json", 0, 60)
    assert log.reserved == 60
    log.record(event="fetch", source="hub", path="config.json", url="u",
               start=0, nbytes=50, status=200, release=rid)
    assert log.reserved == 0
    assert log.totals["meta"] == 50
    log.release(rid)
    log.release(rid)
    log.release(99999)
    assert log.reserved == 0
    # budget now 50 used
    rid2 = log.preflight("config.json", 0, 50)
    log.release(rid2)
    with pytest.raises(ReadThresholdExceeded):
        log.preflight("config.json", 0, 51)


def test_preflight_never_records_bytes():
    log = ByteLog()
    rid = log.preflight("config.json", 0, 1000)
    assert log.totals == c()
    assert log.events == []
    log.release(rid)


# --------------------------------------------------------------------------- stages


def test_stage_order():
    log = ByteLog()
    assert log.current_stage is None
    log.stage(Stage.HEADERS)
    with pytest.raises(StageOrderError):
        log.stage(Stage.RESOLVE)
    assert log.current_stage is Stage.HEADERS

    log2 = ByteLog()
    for s in (Stage.RESOLVE, Stage.HEADERS, Stage.META, Stage.REPORT):
        ev = log2.stage(s)
        assert ev.event == "stage_start"
        assert ev.stage == s.name
        assert ev.bytes == 0
    assert log2.current_stage is Stage.REPORT
    ev = log2.stage(Stage.REPORT)
    assert ev.event == "stage_start"
    assert len(log2.events) == 5


def test_record_stage_name():
    log = ByteLog()
    ev = log.record(event="fetch", source=None, path="config.json", url=None,
                    start=None, nbytes=3, status=200)
    assert ev.stage == "NONE"
    assert ev.range is None
    log.stage(Stage.META)
    ev = log.record(event="fetch", source=None, path="config.json", url=None,
                    start=5, nbytes=3, status=206)
    assert ev.stage == "META"
    assert ev.range == (5, 8)


# --------------------------------------------------------------------------- record


def test_thread_safety():
    log = ByteLog()
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        for _ in range(1000):
            log.record(event="fetch", source=None, path="config.json", url=None,
                       start=0, nbytes=1, status=200)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert log.totals["meta"] == 8000
    seqs = sorted(e.seq for e in log.events)
    assert seqs == list(range(1, 8001))
    # events list is in seq order
    assert [e.seq for e in log.events] == seqs


def test_record_error_and_bogus_event():
    log = ByteLog()
    ev = log.record(event="error", source="hub", path="@error", url="u",
                    start=None, nbytes=10, status=500)
    assert log.totals["meta"] == 10
    assert ev.bytes_by_class == c(meta=10)
    with pytest.raises(ValueError):
        log.record(event="bogus", source=None, path="x", url=None,
                   start=0, nbytes=1, status=200)
    assert log.totals["meta"] == 10
    assert len(log.events) == 1
    log.record(event="retry", source=None, path="x.json", url=None,
               start=0, nbytes=0, status=503, attempt=2)


def test_record_mixed_range_counts_weight():
    log = ByteLog()
    ev = log.record(event="fetch", source=None, path="m.safetensors", url=None,
                    start=0, nbytes=200, status=206)
    assert log.totals["weight"] > 0
    assert ev.bytes_by_class == c(header=8, weight=192)
    log.set_header_len("n.safetensors", 100)
    ev = log.record(event="fetch", source=None, path="n.safetensors", url=None,
                    start=0, nbytes=200, status=206)
    assert ev.bytes_by_class == c(header=108, weight=92)
    assert log.totals == c(header=116, weight=284)
    assert ev.totals == log.totals


def test_event_fields_and_snapshots():
    log = ByteLog()
    ev = log.record(event="fetch", source="hub", path="config.json", url="https://x",
                    start=0, nbytes=5, status=200, attempt=3, elapsed_ms=1.5, note="n")
    assert isinstance(ev, LogEvent)
    assert ev.seq == 1 and ev.source == "hub" and ev.url == "https://x"
    assert ev.status == 200 and ev.attempt == 3 and ev.elapsed_ms == 1.5
    assert ev.note == "n" and ev.ts > 0
    # copies
    t = log.totals
    t["meta"] = 999
    assert log.totals["meta"] == 5
    evs = log.events
    evs.clear()
    assert len(log.events) == 1
    log.record(event="fetch", source=None, path="config.json", url=None,
               start=0, nbytes=1, status=200)
    assert ev.totals["meta"] == 5  # snapshot not mutated
    assert [e.seq for e in log.events_since(1)] == [2]
    assert [e.seq for e in log.events_since(0)] == [1, 2]
    assert log.events_since(2) == []


# --------------------------------------------------------------------------- note


def test_note_events():
    log = ByteLog()
    for name in ("error", "refused", "card_written"):
        ev = log.note(name, "hello", path="p")
        assert ev.event == name and ev.bytes == 0 and ev.note == "hello"
        assert ev.path == "p"
        assert ev.bytes_by_class == c()
    with pytest.raises(ValueError):
        log.note("fetch", "x")
    assert log.totals == c()


# --------------------------------------------------------------------------- on_event / card


def test_on_event_order_and_card_roundtrip():
    seen: list[LogEvent] = []
    log = ByteLog(threshold_bytes=1000, on_event=seen.append)
    log.stage(Stage.RESOLVE)
    log.record(event="fetch", source="hub", path="@api/revision", url="u",
               start=None, nbytes=7, status=200)
    log.stage(Stage.HEADERS)
    log.record(event="fetch", source="hub", path="m.safetensors", url="u",
               start=0, nbytes=8, status=206)
    with pytest.raises(WeightReadRefused):
        log.preflight("m.safetensors", 8, 100)
    log.note("card_written", "card.json")
    assert [e.seq for e in seen] == [e.seq for e in log.events] == list(range(1, 7))

    card = log.to_card_dict()
    s = json.dumps(card)
    back = json.loads(s)
    assert back == card
    assert card["counting"] == COUNTING
    assert card["threshold_bytes"] == 1000
    assert card["totals"] == c(meta=7, header=8)
    assert len(card["events"]) == 6
    assert card["events"][3]["range"] == [0, 8]
    assert card["events"][1]["range"] is None


def test_on_event_exception_propagates():
    def boom(ev):
        raise RuntimeError("boom")

    log = ByteLog(on_event=boom)
    with pytest.raises(RuntimeError):
        log.record(event="fetch", source=None, path="x.json", url=None,
                   start=0, nbytes=1, status=200)
    # the event was still recorded before the callback raised
    assert log.totals["meta"] == 1


# --------------------------------------------------------------------------- emit ordering


def _run_with_timeout(fn, timeout=10.0):
    """Run fn in a daemon thread; fail (instead of hanging) on deadlock."""
    err: list[BaseException] = []

    def target():
        try:
            fn()
        except BaseException as e:  # noqa: BLE001
            err.append(e)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), "deadlock: call did not finish"
    if err:
        raise err[0]


def test_on_event_strict_seq_order_under_concurrency():
    seen: list[int] = []

    def cb(ev):
        # no extra lock: the ByteLog must serialize callbacks itself
        seen.append(ev.seq)

    log = ByteLog(on_event=cb)
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        for _ in range(200):
            log.record(event="fetch", source=None, path="config.json", url=None,
                       start=0, nbytes=1, status=200)

    def main():
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    _run_with_timeout(main, timeout=30.0)
    assert seen == list(range(1, 1601))
    assert log.totals["meta"] == 1600


def test_on_event_strict_seq_order_mixed_event_kinds():
    seen: list[int] = []
    log = ByteLog(threshold_bytes=50, on_event=lambda ev: seen.append(ev.seq))
    barrier = threading.Barrier(8)

    def worker(i):
        barrier.wait()
        for _ in range(50):
            if i % 4 == 0:
                log.record(event="fetch", source=None, path="m.safetensors", url=None,
                           start=0, nbytes=4, status=206)
            elif i % 4 == 1:
                log.note("error", "x")
            elif i % 4 == 2:
                try:
                    log.preflight("c.json", 0, 60)  # always over budget -> refused event
                except ReadThresholdExceeded:
                    pass
            else:
                try:
                    log.preflight("w.bin", 0, 1)  # weight -> refused event
                except WeightReadRefused:
                    pass

    def main():
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    _run_with_timeout(main, timeout=30.0)
    assert seen == list(range(1, len(log.events) + 1))
    assert len(seen) == 400


def test_reentrant_record_from_on_event_does_not_deadlock():
    seen: list[tuple[int, str]] = []
    log_ref: list[ByteLog] = []

    def cb(ev):
        seen.append((ev.seq, ev.event))
        log = log_ref[0]
        # read views and re-enter the log from inside the callback
        _ = log.totals, log.reserved, log.events, log.current_stage
        if ev.event == "fetch" and ev.path == "config.json":
            log.record(event="retry", source=None, path="nested.json", url=None,
                       start=0, nbytes=2, status=503)
            log.note("error", "nested")
            rid = log.preflight("x.json", 0, 1)
            log.release(rid)

    log = ByteLog(on_event=cb)
    log_ref.append(log)

    def main():
        log.stage(Stage.RESOLVE)
        log.record(event="fetch", source=None, path="config.json", url=None,
                   start=0, nbytes=3, status=200)

    _run_with_timeout(main)
    # callbacks start in seq order even when nested
    assert [s for s, _ in seen] == [1, 2, 3, 4]
    assert [e for _, e in seen] == ["stage_start", "fetch", "retry", "error"]
    assert log.totals["meta"] == 5


def test_reentrant_record_from_on_event_concurrent():
    seen: list[int] = []
    log_ref: list[ByteLog] = []

    def cb(ev):
        seen.append(ev.seq)
        if ev.event == "fetch":
            log_ref[0].note("error", "nested")

    log = ByteLog(on_event=cb)
    log_ref.append(log)
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        for _ in range(100):
            log.record(event="fetch", source=None, path="config.json", url=None,
                       start=0, nbytes=1, status=200)

    def main():
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    _run_with_timeout(main, timeout=30.0)
    assert seen == list(range(1, 1601))
    # each fetch is immediately followed by its nested note
    kinds = [e.event for e in log.events]
    assert kinds == ["fetch", "error"] * 800
