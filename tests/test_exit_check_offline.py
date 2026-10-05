"""Offline rehearsal of scripts/exit_check.py against FakeHub (no network).

Uses the real server (make_server), the real CountingTransport around FakeHub's MockTransport,
and the real T016 common checks, with endpoint=HUB and allowed_hosts={"hub.test", "cdn.test"}.
"""

from __future__ import annotations

import json
import pathlib

import httpx
import pytest

import scout.hub
from scripts import exit_check as ec
from scripts.exit_expectations import DEFAULT_EXPECTATIONS, HF_ENDPOINT_URL, Check, common_checks
from tests.helpers.fakehub import CDN, HUB, FakeHub
from tests.helpers.st_fixtures import dense_tensors, write_dense_repo, write_sharded

SHA_A = "a" * 40
SHA_B = "b" * 40
OFFLINE_HOSTS = {"hub.test", "cdn.test"}
REQUIRED = sorted(DEFAULT_EXPECTATIONS)


@pytest.fixture(autouse=True)
def _no_hf_endpoint(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)


def _dense_hub(tmp_path: pathlib.Path, meta_redirect: str | None = "relative307") -> tuple[FakeHub, dict]:
    hub = FakeHub()
    info = write_dense_repo(tmp_path / "dense", n_shards=2)
    hub.add_repo("org/dense", tmp_path / "dense", sha=SHA_A, meta_redirect=meta_redirect)
    return hub, info


def _run(hub: FakeHub, tmp_path: pathlib.Path, pins: dict, expectations: dict, **kw) -> list[Check]:
    kw.setdefault("allowed_hosts", OFFLINE_HOSTS)
    return ec.run(pins, tmp_path / "out", endpoint=HUB, inner_transport_factory=hub.transport,
                  expectations=expectations, **kw)


def _spy_scan_target(monkeypatch) -> list:
    results = []
    real = ec._scan_target

    def spy(*a, **kw):
        r = real(*a, **kw)
        results.append(r)
        return r
    monkeypatch.setattr(ec, "_scan_target", spy)
    return results


def _named(checks: list[Check], prefix: str) -> list[Check]:
    return [c for c in checks if c.name.startswith(prefix)]


def _write_pins(tmp_path: pathlib.Path, pins: dict) -> str:
    p = tmp_path / "pins.json"
    p.write_text(json.dumps(pins))
    return str(p)


def _no_server(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("make_server must not be called")
    monkeypatch.setattr(ec, "make_server", boom)


# ---------------------------------------------------------------------------
# the offline rehearsal


def test_offline_pass(tmp_path, monkeypatch):
    hub, info = _dense_hub(tmp_path)
    results = _spy_scan_target(monkeypatch)
    checks = _run(hub, tmp_path, {"org/dense": SHA_A}, {"org/dense": lambda r: []})

    failed = [c for c in checks if not c.ok]
    assert not failed, "\n".join(f"{c.name}: expected {c.expected} actual {c.actual}" for c in failed)
    assert checks[0].name == "C0 target set" and checks[0].target == "*"
    names = {c.name.split()[0] for c in checks}
    assert {f"C{i}" for i in range(13)} <= names

    (r,) = results
    assert r.status == "done" and r.target == f"org/dense@{SHA_A}" and r.pin == SHA_A
    wire = r.wire
    prefix = f"/org/dense/resolve/{SHA_A}/"
    # C10 saw >= 2 safetensors records mapped through the absolute CDN Location
    cdn = [w for w in wire if w["url"].startswith(CDN + "/")]
    assert len(cdn) >= 2
    for w in cdn:
        assert w["orig_path"].startswith(prefix) and w["orig_path"].endswith(".safetensors")
        assert w["range"] is not None and w["status"] == 206
    assert {w["orig_path"] for w in cdn} == {prefix + f for f in info["files"]}
    # the redirect hops record the resolved absolute Location and the full resolve path
    hops302 = [w for w in wire if w["status"] == 302]
    assert {w["location"] for w in hops302} == {w["url"] for w in cdn}
    assert all(w["orig_path"] == pathlib.PurePosixPath(httpx.URL(w["url"]).path).as_posix() for w in hops302)
    # C11 includes the 18-byte relative-307 bodies, mapped back via the resolved relative Location
    r307 = [w for w in wire if w["status"] == 307]
    assert r307 and all(w["body_bytes"] == len(b"Temporary Redirect") == 18 for w in r307)
    for w in r307:
        assert w["location"].startswith(f"{HUB}/api/resolve-cache/models/org/dense/{SHA_A}/")
        follow = [x for x in wire if x["url"] == w["location"]]
        assert follow and all(x["orig_path"] == w["orig_path"] for x in follow)
    # the API call is the only record without orig_path
    api = [w for w in wire if w["orig_path"] is None]
    assert len(api) == 1 and httpx.URL(api[0]["url"]).path == f"/api/models/org/dense/revision/{SHA_A}"
    assert sum(w["body_bytes"] for w in wire) == sum(r.final["totals"][k] for k in ("meta", "header", "weight"))


def test_c10_passes_on_real_counting_records(tmp_path, monkeypatch):
    """Real CountingTransport records (full /{owner}/{name}/resolve/{sha}/{file} orig_path) through T016 C10."""
    hub, _ = _dense_hub(tmp_path)
    results = _spy_scan_target(monkeypatch)
    _run(hub, tmp_path, {"org/dense": SHA_A}, {"org/dense": lambda r: []})
    (r,) = results
    for w in r.wire:
        if w["orig_path"] is not None:
            assert w["orig_path"].startswith(f"/org/dense/resolve/{SHA_A}/")
    c10 = _named(common_checks(r, ec.BUDGET_S, allowed_hosts=OFFLINE_HOSTS, expected_endpoint=HUB), "C10")
    assert c10 and all(c.ok for c in c10), [c for c in c10 if not c.ok]
    assert any("is a Card weight file with Range within header" in c.name for c in c10)
    # and a bare file-name orig_path (the shape T016 rejects) would fail C10
    bad = [dict(w, orig_path=w["orig_path"].rsplit("/", 1)[1]) if w["orig_path"] else w for w in r.wire]
    r.wire = bad
    c10_bad = _named(common_checks(r, ec.BUDGET_S, allowed_hosts=OFFLINE_HOSTS, expected_endpoint=HUB), "C10")
    assert not all(c.ok for c in c10_bad)


def test_c12_disallowed_host(tmp_path):
    hub, _ = _dense_hub(tmp_path)
    checks = _run(hub, tmp_path, {"org/dense": SHA_A}, {"org/dense": lambda r: []}, allowed_hosts={"hub.test"})
    c12 = [c for c in checks if c.name == "C12 every wire host allowed"]
    assert len(c12) == 1 and not c12[0].ok and "cdn.test" in c12[0].actual


def test_unlogged_read_fails_c11(tmp_path, monkeypatch):
    hub, _ = _dense_hub(tmp_path)
    real = scout.hub.HubSource.read_file

    def leaky(self, path):
        if path == "config.json":
            url = f"{self.endpoint}/{self.repo}/resolve/{self.revision_sha}/config.json"
            resp = self._client.get(url, follow_redirects=True)
            assert resp.content
        return real(self, path)
    monkeypatch.setattr(scout.hub.HubSource, "read_file", leaky)
    checks = _run(hub, tmp_path, {"org/dense": SHA_A}, {"org/dense": lambda r: []})
    c11 = [c for c in checks if c.name.startswith("C11")]
    assert len(c11) == 1 and not c11[0].ok
    # everything else still ran: the scan finished and C1 passed
    assert [c.ok for c in checks if c.name.startswith("C1 ")] == [True]
    assert len(checks) > 20


def test_range_ignored_fails(tmp_path):
    hub, info = _dense_hub(tmp_path)
    hub.inject(info["files"][0], "ignore_range")
    checks = _run(hub, tmp_path, {"org/dense": SHA_A}, {"org/dense": lambda r: []})
    c1 = [c for c in checks if c.name.startswith("C1 ")]
    assert len(c1) == 1 and not c1[0].ok and "error" in c1[0].actual
    assert not all(c.ok for c in checks)


def test_transport_per_target(tmp_path, monkeypatch):
    hub = FakeHub()
    write_dense_repo(tmp_path / "a", n_shards=2)
    root_b = tmp_path / "b"
    root_b.mkdir()
    (root_b / "config.json").write_text((tmp_path / "a" / "config.json").read_text())
    files_b = write_sharded(root_b, dense_tensors(), 2, stem="weights")["files"]
    hub.add_repo("org/a", tmp_path / "a", sha=SHA_A)
    hub.add_repo("org/b", root_b, sha=SHA_B)

    events: list = []
    factories: list = []
    real_make = ec.make_server

    def spy_make(*a, **kw):
        events.append(("make_server", len(factories)))
        factories.append(kw["client_factory"])
        server = real_make(*a, **kw)
        idx = len(factories) - 1
        real_shutdown = server.shutdown

        def shutdown():
            events.append(("shutdown", idx))
            real_shutdown()
        server.shutdown = shutdown
        return server
    monkeypatch.setattr(ec, "make_server", spy_make)
    results = _spy_scan_target(monkeypatch)

    checks = _run(hub, tmp_path, {"org/a": SHA_A, "org/b": SHA_B},
                  {"org/a": lambda r: [], "org/b": lambda r: []})
    assert all(c.ok for c in checks), [c for c in checks if not c.ok]

    assert len(factories) == 2 and factories[0] is not factories[1]
    counting = [f.args[0] for f in factories]
    assert all(isinstance(c, ec.CountingTransport) for c in counting)
    assert counting[0] is not counting[1]
    assert events.index(("shutdown", 0)) < events.index(("make_server", 1))

    ra, rb = results
    assert ra.target == f"org/a@{SHA_A}" and rb.target == f"org/b@{SHA_B}"

    def mentions(w: dict, repo: str, sha: str, files: list[str]) -> bool:
        orig = w["orig_path"] or ""
        return (f"/{repo}/" in w["url"] or f"/{repo}/" in orig or sha in w["url"]
                or any(orig.endswith("/" + f) for f in files))
    files_a = [p.name for p in (tmp_path / "a").glob("*.safetensors")]
    assert ra.wire and rb.wire
    assert not [w for w in rb.wire if mentions(w, "org/a", SHA_A, files_a)]
    assert not [w for w in ra.wire if mentions(w, "org/b", SHA_B, files_b)]


def test_run_c0(tmp_path, monkeypatch):
    _no_server(monkeypatch)
    checks = ec.run({"org/a": SHA_A}, tmp_path / "out", endpoint=HUB, allowed_hosts=OFFLINE_HOSTS,
                    inner_transport_factory=FakeHub().transport, expectations={"org/b": lambda r: []})
    assert len(checks) == 1
    (c0,) = checks
    assert c0.name == "C0 target set" and c0.target == "*" and not c0.ok
    assert c0.expected == str(["org/b"]) and c0.actual == str(["org/a"])


# ---------------------------------------------------------------------------
# main(): target set, endpoint, flags, pins, summary


def _valid_pins(extra: dict | None = None, drop: str | None = None) -> dict:
    pins = {r: SHA_A for r in REQUIRED if r != drop}
    pins.update(extra or {})
    return pins


def test_target_set_missing(tmp_path, monkeypatch, capsys):
    _no_server(monkeypatch)
    rc = ec.main(["--pins", _write_pins(tmp_path, _valid_pins(drop="Qwen/Qwen-Image")), "--out", str(tmp_path)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "C0 target set" in out and "EXIT CHECK: FAIL (target set must be exactly:" in out


def test_target_set_extra(tmp_path, monkeypatch):
    _no_server(monkeypatch)
    pins = _valid_pins(extra={"stabilityai/stable-diffusion-xl-base-1.0": SHA_A})
    assert ec.main(["--pins", _write_pins(tmp_path, pins)]) == 1


def test_target_set_renamed(tmp_path, monkeypatch):
    _no_server(monkeypatch)
    pins = _valid_pins(drop="Qwen/Qwen3-8B", extra={"Qwen/Qwen3-8b": SHA_A})
    assert ec.main(["--pins", _write_pins(tmp_path, pins)]) == 1


def test_endpoint_env_refused(tmp_path, monkeypatch, capsys):
    _no_server(monkeypatch)
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example.com")
    assert ec.main(["--pins", _write_pins(tmp_path, _valid_pins())]) == 1
    assert "HF_ENDPOINT=https://mirror.example.com is not allowed for the exit check" in capsys.readouterr().out


def test_no_budget_flag(capsys):
    with pytest.raises(SystemExit) as e:
        ec.main(["--budget-s", "60"])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        ec.main(["--endpoint", HUB])
    assert e.value.code == 2


def test_main_unpinned(tmp_path, monkeypatch, capsys):
    _no_server(monkeypatch)
    pins = {r: "UNPINNED" for r in REQUIRED}
    assert ec.main(["--pins", _write_pins(tmp_path, pins)]) == 2
    assert "pins not set: run `scout resolve <repo>` and commit exit/pins.json" in capsys.readouterr().out


def test_committed_pins_file_is_unpinned():
    data = json.loads((ec.ROOT / "exit" / "pins.json").read_text())
    assert data == {r: "UNPINNED" for r in REQUIRED}


def test_summary_line(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_run(pins, out_dir, **kw):
        seen.update(kw, pins=pins)
        ec.LAST_TARGET_RESULTS[:] = []
        return [Check("*", "C0 target set", "x", "x", True), Check("t", "C1 x", "a", "a", True)]
    monkeypatch.setattr(ec, "run", fake_run)
    monkeypatch.setattr(ec, "_git_info", lambda: ("deadbeef", False))
    monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.co/")   # same endpoint, trailing slash ok
    pins = _valid_pins()
    assert ec.main(["--pins", _write_pins(tmp_path, pins), "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert "EXIT CHECK: PASS" in out
    line = json.loads(out[-1])
    assert line["step"] == "EXIT" and line["hostname"] and line["pins"] == pins
    assert line["endpoint"] == HF_ENDPOINT_URL and line["budget_s"] == 10.0
    assert line["hosts"] == [] and line["git_commit"] == "deadbeef" and line["git_dirty"] is False
    assert line["passed"] == 2 and line["failed"] == 0
    assert seen["endpoint"] == HF_ENDPOINT_URL and seen["budget_s"] == 10.0
    assert tuple(seen["allowed_hosts"]) == ec.ALLOWED_HOSTS


def test_summary_fail_and_hosts(tmp_path, monkeypatch, capsys):
    hub, _ = _dense_hub(tmp_path)
    real_run = ec.run

    def offline_run(pins, out_dir, **kw):
        # main always passes the fixed values; route the scan to FakeHub to exercise the summary
        assert kw["endpoint"] == HF_ENDPOINT_URL and kw["budget_s"] == ec.BUDGET_S
        return real_run({"org/dense": SHA_A}, out_dir, endpoint=HUB, allowed_hosts=OFFLINE_HOSTS,
                        inner_transport_factory=hub.transport, expectations={"org/dense": lambda r: []})
    monkeypatch.setattr(ec, "run", offline_run)
    monkeypatch.setattr(ec, "_git_info", lambda: ("unknown", None))
    assert ec.main(["--pins", _write_pins(tmp_path, _valid_pins()), "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert any(l.startswith("* | C0 target set |") and l.endswith("| PASS") for l in out)
    line = json.loads(out[-1])
    assert line["hosts"] == ["cdn.test", "hub.test"] and line["git_dirty"] is None


def test_git_info_failure_is_unknown(monkeypatch):
    def boom(*a, **kw):
        raise FileNotFoundError("git")
    monkeypatch.setattr(ec.subprocess, "run", boom)
    assert ec._git_info() == ("unknown", None)


def test_git_info_values(monkeypatch):
    import subprocess

    def fake(cmd, **kw):
        assert kw["cwd"] == str(ec.ROOT)
        out = "abc123\n" if cmd[1] == "rev-parse" else " M x.py\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
    monkeypatch.setattr(ec.subprocess, "run", fake)
    assert ec._git_info() == ("abc123", True)


def test_counting_transport_counts_consumed_bytes_only():
    def handler(request):
        return httpx.Response(200, content=b"x" * 100)
    ct = ec.CountingTransport(httpx.MockTransport(handler))
    with httpx.Client(transport=ct) as c:
        with c.stream("GET", "https://hub.test/o/n/resolve/" + SHA_A + "/a%20b.json",
                      headers={"Range": "bytes=0-9"}) as r:
            pass  # body never consumed
        c.get("https://hub.test/other")
    first, second = ct.records
    assert first["body_bytes"] == 0 and first["orig_path"] == f"/o/n/resolve/{SHA_A}/a b.json"
    assert first["range"] == "bytes=0-9" and first["status"] == 200 and first["location"] is None
    assert second["body_bytes"] == 100 and second["orig_path"] is None and second["method"] == "GET"


def test_counting_transport_is_not_an_event_hook(tmp_path):
    client = ec._client_for(ec.CountingTransport(FakeHub().transport()))
    assert not client.event_hooks.get("response")
    assert client.timeout.read == 10.0
    scout.hub.HubSource("o/n", SHA_A, scout.hub.ByteLog(1 << 20), client=client, endpoint=HUB)  # no ValueError
