"""Byte sources: the Source protocol and LocalSource (logged, preflighted local reads).

Every read goes through ByteLog.preflight (which refuses weight bytes and enforces the
non-weight budget) before the file is opened, and the bytes actually read are recorded.
Sources never write files.
"""

from __future__ import annotations

import hashlib
import os
import stat
import time
import typing
from dataclasses import dataclass
from pathlib import Path

from scout.bytelog import ByteLog
from scout.errors import HeaderError, ScoutError

STAT_PATH: str = "@local/stat"


@dataclass(frozen=True)
class RepoFile:
    path: str  # repo-relative, "/" separators
    size: int | None


class Source(typing.Protocol):
    kind: str  # "hub" | "local"
    repo: str  # "owner/name" or "local:/abs/path"
    revision_sha: str
    revision_kind: str  # "git" | "local-stat-hash"
    requested_revision: str | None
    local_path: str | None
    endpoint: str | None
    log: ByteLog

    def resolve(self) -> None:
        """Fill revision_sha and the file list; logs under the caller's RESOLVE stage."""
        ...

    def files(self) -> list[RepoFile]:
        """After resolve(); sorted by path."""
        ...

    def read_file(self, path: str) -> bytes | None:
        """Whole non-weight file; None if not in files()."""
        ...

    def read_range(self, path: str, start: int, length: int) -> bytes:
        ...


class LocalSource:
    """A local folder. The revision key is a hash of (path, size, mtime_ns) of every visible file."""

    def __init__(self, root: str | os.PathLike, log: ByteLog) -> None:
        r = Path(root).resolve(strict=True)
        if not r.is_dir():
            raise ScoutError(f"not a directory: {r}")
        self._root = r
        self.kind = "local"
        self.repo = f"local:{r}"
        self.local_path = str(r)
        self.requested_revision: str | None = None
        self.endpoint: str | None = None
        self.revision_kind = "local-stat-hash"
        self.revision_sha = ""
        self.log = log
        self._files: list[RepoFile] | None = None
        self._stats: dict[str, tuple[int, int]] = {}  # rel -> (size, mtime_ns) at resolve()

    # ------------------------------------------------------------------ resolve

    def resolve(self) -> None:
        stats: dict[str, tuple[int, int]] = {}
        for dirpath, dirnames, filenames in os.walk(self._root, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for name in sorted(filenames):
                if name.startswith("."):
                    continue
                full = os.path.join(dirpath, name)
                rel = Path(os.path.relpath(full, self._root)).as_posix()
                try:
                    st = os.stat(full)  # follows symlinks (HF cache snapshots -> ../blobs/)
                except OSError as exc:
                    if os.path.islink(full):
                        raise ScoutError(f"broken symlink {rel}") from exc
                    raise
                if not stat.S_ISREG(st.st_mode):
                    continue  # fifos, sockets, devices: never listed, never read
                stats[rel] = (st.st_size, st.st_mtime_ns)

        rels = sorted(stats)
        key = "\n".join(f"{rel}\t{stats[rel][0]}\t{stats[rel][1]}" for rel in rels)
        self._stats = stats
        self._files = [RepoFile(rel, stats[rel][0]) for rel in rels]
        self.revision_sha = hashlib.sha256(key.encode()).hexdigest()
        self.log.record(
            event="fetch", source="local", path=STAT_PATH, url=None, start=None,
            nbytes=0, status=None, note=f"{len(rels)} files",
        )

    def files(self) -> list[RepoFile]:
        if self._files is None:
            raise ScoutError("LocalSource.files() called before resolve()")
        return list(self._files)

    # ------------------------------------------------------------------ reads

    def _abs(self, path: str) -> str:
        return os.path.join(self._root, *path.split("/"))

    def _check_unchanged(self, f: typing.BinaryIO, path: str) -> None:
        st = os.fstat(f.fileno())
        if (st.st_size, st.st_mtime_ns) != self._stats[path]:
            raise ScoutError(f"{path} changed since resolve(); re-run resolve")

    def read_file(self, path: str) -> bytes | None:
        if path not in self._stats_or_raise():
            return None
        size = self._stats[path][0]
        r = self.log.preflight(path, 0, size)
        try:
            t0 = time.perf_counter()
            with open(self._abs(path), "rb") as f:
                self._check_unchanged(f, path)
                data = f.read(size)  # never more than preflight approved
            self.log.record(
                event="fetch", source="local", path=path, url=None, start=0, nbytes=len(data),
                status=None, elapsed_ms=(time.perf_counter() - t0) * 1000.0, release=r,
            )
        finally:
            self.log.release(r)
        if len(data) != size:
            raise ScoutError(f"short read {path}: {len(data)} of {size} bytes")
        return data

    def read_range(self, path: str, start: int, length: int) -> bytes:
        if path not in self._stats_or_raise():
            raise ScoutError(f"{path} is not in {self.repo}")
        r = self.log.preflight(path, start, start + length)
        try:
            t0 = time.perf_counter()
            with open(self._abs(path), "rb") as f:
                self._check_unchanged(f, path)
                f.seek(start)
                data = f.read(length)
            self.log.record(
                event="fetch", source="local", path=path, url=None, start=start, nbytes=len(data),
                status=None, elapsed_ms=(time.perf_counter() - t0) * 1000.0, release=r,
            )
        finally:
            self.log.release(r)
        if len(data) < length:
            raise HeaderError(f"short read {path}")
        return data

    def _stats_or_raise(self) -> dict[str, tuple[int, int]]:
        if self._files is None:
            raise ScoutError("LocalSource read before resolve()")
        return self._stats
