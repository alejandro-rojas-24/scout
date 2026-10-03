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

from scout.bytelog import WEIGHT_EXTENSIONS, ByteLog
from scout.errors import HeaderError, ScoutError, WeightReadRefused

STAT_PATH: str = "@local/stat"


def _is_weight_name(name: str) -> bool:
    return name.lower().endswith(WEIGHT_EXTENSIONS)


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
        self._idents: dict[str, tuple[int, int]] = {}  # rel -> (st_dev, st_ino) at resolve()
        self._effective: dict[str, str] = {}  # rel -> classification name

    # ------------------------------------------------------------------ resolve

    def resolve(self) -> None:
        stats: dict[str, tuple[int, int]] = {}
        idents: dict[str, tuple[int, int]] = {}  # rel -> (st_dev, st_ino) of the target
        reals: dict[str, str] = {}  # rel -> realpath
        links: list[str] = []

        def _onerror(exc: OSError) -> None:
            raise ScoutError(f"cannot list {exc.filename}: {exc.strerror}") from exc

        for dirpath, dirnames, filenames in os.walk(self._root, onerror=_onerror, followlinks=False):
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
                idents[rel] = (st.st_dev, st.st_ino)
                reals[rel] = os.path.realpath(full)
                if os.path.islink(full):
                    links.append(rel)

        rels = sorted(stats)
        key = "\n".join(f"{rel}\t{stats[rel][0]}\t{stats[rel][1]}" for rel in rels)
        self._stats = stats
        self._idents = idents
        self._effective = self._classification_names(rels, idents, reals)
        self._files = [RepoFile(rel, stats[rel][0]) for rel in rels]
        self.revision_sha = hashlib.sha256(key.encode()).hexdigest()
        note = f"{len(rels)} files"
        if links:
            note += "; symlinks: " + "; ".join(f"{rel} -> {reals[rel]}" for rel in sorted(links))
        self.log.record(
            event="fetch", source="local", path=STAT_PATH, url=None, start=None,
            nbytes=0, status=None, note=note,
        )

    @staticmethod
    def _classification_names(
        rels: list[str], idents: dict[str, tuple[int, int]], reals: dict[str, str]
    ) -> dict[str, str]:
        """rel -> the name passed to preflight/record.

        ByteLog classifies by name, so a non-weight name that aliases weight bytes (a symlink
        to a weight file, or a hard link / second link to the same inode as a weight-named
        file) must be classified by its weight-class alias. If the file's own name is already
        weight-class it is used unchanged (safetensors header semantics apply). Otherwise, if
        any alias is weight-class, the effective name is f"{rel} -> {alias}", which ends with
        the alias's weight extension and is therefore refused by preflight; .safetensors
        aliases are refused outright in _effective_name (no header semantics on an alias).
        """
        by_inode: dict[tuple[int, int], list[str]] = {}
        for rel in rels:
            by_inode.setdefault(idents[rel], []).append(rel)
        out: dict[str, str] = {}
        for rel in rels:
            if _is_weight_name(rel):
                out[rel] = rel
                continue
            aliases = [os.path.basename(reals[rel])]
            aliases += [os.path.basename(o) for o in by_inode[idents[rel]] if o != rel]
            weighty = sorted({a for a in aliases if _is_weight_name(a)},
                             key=lambda a: (not a.lower().endswith(".safetensors"), a))
            out[rel] = f"{rel} -> {weighty[0]}" if weighty else rel
        return out

    def files(self) -> list[RepoFile]:
        if self._files is None:
            raise ScoutError("LocalSource.files() called before resolve()")
        return list(self._files)

    # ------------------------------------------------------------------ reads

    def _abs(self, path: str) -> str:
        return os.path.join(self._root, *path.split("/"))

    def _check_unchanged(self, f: typing.BinaryIO, path: str) -> None:
        st = os.fstat(f.fileno())
        if (st.st_size, st.st_mtime_ns) != self._stats[path] or (st.st_dev, st.st_ino) != self._idents[path]:
            raise ScoutError(f"{path} changed since resolve(); re-run resolve")

    def read_file(self, path: str) -> bytes | None:
        if path not in self._stats_or_raise():
            return None
        size = self._stats[path][0]
        name = self._effective_name(path)
        r = self.log.preflight(name, 0, size)
        try:
            t0 = time.perf_counter()
            with open(self._abs(path), "rb") as f:
                self._check_unchanged(f, path)
                data = f.read(size)  # never more than preflight approved
            self.log.record(
                event="fetch", source="local", path=name, url=None, start=0, nbytes=len(data),
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
        name = self._effective_name(path)
        r = self.log.preflight(name, start, start + length)
        try:
            t0 = time.perf_counter()
            with open(self._abs(path), "rb") as f:
                self._check_unchanged(f, path)
                f.seek(start)
                data = f.read(length)
            self.log.record(
                event="fetch", source="local", path=name, url=None, start=start, nbytes=len(data),
                status=None, elapsed_ms=(time.perf_counter() - t0) * 1000.0, release=r,
            )
        finally:
            self.log.release(r)
        if len(data) < length:
            raise HeaderError(f"short read {path}")
        return data

    def _effective_name(self, path: str) -> str:
        name = self._effective[path]
        if name != path and name.lower().endswith(".safetensors"):
            msg = f"{path} aliases safetensors weights ({name}); reads through an alias are refused"
            self.log.note("refused", msg, path=name)
            raise WeightReadRefused(msg)
        return name

    def _stats_or_raise(self) -> dict[str, tuple[int, int]]:
        if self._files is None:
            raise ScoutError("LocalSource read before resolve()")
        return self._stats
