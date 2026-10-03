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
MAX_SYMLINK_HOPS: int = 40  # same as Linux MAXSYMLINKS
MAX_NOTE_LINKS: int = 20  # symlink entries listed in the persisted @local/stat note


def _is_weight_name(name: str) -> bool:
    return name.lower().endswith(WEIGHT_EXTENSIONS)


def _weight_ext(name: str) -> str | None:
    low = name.lower()
    for ext in WEIGHT_EXTENSIONS:
        if low.endswith(ext):
            return ext
    return None


def _link_chain(full: str, rel: str) -> list[str]:
    """Basenames of every hop target of the symlink chain starting at `full` (empty if not a link).

    Follows os.readlink hop by hop (relative targets are joined to the link's directory and left
    unnormalised so the OS resolves `..` physically). More than MAX_SYMLINK_HOPS hops, or a cycle,
    raises ScoutError("symlink chain too long ..."). A readlink failure raises OSError.
    """
    hops: list[str] = []
    cur = full
    while os.path.islink(cur):
        if len(hops) >= MAX_SYMLINK_HOPS:
            raise ScoutError(f"symlink chain too long {rel} (more than {MAX_SYMLINK_HOPS} hops)")
        target = os.readlink(cur)
        cur = os.path.join(os.path.dirname(cur), target)
        hops.append(os.path.basename(target.rstrip("/\\")))
    return hops


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
        self._nlinks: dict[str, int] = {}  # rel -> st_nlink at resolve()
        self._effective: dict[str, str] = {}  # rel -> classification name

    # ------------------------------------------------------------------ resolve

    def resolve(self) -> None:
        stats: dict[str, tuple[int, int]] = {}
        idents: dict[str, tuple[int, int]] = {}  # rel -> (st_dev, st_ino) of the target
        nlinks: dict[str, int] = {}  # rel -> st_nlink of the target
        reals: dict[str, str] = {}  # rel -> realpath
        hops: dict[str, list[str]] = {}  # rel -> basenames of every symlink hop target
        hidden_names: dict[tuple[int, int], set[str]] = {}  # (dev, ino) -> hidden alias names
        links: list[str] = []

        def _onerror(exc: OSError) -> None:
            raise ScoutError(f"cannot list {exc.filename}: {exc.strerror}") from exc

        for dirpath, dirnames, filenames in os.walk(self._root, onerror=_onerror, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for name in sorted(filenames):
                full = os.path.join(dirpath, name)
                rel = Path(os.path.relpath(full, self._root)).as_posix()
                if name.startswith("."):
                    # Never listed, but a hidden weight-named link to a listed file's inode
                    # still taints it. Errors here are ignored: hidden files never fail resolve().
                    try:
                        chain = _link_chain(full, rel)
                        hst = os.stat(full)
                    except (OSError, ScoutError):
                        continue
                    hidden_names.setdefault((hst.st_dev, hst.st_ino), set()).update([name, *chain])
                    continue
                try:
                    chain = _link_chain(full, rel)
                    st = os.stat(full)  # follows symlinks (HF cache snapshots -> ../blobs/)
                except OSError as exc:
                    if os.path.islink(full):
                        raise ScoutError(f"broken symlink {rel}") from exc
                    raise
                if not stat.S_ISREG(st.st_mode):
                    continue  # fifos, sockets, devices: never listed, never read
                stats[rel] = (st.st_size, st.st_mtime_ns)
                idents[rel] = (st.st_dev, st.st_ino)
                nlinks[rel] = st.st_nlink
                reals[rel] = os.path.realpath(full)
                hops[rel] = chain
                if chain:
                    links.append(rel)

        rels = sorted(stats)
        key = "\n".join(f"{rel}\t{stats[rel][0]}\t{stats[rel][1]}" for rel in rels)
        self._stats = stats
        self._idents = idents
        self._nlinks = nlinks
        self._effective = self._classification_names(rels, idents, reals, hops, hidden_names)
        self._files = [RepoFile(rel, stats[rel][0]) for rel in rels]
        self.revision_sha = hashlib.sha256(key.encode()).hexdigest()
        self.log.record(
            event="fetch", source="local", path=STAT_PATH, url=None, start=None,
            nbytes=0, status=None, note=self._stat_note(len(rels), sorted(links), reals),
        )

    def _stat_note(self, n: int, links: list[str], reals: dict[str, str]) -> str:
        """The @local/stat note. It is persisted in the Card, so it never holds an absolute path:
        targets are relative to root, or "<outside>/basename" outside root/../.."""
        note = f"{n} files"
        if not links:
            return note
        anchor = str(self._root.parent.parent)
        entries = []
        for rel in links[:MAX_NOTE_LINKS]:
            real = reals[rel]
            try:
                inside = os.path.commonpath([real, anchor]) == anchor
            except ValueError:
                inside = False
            target = (Path(os.path.relpath(real, self._root)).as_posix() if inside
                      else f"<outside>/{os.path.basename(real)}")
            entries.append(f"{rel} -> {target}")
        if len(links) > MAX_NOTE_LINKS:
            entries.append(f"(+{len(links) - MAX_NOTE_LINKS} more)")
        return note + "; symlinks: " + "; ".join(entries)

    @staticmethod
    def _classification_names(
        rels: list[str],
        idents: dict[str, tuple[int, int]],
        reals: dict[str, str],
        hops: dict[str, list[str]],
        hidden_names: dict[tuple[int, int], set[str]],
    ) -> dict[str, str]:
        """rel -> the name passed to preflight/record.

        ByteLog classifies by name, so every name that reaches a file's bytes counts. The
        aliases of a listed file are all names attached to its target inode: the basename of
        each listed file on that inode, every symlink hop target of those files (followed with
        readlink, so readme.md -> model.bin -> <hash> yields model.bin), the realpath basename,
        and hidden files in walked directories on the same inode.

        - Own name non-weight: if any alias is weight-class, the effective name is
          f"{rel} -> {alias}" (ends with the alias's weight extension, so preflight refuses
          it); .safetensors aliases are preferred and refused outright in _effective_name.
        - Own name .safetensors: if any alias has a different weight extension (.bin, .pt, ...),
          the effective name is f"{rel} -> {alias}", which preflight refuses (no safetensors
          header semantics over another format). Extensionless aliases (HF blobs) and other
          .safetensors names keep the own name.
        - Own name any other weight extension: unchanged (already refused).
        """
        names: dict[tuple[int, int], set[str]] = {}
        for rel in rels:
            bucket = names.setdefault(idents[rel], set())
            bucket.add(os.path.basename(rel))
            bucket.add(os.path.basename(reals[rel]))
            bucket.update(hops[rel])
        for ident, hidden in hidden_names.items():
            if ident in names:
                names[ident].update(hidden)
        out: dict[str, str] = {}
        for rel in rels:
            own_ext = _weight_ext(rel)
            aliases = names[idents[rel]]
            if own_ext is None:
                weighty = sorted((a for a in aliases if _is_weight_name(a)),
                                 key=lambda a: (not a.lower().endswith(".safetensors"), a))
            elif own_ext == ".safetensors":
                weighty = sorted(a for a in aliases if _weight_ext(a) not in (None, ".safetensors"))
            else:
                weighty = []
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
        if (
            (st.st_size, st.st_mtime_ns) != self._stats[path]
            or (st.st_dev, st.st_ino) != self._idents[path]
            or st.st_nlink != self._nlinks[path]  # a hard link (possibly weight-named) was added
        ):
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
