"""Data integrity: SHA-256 checksums of the inputs and key intermediate files.

The licensed raw exports cannot be published, but their checksums can: a
checksum reveals nothing about the content, yet anyone holding the same
exports can confirm that they are byte-for-byte the files behind the
published results (``python -m jsemicro verify``).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .config import PATHS, data_root

TRACKED = ("raw_tx", "raw_md", "tx_flat", "tx_trimmed", "md_flat")


def sha256(path: Path, block: int = 1 << 24) -> str:
    """SHA-256 of a file, streamed in 16 MB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for buf in iter(lambda: fh.read(block), b""):
            h.update(buf)
    return h.hexdigest()


def current_checksums() -> dict[str, str]:
    """Checksums of the tracked files that exist, keyed by path relative to the data root."""
    out = {}
    for attr in TRACKED:
        p = getattr(PATHS, attr)
        if p.exists():
            out[str(p.relative_to(data_root()))] = sha256(p)
    return out


def write_checksums() -> Path:
    path = data_root() / "SHA256SUMS"
    lines = [f"{h}  {name}" for name, h in current_checksums().items()]
    path.write_text("\n".join(lines) + "\n")
    return path


def read_checksums(path: Path | None = None) -> dict[str, str]:
    path = path or data_root() / "SHA256SUMS"
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            h, name = line.split(maxsplit=1)
            out[name.strip()] = h
    return out


def verify_checksums(path: Path | None = None) -> list[str]:
    """Names of files whose checksum differs from the record (missing files are skipped)."""
    want = read_checksums(path)
    have = current_checksums()
    return [n for n, h in want.items() if n in have and have[n] != h]
