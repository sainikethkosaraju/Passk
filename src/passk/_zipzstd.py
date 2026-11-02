"""Read members of zip archives that use compression method 93 (zstd), as Inspect's `.eval`
logs do. Python's `zipfile` cannot decompress them, so the compressed bytes are located via the
central directory and decompressed here. Needs `zstandard` (extra `passk[eval]`) or Python 3.14."""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path

ZSTD = 93


def _decompress(data: bytes, size_hint: int) -> bytes:
    try:
        import zstandard

        return zstandard.ZstdDecompressor().decompress(data, max_output_size=max(size_hint, 1) * 2 + 64)
    except ImportError:
        pass
    try:
        from compression import zstd  # type: ignore[import-not-found]  # Python 3.14+

        return zstd.decompress(data)
    except ImportError as e:
        raise RuntimeError("this .eval log is zstd-compressed: install `passk[eval]` "
                           "(zstandard) to read it") from e


def read_member(path: str | Path, z: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    if info.compress_type != ZSTD:
        return z.read(info)
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        header = f.read(30)
        if header[:4] != b"PK\x03\x04":
            raise ValueError("corrupt zip local header")
        name_len, extra_len = struct.unpack("<HH", header[26:30])
        f.seek(info.header_offset + 30 + name_len + extra_len)
        raw = f.read(info.compress_size)
    out = _decompress(raw, info.file_size)
    if len(out) != info.file_size:
        raise ValueError(f"{info.filename}: size mismatch after zstd decompression")
    return out
