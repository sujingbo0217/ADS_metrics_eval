"""
.fbin I/O helpers.

File layout (little-endian):
    offset 0  : uint32  N   (number of vectors)
    offset 4  : uint32  D   (dimension)
    offset 8  : float32 x N*D   (row-major)

This matches the big-ann-benchmarks / DiskANN fbin convention so the same
files can be consumed by the CUDA side without extra parsing.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np

HEADER_DTYPE = np.dtype(np.uint32).newbyteorder("<")
VEC_DTYPE = np.dtype(np.float32).newbyteorder("<")
HEADER_BYTES = 8


def write_fbin(path: str, arr: np.ndarray) -> None:
    """Write a 2-D float array to a .fbin file."""
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {arr.shape}")
    arr = np.ascontiguousarray(arr, dtype=VEC_DTYPE)
    n, d = arr.shape
    if n > np.iinfo(np.uint32).max or d > np.iinfo(np.uint32).max:
        raise ValueError(f"N={n}, D={d} exceeds uint32 range")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        np.array([n, d], dtype=HEADER_DTYPE).tofile(f)
        arr.tofile(f)


def read_fbin(path: str, mmap: bool = False) -> np.ndarray:
    """Read a .fbin file and return a (N, D) float32 array."""
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(HEADER_BYTES), dtype=HEADER_DTYPE)
    n, d = int(header[0]), int(header[1])

    expected = HEADER_BYTES + n * d * VEC_DTYPE.itemsize
    actual = os.path.getsize(path)
    if actual != expected:
        raise IOError(
            f"{path}: size mismatch (header says N={n}, D={d} -> "
            f"{expected} bytes, file is {actual} bytes)"
        )

    if mmap:
        return np.memmap(
            path, dtype=VEC_DTYPE, mode="r",
            offset=HEADER_BYTES, shape=(n, d),
        )
    with open(path, "rb") as f:
        f.seek(HEADER_BYTES)
        return np.frombuffer(f.read(), dtype=VEC_DTYPE).reshape(n, d).copy()


def peek_fbin(path: str) -> Tuple[int, int]:
    """Return (N, D) without loading the data."""
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(HEADER_BYTES), dtype=HEADER_DTYPE)
    return int(header[0]), int(header[1])
