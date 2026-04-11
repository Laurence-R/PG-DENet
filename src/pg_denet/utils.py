"""Shared utility helpers."""

import time


def timed(name: str, fn, *args, **kwargs):
    """Execute *fn* and print elapsed time; return its result."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    ms = (time.perf_counter() - t0) * 1000
    print(f"  {name:<20s} {ms:8.1f} ms")
    return result


def timed_ms(name: str, fn, *args, **kwargs):
    """Execute *fn*, print elapsed time, return ``(result, elapsed_ms)``."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    ms = (time.perf_counter() - t0) * 1000
    print(f"  {name:<20s} {ms:8.1f} ms")
    return result, ms


def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    """Pretty-print a table to the terminal."""
    col_w = [max(len(h), *(len(r[i]) for r in rows)) + 2 for i, h in enumerate(headers)]
    sep = "  ".join("-" * w for w in col_w)
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(f"\n--- {title} ---")
    print(header_line)
    print(sep)
    for row in rows:
        print("  ".join(v.ljust(w) for v, w in zip(row, col_w)))
