"""Background GPU memory profiler for the activation extraction wrapper.

start_profiler() spawns a daemon thread that polls torch.cuda memory stats
every interval_sec and appends a CSV row. mark_phase() lets the wrapper tag
phase boundaries (model_load, auto_batch_probe, strategy_chat_last, upload,
etc.) which render as vertical dashed lines on the rendered PNG.

stop() ends the polling thread; render_png() draws the plot. Both are safe
to call when CUDA isn't available — they short-circuit.
"""
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

JOIN_GRACE = 5  # seconds to wait for the polling thread on stop()


class GPUProfiler:
    def __init__(self, csv_path: Path, phases_path: Path, interval_sec: float = 3.0):
        self.csv_path = csv_path
        self.phases_path = phases_path
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0: float | None = None

    def start(self) -> None:
        try:
            import torch
            if not torch.cuda.is_available():
                return
        except Exception:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w") as f:
            f.write("t_seconds,memory_allocated_mib,memory_reserved_mib,memory_total_mib\n")
        with self.phases_path.open("w") as f:
            f.write("t_seconds,phase\n")
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        try:
            import torch
        except Exception:
            return
        try:
            total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        except Exception:
            total = 0
        while not self._stop.is_set():
            try:
                t = time.monotonic() - (self._t0 or time.monotonic())
                alloc = torch.cuda.memory_allocated(0) // (1024 * 1024)
                reserved = torch.cuda.memory_reserved(0) // (1024 * 1024)
                with self.csv_path.open("a") as f:
                    f.write(f"{t:.2f},{alloc},{reserved},{total}\n")
            except Exception:
                pass
            self._stop.wait(self.interval_sec)

    def mark_phase(self, name: str) -> None:
        if self._t0 is None:
            return
        try:
            t = time.monotonic() - self._t0
            with self.phases_path.open("a") as f:
                f.write(f"{t:.2f},{name}\n")
        except Exception:
            pass

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=JOIN_GRACE)


def render_png(csv_path: Path, phases_path: Path, png_path: Path, title: str = "") -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    rows: list[tuple[float, int, int, int]] = []
    try:
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append((
                    float(row["t_seconds"]),
                    int(row["memory_allocated_mib"]),
                    int(row["memory_reserved_mib"]),
                    int(row["memory_total_mib"]),
                ))
    except Exception:
        return False
    if not rows:
        return False
    phases: list[tuple[float, str]] = []
    try:
        with phases_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                phases.append((float(row["t_seconds"]), row["phase"]))
    except Exception:
        pass

    times = [r[0] for r in rows]
    alloc = [r[1] for r in rows]
    reserved = [r[2] for r in rows]
    total = rows[-1][3] if rows else 0

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(times, alloc, alpha=0.4, label="memory_allocated", color="#1f77b4")
    ax.plot(times, reserved, label="memory_reserved", color="#ff7f0e", linewidth=1.5)
    if total > 0:
        ax.axhline(total, linestyle=":", color="red", label=f"total ({total} MiB)")
    for t, name in phases:
        ax.axvline(t, linestyle="--", color="gray", alpha=0.5)
        ax.text(t, ax.get_ylim()[1] * 0.98, name, rotation=90,
                verticalalignment="top", fontsize=8, color="gray")
    ax.set_xlabel("seconds since profile start")
    ax.set_ylabel("MiB")
    ax.set_title(title or csv_path.stem)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    return True
