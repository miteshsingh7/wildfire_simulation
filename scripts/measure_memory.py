"""Measure peak RAM across parent and child process tree using psutil (Task 6).

Monitors:
1. PyTorchFire 12-hour simulation on canonical grid (1,128 x 1,124).
2. Cell2Fire C++ subprocess execution on canonical grid.

Samples process tree memory every 20ms to capture maximum RSS of parent + all child processes.
"""

import threading
import time
from pathlib import Path

import pandas as pd
import psutil
import yaml

from wf052.sim.cell2fire_engine import run_cell2fire_simulation
from wf052.sim.pytorchfire_engine import run_pytorchfire_simulation


class ProcessTreeMemoryMonitor:
    """High-frequency background monitor sampling process tree RSS."""

    def __init__(self, interval_sec: float = 0.02) -> None:
        self.interval_sec = interval_sec
        self.stop_event = threading.Event()
        self.peak_rss_bytes = 0
        self.peak_parent_rss_bytes = 0
        self.peak_children_rss_bytes = 0
        self.num_samples = 0
        self.thread: threading.Thread | None = None

    def _monitor(self) -> None:
        parent = psutil.Process()
        while not self.stop_event.is_set():
            try:
                parent_rss = parent.memory_info().rss
                children_rss = 0
                for child in parent.children(recursive=True):
                    try:
                        children_rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                total_rss = parent_rss + children_rss
                if total_rss > self.peak_rss_bytes:
                    self.peak_rss_bytes = total_rss
                    self.peak_parent_rss_bytes = parent_rss
                    self.peak_children_rss_bytes = children_rss
                self.num_samples += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(self.interval_sec)

    def start(self) -> None:
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self) -> tuple[float, float, float]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        mb = 1024.0 * 1024.0
        return (
            self.peak_rss_bytes / mb,
            self.peak_parent_rss_bytes / mb,
            self.peak_children_rss_bytes / mb,
        )


def measure_engine_memory() -> pd.DataFrame:
    with open("configs/phase0.yaml") as f:
        _ = yaml.safe_load(f)

    results = []

    # 1. PyTorchFire Peak Memory
    print("Benchmarking PyTorchFire memory...")
    monitor = ProcessTreeMemoryMonitor(interval_sec=0.02)
    monitor.start()
    t0 = time.perf_counter()
    _ = run_pytorchfire_simulation(config_path="configs/phase0.yaml", seed=42)
    elapsed_pt = time.perf_counter() - t0
    peak_total_pt, peak_parent_pt, peak_children_pt = monitor.stop()

    results.append({
        "engine": "PyTorchFire",
        "runtime_sec": elapsed_pt,
        "peak_tree_ram_mb": peak_total_pt,
        "parent_ram_mb": peak_parent_pt,
        "children_ram_mb": peak_children_pt,
        "monitoring_method": "psutil process tree (parent + children recursive @ 20ms)",
    })
    print(
        f"  PyTorchFire: Total={peak_total_pt:.2f} MB "
        f"(Parent={peak_parent_pt:.2f} MB, Child={peak_children_pt:.2f} MB) in {elapsed_pt:.2f}s"
    )

    # 2. Cell2Fire Peak Memory
    print("Benchmarking Cell2Fire memory...")
    monitor_c2f = ProcessTreeMemoryMonitor(interval_sec=0.02)
    monitor_c2f.start()
    t0 = time.perf_counter()
    _ = run_cell2fire_simulation(config_path="configs/phase0.yaml", seed=42)
    elapsed_c2f = time.perf_counter() - t0
    peak_total_c2f, peak_parent_c2f, peak_children_c2f = monitor_c2f.stop()

    results.append({
        "engine": "Cell2Fire",
        "runtime_sec": elapsed_c2f,
        "peak_tree_ram_mb": peak_total_c2f,
        "parent_ram_mb": peak_parent_c2f,
        "children_ram_mb": peak_children_c2f,
        "monitoring_method": "psutil process tree (parent + children recursive @ 20ms)",
    })
    print(
        f"  Cell2Fire: Total={peak_total_c2f:.2f} MB "
        f"(Parent={peak_parent_c2f:.2f} MB, Child={peak_children_c2f:.2f} MB) in {elapsed_c2f:.2f}s"
    )

    df = pd.DataFrame(results)
    out_csv = Path("reports/process_tree_memory.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")
    return df


def main() -> None:
    df = measure_engine_memory()
    print("\n=== Process Tree RAM Summary ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
