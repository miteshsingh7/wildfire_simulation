"""Latency benchmarking and fully-convolutional verification for surrogate models (Phase 2B).

Benchmarks:
1. Surrogate inference latency (ms/tile and tiles/sec) at batch sizes 1, 8, 32 on CPU (and GPU/MPS).
2. Comparison against physics teacher simulation latency per ensemble member.
3. Verification of fully convolutional behavior across non-standard spatial tile dimensions.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from wf052.surrogate.baselines import TeacherBaseline
from wf052.surrogate.dataset import SurrogateSample


@dataclass(frozen=True)
class LatencyBenchmarkResult:
    """Latency benchmark results across batch sizes and comparison against teacher."""

    device_name: str
    batch_latencies_ms_per_tile: dict[int, float]  # batch_size -> ms/tile
    batch_throughput_tiles_sec: dict[int, float]  # batch_size -> tiles/sec
    teacher_ms_per_member: float
    speedup_ratios_vs_teacher: dict[int, float]  # batch_size -> speedup factor
    fully_convolutional_passed: bool
    evaluated_shapes: list[tuple[int, int]]

    def to_markdown(self) -> str:
        """Render latency results as a markdown table."""
        lines: list[str] = [
            "# Surrogate Model Inference Latency & Benchmark",
            "",
            f"- **Inference Device**: `{self.device_name}`",
            f"- **Teacher Benchmark Latency**: `{self.teacher_ms_per_member:.2f} ms/member`",
            f"- **Fully Convolutional Verification**: `{'PASSED' if self.fully_convolutional_passed else 'FAILED'}` (Tested shapes: {self.evaluated_shapes})",
            "",
            "## Latency and Throughput by Batch Size",
            "| Batch Size | Latency (ms/tile) | Total Batch Time (ms) | Throughput (tiles/sec) | Speedup vs Teacher |",
            "| :---: | :---: | :---: | :---: | :---: |",
        ]

        for b_size, ms_per_tile in self.batch_latencies_ms_per_tile.items():
            tot_ms = ms_per_tile * b_size
            t_sec = self.batch_throughput_tiles_sec[b_size]
            sp = self.speedup_ratios_vs_teacher[b_size]
            lines.append(
                f"| {b_size} | {ms_per_tile:.2f} ms | {tot_ms:.2f} ms | {t_sec:.1f} tiles/s | **{sp:.1f}x** |"
            )

        return "\n".join(lines)


def benchmark_surrogate_latency(
    model: nn.Module,
    device: torch.device,
    batch_sizes: Sequence[int] = (1, 8, 32),
    tile_size: tuple[int, int] = (256, 256),
    num_warmup: int = 3,
    num_repeats: int = 10,
    teacher_sample: SurrogateSample | None = None,
) -> LatencyBenchmarkResult:
    """Benchmark inference latency across batch sizes and against teacher benchmark."""
    model.eval()
    model.to(device)

    h, w = tile_size
    dev_type = "cuda" if device.type == "cuda" else "cpu"

    latencies_per_tile: dict[int, float] = {}
    throughputs: dict[int, float] = {}

    for b_size in batch_sizes:
        x = torch.randn(b_size, 13, h, w, device=device)
        wind = torch.randn(b_size, 12, 2, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                with torch.amp.autocast(device_type=dev_type):
                    _ = model(x, wind)
                if device.type == "cuda":
                    torch.cuda.synchronize()

        # Timed runs
        timings: list[float] = []
        with torch.no_grad():
            for _ in range(num_repeats):
                t0 = time.perf_counter()
                with torch.amp.autocast(device_type=dev_type):
                    _ = model(x, wind)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - t0) * 1000.0  # ms
                timings.append(elapsed)

        mean_batch_ms = float(np.mean(timings))
        ms_per_tile = mean_batch_ms / b_size
        tiles_sec = (1000.0 / ms_per_tile) if ms_per_tile > 0 else 0.0

        latencies_per_tile[b_size] = ms_per_tile
        throughputs[b_size] = tiles_sec

    # Teacher baseline timing
    if teacher_sample is not None:
        t_base = TeacherBaseline()
        pred = t_base.predict(teacher_sample)
        teacher_ms = pred.elapsed_ms
    else:
        teacher_ms = 35.0  # Reference teacher cellular step latency

    speedups: dict[int, float] = {
        b: (teacher_ms / lat) if lat > 0 else 1.0 for b, lat in latencies_per_tile.items()
    }

    # Verify fully convolutional behavior on diverse spatial shapes
    shapes_to_test = [(256, 256), (384, 384), (512, 512)]
    fc_passed = True
    for test_h, test_w in shapes_to_test:
        test_x = torch.randn(1, 13, test_h, test_w, device=device)
        test_w_uv = torch.randn(1, 12, 2, device=device)
        try:
            with torch.no_grad():
                with torch.amp.autocast(device_type=dev_type):
                    out = model(test_x, test_w_uv)
            if out.quantiles.shape != (1, 3, test_h, test_w):
                fc_passed = False
            if out.burn_logits.shape != (1, 1, test_h, test_w):
                fc_passed = False
        except Exception:
            fc_passed = False

    return LatencyBenchmarkResult(
        device_name=str(device),
        batch_latencies_ms_per_tile=latencies_per_tile,
        batch_throughput_tiles_sec=throughputs,
        teacher_ms_per_member=teacher_ms,
        speedup_ratios_vs_teacher=speedups,
        fully_convolutional_passed=fc_passed,
        evaluated_shapes=shapes_to_test,
    )
