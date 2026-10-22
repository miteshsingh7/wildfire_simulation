"""PyTorch ML surrogate models for fast wildfire spread emulation.

Exposes:
- Dataset and data contract: SurrogateSample, SurrogateBatch, SurrogateDataset
- Stand-in data generation & exclusions: generate_ensemble_samples, load_fire_rasters_crop, verify_fire_not_excluded
- Diagnostics: compute_data_diagnostics, DataDiagnosticsReport
- Compact storage: save_compact_dataset, CompactSurrogateDataset, compute_storage_footprint
- Router Adapter: surrogate_output_to_arrival_field, stitch_tiles_to_arrival_field, TilePrediction
- Conformal Calibration: SplitConformalCalibrator, ConformalCalibrationResult
- Latency & Benchmarks: benchmark_surrogate_latency, LatencyBenchmarkResult
- Architecture: WindConditionedUNet, WindEncoder, FiLMBlock
- Losses: SurrogateMultiTaskLoss, PinballLoss, QuantileOrderingLoss
- Baselines: AnalyticWindEllipseBaseline, TeacherBaseline
- Evaluation: evaluate_model_on_samples, generate_fidelity_report_markdown
- Trainer: SurrogateTrainer, TrainConfig, set_reproducible_seed, split_leave_fire_out
"""

from wf052.surrogate.adapter import (
    TilePrediction,
    stitch_tiles_to_arrival_field,
    surrogate_output_to_arrival_field,
)
from wf052.surrogate.baselines import (
    AnalyticWindEllipseBaseline,
    BaselinePrediction,
    TeacherBaseline,
)
from wf052.surrogate.calibration import (
    ConformalCalibrationResult,
    SplitConformalCalibrator,
)
from wf052.surrogate.dataset import (
    SurrogateBatch,
    SurrogateDataset,
    SurrogateSample,
    collate_surrogate_batch,
    compute_slope_aspect,
    fbfm40_to_fuel_group,
)
from wf052.surrogate.diagnostics import (
    DataDiagnosticsReport,
    compute_data_diagnostics,
)
from wf052.surrogate.evaluation import (
    HorizonMetrics,
    SurrogateFidelityMetrics,
    evaluate_model_on_samples,
    generate_fidelity_report_markdown,
)
from wf052.surrogate.latency import (
    LatencyBenchmarkResult,
    benchmark_surrogate_latency,
)
from wf052.surrogate.losses import (
    LossBreakdown,
    PinballLoss,
    QuantileOrderingLoss,
    SurrogateMultiTaskLoss,
)
from wf052.surrogate.model import (
    FiLMBlock,
    ModelOutput,
    WindConditionedUNet,
    WindEncoder,
)
from wf052.surrogate.standin_generator import (
    SEVEN_TRAINING_FIRES,
    generate_analytic_arrival_surface,
    generate_ensemble_samples,
    get_teacher_config_hash,
    load_fire_rasters_crop,
    load_held_out_fire_exclusions,
    simulate_cellular_standin_arrival,
    verify_fire_not_excluded,
)
from wf052.surrogate.storage import (
    CompactSurrogateDataset,
    StorageFootprintReport,
    compute_storage_footprint,
    save_compact_dataset,
)
from wf052.surrogate.trainer import (
    SurrogateTrainer,
    TrainConfig,
    set_reproducible_seed,
    split_leave_fire_out,
)

__all__ = [
    "AnalyticWindEllipseBaseline",
    "BaselinePrediction",
    "CompactSurrogateDataset",
    "ConformalCalibrationResult",
    "DataDiagnosticsReport",
    "FiLMBlock",
    "HorizonMetrics",
    "LatencyBenchmarkResult",
    "LossBreakdown",
    "ModelOutput",
    "PinballLoss",
    "QuantileOrderingLoss",
    "SEVEN_TRAINING_FIRES",
    "SplitConformalCalibrator",
    "StorageFootprintReport",
    "SurrogateBatch",
    "SurrogateDataset",
    "SurrogateFidelityMetrics",
    "SurrogateMultiTaskLoss",
    "SurrogateSample",
    "SurrogateTrainer",
    "TeacherBaseline",
    "TilePrediction",
    "TrainConfig",
    "WindConditionedUNet",
    "WindEncoder",
    "benchmark_surrogate_latency",
    "collate_surrogate_batch",
    "compute_data_diagnostics",
    "compute_slope_aspect",
    "compute_storage_footprint",
    "evaluate_model_on_samples",
    "fbfm40_to_fuel_group",
    "generate_analytic_arrival_surface",
    "generate_ensemble_samples",
    "generate_fidelity_report_markdown",
    "get_teacher_config_hash",
    "load_fire_rasters_crop",
    "load_held_out_fire_exclusions",
    "save_compact_dataset",
    "set_reproducible_seed",
    "simulate_cellular_standin_arrival",
    "split_leave_fire_out",
    "stitch_tiles_to_arrival_field",
    "surrogate_output_to_arrival_field",
    "verify_fire_not_excluded",
]
