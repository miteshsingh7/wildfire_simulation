"""PyTorch ML surrogate models for fast wildfire spread emulation.

Exposes:
- Dataset and data contract: SurrogateSample, SurrogateBatch, SurrogateDataset
- Stand-in data generation: generate_ensemble_samples, load_fire_rasters_crop
- Architecture: WindConditionedUNet, WindEncoder, FiLMBlock
- Losses: SurrogateMultiTaskLoss, PinballLoss, QuantileOrderingLoss
- Baselines: AnalyticWindEllipseBaseline, TeacherBaseline
- Evaluation: evaluate_model_on_samples, generate_fidelity_report_markdown
- Trainer: SurrogateTrainer, TrainConfig, set_reproducible_seed, split_leave_fire_out
"""

from wf052.surrogate.baselines import (
    AnalyticWindEllipseBaseline,
    BaselinePrediction,
    TeacherBaseline,
)
from wf052.surrogate.dataset import (
    SurrogateBatch,
    SurrogateDataset,
    SurrogateSample,
    collate_surrogate_batch,
    compute_slope_aspect,
    fbfm40_to_fuel_group,
)
from wf052.surrogate.evaluation import (
    HorizonMetrics,
    SurrogateFidelityMetrics,
    evaluate_model_on_samples,
    generate_fidelity_report_markdown,
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
    "FiLMBlock",
    "HorizonMetrics",
    "LossBreakdown",
    "ModelOutput",
    "PinballLoss",
    "QuantileOrderingLoss",
    "SEVEN_TRAINING_FIRES",
    "SurrogateBatch",
    "SurrogateDataset",
    "SurrogateFidelityMetrics",
    "SurrogateMultiTaskLoss",
    "SurrogateSample",
    "SurrogateTrainer",
    "TeacherBaseline",
    "TrainConfig",
    "WindConditionedUNet",
    "WindEncoder",
    "collate_surrogate_batch",
    "compute_slope_aspect",
    "evaluate_model_on_samples",
    "fbfm40_to_fuel_group",
    "generate_analytic_arrival_surface",
    "generate_ensemble_samples",
    "generate_fidelity_report_markdown",
    "get_teacher_config_hash",
    "load_fire_rasters_crop",
    "set_reproducible_seed",
    "split_leave_fire_out",
]
