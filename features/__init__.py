"""Feature engineering, order book heatmaps, wall tracking, and triple-barrier labels."""

from features.labels import compute_class_balance_report, generate_triple_barrier_labels
from features.tabular import (
    compute_center_of_mass,
    compute_depth_imbalance,
    compute_ofi,
    compute_rolling_tabular_features,
    compute_tabular_features,
)
from features.tensor import HeatmapTensorBuilder, TensorNormalizer, build_heatmap_tensor
from features.walls import (
    WallRecord,
    WallTracker,
    compute_wall_trust_features,
    detect_liquidity_walls,
    extract_wall_summary_features,
)

__all__ = [
    "compute_tabular_features",
    "compute_depth_imbalance",
    "compute_ofi",
    "compute_center_of_mass",
    "compute_rolling_tabular_features",
    "build_heatmap_tensor",
    "HeatmapTensorBuilder",
    "TensorNormalizer",
    "WallTracker",
    "WallRecord",
    "detect_liquidity_walls",
    "extract_wall_summary_features",
    "compute_wall_trust_features",
    "generate_triple_barrier_labels",
    "compute_class_balance_report",
]

__version__ = "0.1.0"
