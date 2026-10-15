"""ML models: baselines, deep LOB, heatmap CNN, fusion, wall trust."""

from models.baselines import (
    TabularBaselineModel,
    decode_labels,
    encode_labels,
    evaluate_classifier,
)
from models.datasets import DeepLOBDataset, FusionDataset, HeatmapDataset
from models.deeplob import DeepLOB
from models.fusion import DualBranchFusion, SpatialBranch, TabularBranch
from models.heatmap_cnn import HeatmapCNN
from models.trust import (
    WALL_FEATURE_NAMES,
    DecileBinResult,
    TrustDecileReport,
    WallTrustModel,
    compute_trust_decile_analysis,
    extract_wall_feature_vector,
    label_wall_outcome,
)

__all__ = [
    "TabularBaselineModel",
    "encode_labels",
    "decode_labels",
    "evaluate_classifier",
    "WallTrustModel",
    "extract_wall_feature_vector",
    "label_wall_outcome",
    "compute_trust_decile_analysis",
    "DecileBinResult",
    "TrustDecileReport",
    "WALL_FEATURE_NAMES",
    "DeepLOB",
    "HeatmapCNN",
    "DualBranchFusion",
    "SpatialBranch",
    "TabularBranch",
    "HeatmapDataset",
    "DeepLOBDataset",
    "FusionDataset",
]

__version__ = "0.1.0"
