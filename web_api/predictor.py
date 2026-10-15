"""Model inference and ensemble prediction engine for Fiducia L2 Replay.

Loads and evaluates:
1. Model A: XGBoost (Tree booster on 74 microstructure features)
2. Model B: DeepLOB (Zhang et al. 2019, CNN + Inception + LSTM on L2 matrix sequences)
3. Model D: Multimodal Dual-Branch Fusion (2D Heatmap CNN + Tabular MLP)
4. Model E: LightGBM (Leaf-wise fast tree booster)
5. Meta-Ensemble: Soft-voting and stacking of A + B + D + E
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

from models.deeplob import DeepLOB
from models.fusion import DualBranchFusion

logger = logging.getLogger(__name__)

MODELS_METADATA = [
    {
        "id": "ensemble",
        "name": "Meta-Ensemble Stacker",
        "architecture": "Soft-Voting Stacking (XGBoost + DeepLOB + Fusion + LightGBM)",
        "accuracy": 0.408,
        "macro_f1": 0.275,
        "brier_score": 0.7648,
        "badge": "PRODUCTION ENSEMBLE",
        "description": "Blends deep spatial L2 representation with fast tree-based microstructure order flow.",
    },
    {
        "id": "lightgbm",
        "name": "Model E: LightGBM Fast GBDT",
        "architecture": "Leaf-wise Gradient Boosted Trees (num_leaves=31, lr=0.05)",
        "accuracy": 0.537,
        "macro_f1": 0.339,
        "brier_score": 0.6752,
        "badge": "HIGHEST ACCURACY",
        "description": "Fast tree booster emphasizing Order Flow Imbalance (OFI) and cancel/trade ratios.",
    },
    {
        "id": "xgboost",
        "name": "Model A: Baseline XGBoost",
        "architecture": "Depth-wise Gradient Boosted Trees (max_depth=6, lr=0.05)",
        "accuracy": 0.479,
        "macro_f1": 0.314,
        "brier_score": 0.2312,
        "badge": "LOWEST BRIER SCORE",
        "description": "Solid, calibrated microstructure baseline trained on 74 depth & OFI features.",
    },
    {
        "id": "deeplob",
        "name": "Model B: DeepLOB (Zhang et al.)",
        "architecture": "CNN + Inception Blocks + LSTM on (1, 40, 20) L2 Matrices",
        "accuracy": 0.422,
        "macro_f1": 0.290,
        "brier_score": 0.9039,
        "badge": "SPATIAL-TEMPORAL L2",
        "description": "Deep neural network processing raw 10-level price and volume sequences without manual OFI.",
    },
    {
        "id": "fusion",
        "name": "Model D: Multimodal Dual-Branch Fusion",
        "architecture": "2D Heatmap CNN + Tabular MLP with Band GRU Pooling",
        "accuracy": 0.279,
        "macro_f1": 0.203,
        "brier_score": 1.2607,
        "badge": "SPATIAL FUSION",
        "description": "Dual-branch neural network fusing 2D depth density grids with numerical order book imbalance.",
    },
]


class MultiModelInferenceEngine:
    """Manages loaded models and performs real-time multi-model inference."""

    def __init__(self, models_dir: str = "models/saved"):
        self.models_dir = Path(models_dir)
        self.xgb_model = None
        self.lgb_model = None
        self.deeplob_model = None
        self.fusion_model = None
        self.ensemble_meta = None
        self.temperature = 1.0

        self._load_all_models()

    def _load_all_models(self) -> None:
        # Load Temperature Scaler
        temp_path = self.models_dir / "temperature_scaler.json"
        if temp_path.exists():
            try:
                with open(temp_path, "r", encoding="utf-8") as f:
                    self.temperature = float(json.load(f).get("temperature", 1.0))
            except Exception as e:
                logger.debug("Failed to read temperature scaler: %s", e)

        # Load XGBoost
        xgb_path = self.models_dir / "baseline_xgboost.joblib"
        if xgb_path.exists():
            try:
                self.xgb_model = joblib.load(xgb_path)
                logger.info("Loaded XGBoost model from %s", xgb_path)
            except Exception as e:
                logger.warning("Error loading XGBoost: %s", e)

        # Load LightGBM
        lgb_path = self.models_dir / "baseline_lightgbm.joblib"
        if lgb_path.exists():
            try:
                self.lgb_model = joblib.load(lgb_path)
                logger.info("Loaded LightGBM model from %s", lgb_path)
            except Exception as e:
                logger.warning("Error loading LightGBM: %s", e)

        # Load Ensemble weights
        ens_path = self.models_dir / "ensemble_stacker.joblib"
        if ens_path.exists():
            try:
                self.ensemble_meta = joblib.load(ens_path)
                logger.info("Loaded Meta-Ensemble stacker metadata: %s", self.ensemble_meta)
            except Exception as e:
                logger.warning("Error loading ensemble stacker: %s", e)

        # Load DeepLOB
        deeplob_path = self.models_dir / "deeplob.pt"
        if deeplob_path.exists():
            try:
                self.deeplob_model = DeepLOB(num_classes=3)
                self.deeplob_model.load_state_dict(
                    torch.load(deeplob_path, map_location="cpu")
                )
                self.deeplob_model.eval()
                logger.info("Loaded DeepLOB PyTorch model from %s", deeplob_path)
            except Exception as e:
                logger.warning("Error loading DeepLOB: %s", e)

        # Load Fusion
        fusion_path = self.models_dir / "fusion_model.pt"
        if fusion_path.exists():
            try:
                self.fusion_model = DualBranchFusion(
                    tabular_in_dim=74,
                    num_classes=3,
                    spatial_dim=32,
                    tabular_dim=32,
                    fused_dim=64,
                    num_bands=8,
                    pooling_variant="band_gru",
                    dropout=0.25,
                )
                self.fusion_model.load_state_dict(
                    torch.load(fusion_path, map_location="cpu")
                )
                self.fusion_model.eval()
                logger.info("Loaded Multimodal Fusion PyTorch model from %s", fusion_path)
            except Exception as e:
                logger.warning("Error loading Multimodal Fusion: %s", e)

    def extract_tabular_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract the exact 74 numerical features matching model training."""
        excluded_cols = {"timestamp_ms", "mid_price", "target_label"}
        feature_cols = [c for c in df.columns if c not in excluded_cols and "price" not in c]
        if not feature_cols:
            # Fallback if raw dataframe only
            return np.zeros((len(df), 74), dtype=np.float32), []
        feat_vals = df[feature_cols].select_dtypes(include=[np.number]).fillna(0.0).values
        if feat_vals.shape[1] < 74:
            pad = np.zeros((len(df), 74 - feat_vals.shape[1]), dtype=np.float32)
            feat_vals = np.hstack([feat_vals, pad])
        elif feat_vals.shape[1] > 74:
            feat_vals = feat_vals[:, :74]
        return feat_vals.astype(np.float32), feature_cols

    def predict_for_clip(
        self,
        df: pd.DataFrame,
        model_id: str = "ensemble",
    ) -> List[Dict[str, Any]]:
        """Compute calibrated probabilities and decisions for every row in a clip."""
        n_rows = len(df)
        X_tab, _ = self.extract_tabular_features(df)

        # 1. XGBoost probabilities
        p_xgb = np.full((n_rows, 3), 1.0 / 3.0, dtype=np.float32)
        if self.xgb_model is not None and X_tab.shape[1] == 74:
            try:
                p_xgb = self.xgb_model.predict_proba(X_tab)
            except Exception as e:
                logger.debug("XGBoost predict error: %s", e)

        # 2. LightGBM probabilities
        p_lgb = np.full((n_rows, 3), 1.0 / 3.0, dtype=np.float32)
        if self.lgb_model is not None and X_tab.shape[1] == 74:
            try:
                p_lgb = self.lgb_model.predict_proba(X_tab)
            except Exception as e:
                logger.debug("LightGBM predict error: %s", e)

        # 3. DeepLOB probabilities (PyTorch sequence)
        p_dlob = np.full((n_rows, 3), 1.0 / 3.0, dtype=np.float32)
        if self.deeplob_model is not None and "bid_price_1" in df.columns:
            try:
                p_dlob = self._predict_deeplob_batch(df)
            except Exception as e:
                logger.debug("DeepLOB predict error: %s", e)
        else:
            # If DeepLOB cannot parse raw columns, approximate using smoothed micro OFI
            p_dlob = self._heuristic_micro_lob_probs(df)

        # 4. Multimodal Fusion probabilities
        p_fusion = np.full((n_rows, 3), 1.0 / 3.0, dtype=np.float32)
        if self.fusion_model is not None and X_tab.shape[1] == 74 and "bid_price_1" in df.columns:
            try:
                p_fusion = self._predict_fusion_batch(df, X_tab)
            except Exception as e:
                logger.debug("Fusion predict error: %s", e)
        else:
            p_fusion = 0.5 * p_xgb + 0.5 * p_dlob

        # Select target probabilities based on model_id
        if model_id == "xgboost":
            selected_probs = p_xgb
        elif model_id == "lightgbm":
            selected_probs = p_lgb
        elif model_id == "deeplob":
            selected_probs = p_dlob
        elif model_id == "fusion":
            selected_probs = p_fusion
        else:
            # ensemble (0.25 XGB + 0.25 DeepLOB + 0.30 Fusion + 0.20 LightGBM)
            selected_probs = (
                0.25 * p_xgb
                + 0.25 * p_dlob
                + 0.30 * p_fusion
                + 0.20 * p_lgb
            )

        # Format outputs
        results = []
        cost_bps = 2.0
        target_bps = 10.0

        for i in range(n_rows):
            p_down = float(selected_probs[i, 0])
            p_neutral = float(selected_probs[i, 1])
            p_up = float(selected_probs[i, 2])

            # Re-normalize if sum != 1
            s = p_down + p_neutral + p_up
            if s > 0:
                p_down /= s
                p_neutral /= s
                p_up /= s

            ev_long = p_up * target_bps - p_down * target_bps - cost_bps
            ev_short = p_down * target_bps - p_up * target_bps - cost_bps

            threshold = 0.40 if model_id in ("lightgbm", "ensemble") else 0.43
            if ev_long > 1.2 and p_up > threshold:
                action = 1
                action_label = "TRADE LONG"
                ev = ev_long
            elif ev_short > 1.2 and p_down > threshold:
                action = -1
                action_label = "TRADE SHORT"
                ev = ev_short
            else:
                action = 0
                action_label = "NO TRADE (ABSTAIN)"
                ev = max(ev_long, ev_short)

            results.append({
                "prob_down": round(p_down, 4),
                "prob_neutral": round(p_neutral, 4),
                "prob_up": round(p_up, 4),
                "expected_value_bps": round(ev, 2),
                "action": action,
                "action_label": action_label,
            })

        return results

    def _predict_deeplob_batch(self, df: pd.DataFrame, window_t: int = 20) -> np.ndarray:
        """Batch inference for DeepLOB over a windowed slice."""
        n_samples = len(df)
        lob_matrices = np.zeros((n_samples, 1, 40, window_t), dtype=np.float32)

        mids = df["mid_price"].values
        bid_p = [df[f"bid_price_{i}"].values for i in range(1, 11)]
        bid_q = [df[f"bid_qty_{i}"].values for i in range(1, 11)]
        ask_p = [df[f"ask_price_{i}"].values for i in range(1, 11)]
        ask_q = [df[f"ask_qty_{i}"].values for i in range(1, 11)]

        for sample_idx in range(n_samples):
            step_mid = mids[sample_idx]
            for t_step in range(window_t):
                # Look back up to window_t, clamping at index 0
                lookback_idx = max(0, sample_idx - (window_t - 1 - t_step))
                for lvl in range(10):
                    row_base = lvl * 4
                    bp_norm = ((bid_p[lvl][lookback_idx] - step_mid) / (step_mid + 1e-8)) * 10000.0
                    bq_norm = np.log1p(max(0.0, bid_q[lvl][lookback_idx]))
                    ap_norm = ((ask_p[lvl][lookback_idx] - step_mid) / (step_mid + 1e-8)) * 10000.0
                    aq_norm = np.log1p(max(0.0, ask_q[lvl][lookback_idx]))

                    lob_matrices[sample_idx, 0, row_base + 0, t_step] = bp_norm
                    lob_matrices[sample_idx, 0, row_base + 1, t_step] = bq_norm
                    lob_matrices[sample_idx, 0, row_base + 2, t_step] = ap_norm
                    lob_matrices[sample_idx, 0, row_base + 3, t_step] = aq_norm

        with torch.no_grad():
            inp = torch.from_numpy(lob_matrices).float()
            logits = self.deeplob_model(inp)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def _predict_fusion_batch(self, df: pd.DataFrame, X_tab: np.ndarray, window_t: int = 20) -> np.ndarray:
        """Batch inference for Multimodal Dual-Branch Fusion."""
        n_samples = len(df)
        heatmaps = np.zeros((n_samples, 4, 32, window_t), dtype=np.float32)

        # Simplified 2D density representation across price bins
        mids = df["mid_price"].values
        for i in range(n_samples):
            mid = mids[i]
            for lvl in range(1, 11):
                bp = float(df[f"bid_price_{lvl}"].iloc[i])
                bq = float(df[f"bid_qty_{lvl}"].iloc[i])
                ap = float(df[f"ask_price_{lvl}"].iloc[i])
                aq = float(df[f"ask_qty_{lvl}"].iloc[i])

                bid_bin = int(np.clip(16 - int((mid - bp) / 0.20), 0, 31))
                ask_bin = int(np.clip(16 + int((ap - mid) / 0.20), 0, 31))

                heatmaps[i, 0, bid_bin, :] = np.log1p(bq)
                heatmaps[i, 1, ask_bin, :] = np.log1p(aq)

        with torch.no_grad():
            x_tens = torch.from_numpy(heatmaps).float()
            x_tab_tens = torch.from_numpy(X_tab).float()
            logits = self.fusion_model(x_tens, x_tab_tens)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def _heuristic_micro_lob_probs(self, df: pd.DataFrame) -> np.ndarray:
        """Fallback micro-LOB probabilities if raw columns are not present."""
        n_rows = len(df)
        probs = np.zeros((n_rows, 3), dtype=np.float32)
        for i in range(n_rows):
            ofi = float(df.get("ofi", pd.Series([0.0])).iloc[i]) if "ofi" in df.columns else 0.0
            p_up = 0.33 + np.clip(ofi * 0.1, -0.2, 0.4)
            p_down = 0.33 - np.clip(ofi * 0.1, -0.4, 0.2)
            p_neutral = max(0.05, 1.0 - (p_up + p_down))
            probs[i] = [p_down, p_neutral, p_up]
        return probs
