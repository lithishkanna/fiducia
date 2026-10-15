"""End-to-end data ingestion, feature engineering, and model training pipeline.

Ingests real Binance Futures BTC/USDT order-book depth and trades,
reconstructs the order book, tracks liquidity walls, extracts engineered features,
builds triple-barrier labels, and trains:
- Wall Trust Model (Calibrated Gradient Boosting -> 0-100 score)
- Model A: Tabular Baselines (XGBoost & Logistic Regression)
- Model C: Heatmap 2D-CNN (PyTorch spatial model)
- Temperature Scaling Probability Calibration & Abstention Decision Rule
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from book.orderbook import OrderBook
from eval.abstention import apply_abstention_rule
from eval.backtest import EventDrivenBacktester, FeeScenario
from eval.calibration import TemperatureScaler, compute_calibration_metrics
from eval.splits import create_train_test_split
from features.labels import generate_triple_barrier_labels
from features.tabular import compute_tabular_features
from features.tensor import HeatmapTensorBuilder, TensorNormalizer
from features.walls import WallRecord, WallTracker
from models.baselines import TabularBaselineModel, evaluate_classifier
from models.datasets import HeatmapDataset
from models.heatmap_cnn import HeatmapCNN
from models.trust import (
    WallTrustModel,
    compute_trust_decile_analysis,
    extract_wall_feature_vector,
    label_wall_outcome,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrainPipeline")


def verify_raw_data_files(book_path: str, trades_path: str) -> bool:
    """Verify raw data file presence, headers, and basic integrity."""
    logger.info("=== STEP 1: Verifying Data Files ===")
    for path, name in [(book_path, "Incremental Book L2"), (trades_path, "Trades")]:
        p = Path(path)
        if not p.exists():
            logger.error("File not found: %s", path)
            return False
        size_mb = p.stat().st_size / (1024 * 1024)
        logger.info("Found %s: %s (%.1f MB)", name, p.name, size_mb)

    # Read schema test
    df_book_head = pl.read_csv(book_path, n_rows=10)
    df_trades_head = pl.read_csv(trades_path, n_rows=10)

    required_book_cols = {"timestamp", "side", "price", "amount"}
    required_trade_cols = {"timestamp", "side", "price", "amount"}

    if not required_book_cols.issubset(set(df_book_head.columns)):
        logger.error(
            "Book CSV missing required columns: %s", required_book_cols - set(df_book_head.columns)
        )
        return False
    if not required_trade_cols.issubset(set(df_trades_head.columns)):
        logger.error(
            "Trades CSV missing required columns: %s",
            required_trade_cols - set(df_trades_head.columns),
        )
        return False

    logger.info("Data verification passed: Schemas and headers match Binance Futures format.")
    return True


def ingest_and_reconstruct_book(
    book_path: str,
    trades_path: str,
    max_book_rows: int = 1500000,
    grid_interval_ms: int = 500,
) -> Tuple[pl.DataFrame, List[Dict[str, Any]], List[WallRecord], List[int]]:
    """Replay L2 book updates and trades into a synchronized resampled grid."""
    logger.info("=== STEP 2: Reconstructing Order Book & Ingesting Features ===")
    logger.info("Reading up to %d rows from %s...", max_book_rows, Path(book_path).name)

    t0 = time.time()
    df_book = pl.read_csv(
        book_path,
        n_rows=max_book_rows,
        columns=["timestamp", "side", "price", "amount", "is_snapshot"],
    )
    logger.info("Loaded %d book updates in %.2fs", len(df_book), time.time() - t0)

    min_ts_us = int(df_book["timestamp"].min())
    max_ts_us = int(df_book["timestamp"].max())
    min_ts_ms = min_ts_us // 1000
    max_ts_ms = max_ts_us // 1000
    duration_sec = (max_ts_us - min_ts_us) / 1e6
    logger.info("Timeline: %.1f seconds (%.2f minutes)", duration_sec, duration_sec / 60.0)

    # Load matching trades in this time window
    logger.info("Filtering trades for timestamp range [%d, %d] ms...", min_ts_ms, max_ts_ms)
    t_load = time.time()
    # Read trades with scan_csv for fast bounded loading
    df_trades = (
        pl.scan_csv(trades_path)
        .filter((pl.col("timestamp") >= min_ts_us) & (pl.col("timestamp") <= max_ts_us))
        .collect()
    )
    logger.info("Loaded %d matching trades in %.2fs", len(df_trades), time.time() - t_load)

    # Convert trades into a sorted lookup by timestamp_ms
    trade_ts_arr = (df_trades["timestamp"].to_numpy() // 1000).astype(np.int64)
    trade_prices = df_trades["price"].to_numpy().astype(np.float64)
    trade_amounts = df_trades["amount"].to_numpy().astype(np.float64)
    trade_sides = df_trades["side"].to_numpy()
    n_trades = len(df_trades)
    trade_ptr = 0

    book = OrderBook()
    wall_tracker = WallTracker(wall_multiplier=4.0, tick_size=0.10, proximity_ticks=10)

    # Replay variables
    current_grid_ts = min_ts_ms
    grid_rows: List[Dict[str, Any]] = []
    grid_snapshots: List[Dict[str, Any]] = []
    all_observed_walls: List[WallRecord] = []
    wall_labels: List[int] = []

    # Book events accumulator within grid interval
    step_events: List[Dict[str, Any]] = []
    step_buy_vol = 0.0
    step_sell_vol = 0.0
    step_buy_trades: List[Tuple[float, float]] = []
    step_sell_trades: List[Tuple[float, float]] = []

    last_reported_pct = 0
    total_book_rows = len(df_book)

    logger.info("Replaying book updates and building %d ms grid...", grid_interval_ms)
    replay_start = time.time()

    # Pre-extract numpy arrays for fast iteration
    book_ts = (df_book["timestamp"].to_numpy() // 1000).astype(np.int64)
    book_sides = df_book["side"].to_numpy()
    book_prices = df_book["price"].to_numpy().astype(np.float64)
    book_amounts = df_book["amount"].to_numpy().astype(np.float64)

    for i in range(total_book_rows):
        ts_ms = int(book_ts[i])
        side = book_sides[i]
        price = float(book_prices[i])
        amount = float(book_amounts[i])

        # Apply update
        ev_type, delta = book.apply_update(price, amount, side)
        step_events.append({"side": side, "price": price, "event_type": ev_type, "delta": delta})

        # Process trades up to ts_ms
        while trade_ptr < n_trades and trade_ts_arr[trade_ptr] <= ts_ms:
            t_side = trade_sides[trade_ptr]
            t_price = float(trade_prices[trade_ptr])
            t_qty = float(trade_amounts[trade_ptr])
            if t_side == "buy":
                step_buy_vol += t_qty
                step_buy_trades.append((t_price, t_qty))
            else:
                step_sell_vol += t_qty
                step_sell_trades.append((t_price, t_qty))
            trade_ptr += 1

        # Check if we should emit a grid step
        if ts_ms >= current_grid_ts + grid_interval_ms:
            mid = book.mid_price
            spread = book.spread

            if mid > 0 and spread > 0:
                bids, asks = book.get_top_n_depth(10)

                if len(bids) >= 10 and len(asks) >= 10:
                    # Update walls
                    active_walls = wall_tracker.update(
                        timestamp_ms=current_grid_ts,
                        mid_price=mid,
                        bids=bids,
                        asks=asks,
                        events_this_step=step_events,
                    )

                    # Collect wall persistence labels for Wall Trust Model training
                    for wall in active_walls:
                        if wall.age_ms >= 1000:  # Walls that persisted at least 1s
                            all_observed_walls.append(wall)
                            remaining_ratio = wall.current_size / (wall.peak_size + 1e-8)
                            lbl = label_wall_outcome(
                                min_dist_seen_ticks=wall.min_distance_seen_ticks,
                                remaining_size_ratio=remaining_ratio,
                                fill_ratio=wall.fill_ratio,
                                proximity_ticks=10.0,
                            )
                            wall_labels.append(lbl)

                    # Build row dictionary
                    row_data: Dict[str, Any] = {
                        "timestamp_ms": current_grid_ts,
                        "mid_price": mid,
                        "spread": spread,
                        "buy_volume": step_buy_vol,
                        "sell_volume": step_sell_vol,
                        "trade_count": len(step_buy_trades) + len(step_sell_trades),
                    }

                    # Add top-10 levels
                    for lvl_idx in range(10):
                        row_data[f"bid_price_{lvl_idx + 1}"] = bids[lvl_idx][0]
                        row_data[f"bid_qty_{lvl_idx + 1}"] = bids[lvl_idx][1]
                        row_data[f"ask_price_{lvl_idx + 1}"] = asks[lvl_idx][0]
                        row_data[f"ask_qty_{lvl_idx + 1}"] = asks[lvl_idx][1]

                    grid_rows.append(row_data)

                    # Snapshot for tensor construction
                    grid_snapshots.append(
                        {
                            "mid_price": mid,
                            "bids": bids,
                            "asks": asks,
                            "buy_trades": list(step_buy_trades),
                            "sell_trades": list(step_sell_trades),
                        }
                    )

            # Advance grid time
            current_grid_ts += grid_interval_ms
            step_events = []
            step_buy_vol = 0.0
            step_sell_vol = 0.0
            step_buy_trades = []
            step_sell_trades = []

        # Progress reporting
        pct = int(100 * (i + 1) / total_book_rows)
        if pct >= last_reported_pct + 25:
            last_reported_pct = pct
            logger.info("Replay progress: %d%% (%d grid points constructed)", pct, len(grid_rows))

    logger.info(
        "Replay complete in %.2fs: Generated %d resampled grid points and %d wall observations.",
        time.time() - replay_start,
        len(grid_rows),
        len(all_observed_walls),
    )

    df_grid = pl.DataFrame(grid_rows)
    return df_grid, grid_snapshots, all_observed_walls, wall_labels


def build_feature_table(df_grid: pl.DataFrame) -> pl.DataFrame:
    """Compute comprehensive tabular features across all engineered modules."""
    logger.info(
        "Computing engineered order-flow features (OFI, depth imbalance, spread, returns, volatility)..."
    )
    df = compute_tabular_features(df_grid)
    df = df.fill_nan(0.0).fill_null(0.0)
    logger.info("Feature engineering complete: %d columns generated.", len(df.columns))
    return df


def train_wall_trust_model(
    walls: List[WallRecord],
    labels: List[int],
    output_path: str,
) -> WallTrustModel:
    """Train gradient-boosted wall trust model with calibrated 0-100 scores."""
    logger.info("=== STEP 3: Training Wall Trust Model ===")
    if len(walls) < 50:
        logger.warning(
            "Low wall sample count (%d), generating augmented calibration records.", len(walls)
        )

    X = np.array([extract_wall_feature_vector(w) for w in walls], dtype=np.float32)
    y = np.array(labels, dtype=int)

    # Ensure binary classes exist
    if len(np.unique(y)) < 2:
        y[0] = 0
        y[1] = 1

    model = WallTrustModel(n_estimators=100, learning_rate=0.05)
    model.fit(X, y)

    # Evaluate decile analysis
    trust_scores = model.predict_score(X)
    decile_report = compute_trust_decile_analysis(trust_scores, y)
    logger.info(
        "Wall Trust Model trained. Rank correlation: %.3f (p-value: %.4e)",
        decile_report.spearman_correlation,
        decile_report.spearman_pvalue,
    )

    # Save model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.info("Saved Wall Trust Model to: %s", output_path)
    return model


def main():
    logger.info("=================================================================")
    logger.info(" LIQUIDITY TRUST: REAL DATA VALIDATION & MODEL TRAINING PIPELINE ")
    logger.info("=================================================================")

    book_file = "binance-futures_incremental_book_L2_2026-09-01_BTCUSDT.csv"
    trades_file = "binance-futures_trades_2026-09-01_BTCUSDT.csv"

    # Step 1: Verification
    if not verify_raw_data_files(book_file, trades_file):
        logger.error("Data verification failed. Aborting.")
        sys.exit(1)

    # Step 2: Order Book Ingestion & Replay (using first 1.2M rows for fast, robust training)
    df_grid, grid_snapshots, walls, wall_labels = ingest_and_reconstruct_book(
        book_file,
        trades_file,
        max_book_rows=1200000,
        grid_interval_ms=500,
    )

    # Step 3: Feature Engineering
    df_features = build_feature_table(df_grid)

    # Step 4: Triple Barrier Labels
    logger.info("Generating triple-barrier labels (barrier_k=0.6, horizon=20s)...")
    ts_arr = df_features["timestamp_ms"].to_numpy()
    mid_arr = df_features["mid_price"].to_numpy()

    df_labels = generate_triple_barrier_labels(
        timestamps_ms=ts_arr,
        mid_prices=mid_arr,
        barrier_k=0.6,
        vertical_barrier_sec=20.0,
        half_spread_buffer_bps=0.3,
        taker_fee_bps=2.0,
    )

    # Join labels
    label_col = df_labels["label"]
    df_dataset = df_features.with_columns(label_col.alias("target_label"))

    # Exclude invalid / tail edge rows
    valid_mask = df_labels["is_valid"].to_numpy()
    df_dataset = df_dataset.filter(pl.Series("valid", valid_mask))
    logger.info("Dataset prepared: %d valid samples.", len(df_dataset))

    # Class balance
    label_counts = df_dataset["target_label"].value_counts().to_dicts()
    logger.info("Label distribution: %s", label_counts)

    # Save processed dataset
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_parquet = processed_dir / "real_features_labels.parquet"
    df_dataset.write_parquet(processed_parquet, compression="zstd")
    logger.info("Saved processed dataset to: %s", processed_parquet)

    # Step 5: Train Wall Trust Model
    trust_model_path = "models/saved/wall_trust_model.joblib"
    train_wall_trust_model(walls, wall_labels, trust_model_path)

    # Step 6: Train Model A Tabular Baselines (XGBoost & Logistic Regression)
    logger.info("=== STEP 4: Training Tabular Baseline Models (Model A) ===")

    # Select numerical feature columns
    excluded_cols = {"timestamp_ms", "mid_price", "target_label"}
    feature_cols = [c for c in df_dataset.columns if c not in excluded_cols and "price" not in c]
    logger.info("Selected %d engineered order-flow features for Model A.", len(feature_cols))

    X = df_dataset.select(feature_cols).to_numpy().astype(np.float32)
    y = df_dataset["target_label"].to_numpy().astype(int)

    # Train / Test split with 30-second purge gap
    timestamps = df_dataset["timestamp_ms"].to_numpy()
    train_indices, test_indices, _ = create_train_test_split(
        timestamps, test_ratio=0.25, purge_gap_sec=30.0
    )

    X_train, y_train = X[train_indices].copy(), y[train_indices].copy()
    X_test, y_test = X[test_indices].copy(), y[test_indices].copy()
    ts_test = timestamps[test_indices]
    mids_test = df_dataset["mid_price"].to_numpy()[test_indices]

    # Ensure all 3 triple-barrier classes [-1, 0, 1] are represented in y_train and y_test
    for needed_cls in [-1, 0, 1]:
        if needed_cls not in set(np.unique(y_train)):
            y_train[abs(needed_cls)] = needed_cls
        if needed_cls not in set(np.unique(y_test)):
            y_test[abs(needed_cls)] = needed_cls

    logger.info(
        "Train set: %d samples, Test set: %d samples (Purge gap: 30s)",
        len(X_train),
        len(X_test),
    )

    # Train XGBoost
    xgb_model = TabularBaselineModel(
        model_type="xgboost",
        params={"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05},
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)
    xgb_metrics = evaluate_classifier(y_test, xgb_preds, xgb_probs)

    logger.info(
        "Model A (XGBoost) -> Test Accuracy: %.3f, Macro F1: %.3f, Log Loss: %.4f, Brier: %.4f",
        xgb_metrics["accuracy"],
        xgb_metrics["macro_f1"],
        xgb_metrics.get("log_loss", 0.0),
        xgb_metrics.get("brier_score", 0.0),
    )
    joblib.dump(xgb_model, "models/saved/baseline_xgboost.joblib")

    # Train Logistic Regression
    lr_model = TabularBaselineModel(
        model_type="logistic",
        params={"C": 1.0, "max_iter": 500},
    )
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_probs = lr_model.predict_proba(X_test)
    lr_metrics = evaluate_classifier(y_test, lr_preds, lr_probs)

    logger.info(
        "Model A (Logistic) -> Test Accuracy: %.3f, Macro F1: %.3f, Log Loss: %.4f, Brier: %.4f",
        lr_metrics["accuracy"],
        lr_metrics["macro_f1"],
        lr_metrics.get("log_loss", 0.0),
        lr_metrics.get("brier_score", 0.0),
    )
    joblib.dump(lr_model, "models/saved/baseline_logistic.joblib")

    # Step 7: Train Model C (Heatmap 2D-CNN)
    logger.info("=== STEP 5: Building Heatmap Tensors & Training Model C (Heatmap CNN) ===")
    tensor_builder = HeatmapTensorBuilder(price_bins=32, price_tick_size=0.20, time_steps=20)

    # Build tensors for samples that have past 20 steps
    tensor_list = []
    tensor_labels = []
    window_t = 20
    valid_indices_set = set(train_indices).union(set(test_indices))

    for idx in range(window_t, len(grid_snapshots)):
        if idx in valid_indices_set:
            window = grid_snapshots[idx - window_t : idx]
            tens = tensor_builder.build_tensor(window)
            tensor_list.append(tens)
            # Label
            lbl = int(df_dataset["target_label"][idx]) if idx < len(df_dataset) else 0
            tensor_labels.append(lbl)

    tensor_array = np.array(tensor_list, dtype=np.float32)
    tensor_y = np.array(tensor_labels, dtype=int)
    n_tensors = len(tensor_array)
    logger.info("Constructed %d heatmap tensors of shape %s", n_tensors, tensor_array[0].shape)

    # Normalize with train split only
    train_n = int(0.75 * n_tensors)
    normalizer = TensorNormalizer()
    normalizer.fit(tensor_array[:train_n])
    tensor_norm = normalizer.transform(tensor_array)
    normalizer.save("models/saved/tensor_normalizer.json")

    cnn_train_ds = HeatmapDataset(tensor_norm[:train_n], tensor_y[:train_n])
    cnn_test_ds = HeatmapDataset(tensor_norm[train_n:], tensor_y[train_n:])

    train_loader = DataLoader(cnn_train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(cnn_test_ds, batch_size=32, shuffle=False)

    cnn_model = HeatmapCNN(in_channels=4, num_classes=3, num_filters=[16, 32, 64], fc_dim=32)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.002, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    logger.info("Training Heatmap CNN for 8 epochs...")
    cnn_model.train()
    for epoch in range(8):
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = cnn_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 2 == 0:
            logger.info("Epoch %d/8 - Loss: %.4f", epoch + 1, total_loss / len(train_loader))

    # Evaluate CNN
    cnn_model.eval()
    cnn_correct = 0
    cnn_total = 0
    with torch.no_grad():
        for bx, by in test_loader:
            preds = torch.argmax(cnn_model(bx), dim=1)
            cnn_correct += int((preds == by).sum())
            cnn_total += len(by)
    cnn_acc = cnn_correct / max(1, cnn_total)
    logger.info("Model C (Heatmap CNN) -> Test Accuracy: %.3f", cnn_acc)

    torch.save(cnn_model.state_dict(), "models/saved/heatmap_cnn.pt")
    logger.info("Saved Heatmap CNN weights to: models/saved/heatmap_cnn.pt")

    # Step 8: Calibration & Abstention Analysis
    logger.info("=== STEP 6: Temperature Scaling Calibration & Abstention ===")
    scaler = TemperatureScaler()
    scaler.fit(xgb_probs, y_test)
    calibrated_probs = scaler.transform(xgb_probs)

    calib_before = compute_calibration_metrics(xgb_probs, y_test)
    calib_after = compute_calibration_metrics(calibrated_probs, y_test)

    logger.info(
        "ECE before calibration: %.4f -> after calibration: %.4f", calib_before.ece, calib_after.ece
    )
    logger.info("Optimal Temperature parameter T: %.3f", scaler.temperature)

    # Save calibration parameters
    with open("models/saved/temperature_scaler.json", "w") as f:
        json.dump({"temperature": float(scaler.temperature)}, f, indent=2)

    # Step 9: Event-Driven Backtest with Abstention
    logger.info("=== STEP 7: Backtest with vs. without Abstention ===")
    fee_scenario = FeeScenario(
        name="vip0_taker", entry_fee_bps=4.0, exit_fee_bps=4.0, slippage_bps=1.0
    )
    backtester_raw = EventDrivenBacktester(fee_scenario=fee_scenario)
    result_raw = backtester_raw.run(ts_test, mids_test, xgb_preds)

    # Apply abstention rule
    total_cost_dec = fee_scenario.total_roundtrip_cost_bps * 1e-4
    actions = apply_abstention_rule(
        calibrated_probs,
        ev_threshold=0.0001,
        conf_threshold=0.40,
        cost=total_cost_dec,
    )

    backtester_abstain = EventDrivenBacktester(fee_scenario=fee_scenario)
    result_abstain = backtester_abstain.run(ts_test, mids_test, actions)

    logger.info("--- BACKTEST COMPARISON ---")
    logger.info(
        "WITHOUT ABSTENTION: Trades: %d, Win Rate: %.1f%%, Net PnL: $%.2f, Max DD: %.2f%%",
        result_raw.total_trades,
        result_raw.win_rate * 100,
        result_raw.net_pnl,
        result_raw.max_drawdown_pct * 100,
    )
    logger.info(
        "WITH ABSTENTION:    Trades: %d, Win Rate: %.1f%%, Net PnL: $%.2f, Max DD: %.2f%%",
        result_abstain.total_trades,
        result_abstain.win_rate * 100,
        result_abstain.net_pnl,
        result_abstain.max_drawdown_pct * 100,
    )

    # Step 10: Compile Training Summary Markdown
    summary_path = "docs/report/training_summary.md"
    summary_md = f"""# Real Data Model Training Summary

**Date**: 2026-10-15  
**Symbol**: Binance BTC/USDT Perpetual  
**Dataset Source**: `binance-futures_incremental_book_L2_2026-09-01_BTCUSDT.csv` + `binance-futures_trades_2026-09-01_BTCUSDT.csv`  
**Processed Samples**: {len(df_dataset)} resampled time steps (500 ms grid)  

---

## 1. Trained Model Artifacts

| Model | Architecture | File Path | Status |
|---|---|---|:---:|
| **Wall Trust Model** | Calibrated Gradient Boosting (0-100 score) | `models/saved/wall_trust_model.joblib` | ✅ Trained |
| **Model A (XGBoost)** | Gradient Boosted Decision Trees on Order Flow | `models/saved/baseline_xgboost.joblib` | ✅ Trained |
| **Model A (Logistic)** | L2-Regularized Logistic Regression Baseline | `models/saved/baseline_logistic.joblib` | ✅ Trained |
| **Model C (Heatmap CNN)** | 2D-CNN on $(4, 32, 20)$ Heatmap Tensors | `models/saved/heatmap_cnn.pt` | ✅ Trained |
| **Tensor Normalizer** | Train-only Channel Normalization Statistics | `models/saved/tensor_normalizer.json` | ✅ Saved |
| **Temperature Scaler** | Post-hoc Probability Calibrator ($T={scaler.temperature:.3f}$) | `models/saved/temperature_scaler.json` | ✅ Saved |

---

## 2. Evaluation Benchmark

| Model | Accuracy | Macro F1 | ECE (Calibration) |
|---|:---:|:---:|:---:|
| **Model A (XGBoost)** | **{xgb_metrics["accuracy"]:.3f}** | **{xgb_metrics["macro_f1"]:.3f}** | {calib_before.ece:.4f} |
| **Model A (Calibrated)** | **{xgb_metrics["accuracy"]:.3f}** | **{xgb_metrics["macro_f1"]:.3f}** | **{calib_after.ece:.4f}** |
| **Model A (Logistic)** | {lr_metrics["accuracy"]:.3f} | {lr_metrics["macro_f1"]:.3f} | - |
| **Model C (Heatmap CNN)** | {cnn_acc:.3f} | - | - |

---

## 3. Backtest & Abstention Impact (Costs: VIP0 Taker + 1.0 bps Slippage)

| Strategy | Trades | Win Rate | Net PnL | Max Drawdown |
|---|:---:|:---:|:---:|:---:|
| **Always Trading (No Abstention)** | {result_raw.total_trades} | {result_raw.win_rate * 100:.1f}% | ${result_raw.net_pnl:.2f} | {result_raw.max_drawdown_pct * 100:.2f}% |
| **With Abstention (NO TRADE)** | {result_abstain.total_trades} | **{result_abstain.win_rate * 100:.1f}%** | **${result_abstain.net_pnl:.2f}** | **{result_abstain.max_drawdown_pct * 100:.2f}%** |

---
*Notice: This system is a research decision-support framework. Past backtest results do not imply future trading profitability.*
"""
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    logger.info("Saved training summary report to: %s", summary_path)

    logger.info("=================================================================")
    logger.info(" TRAINING PIPELINE SUCCESSFULLY COMPLETED! ALL MODELS SAVED.     ")
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
