# Progress Log

---

## Phase 0: Setup ✅

**Date**: 2026-10-07
**Status**: Complete

### What was done
- Created repository structure with all packages: `recorder/`, `book/`, `features/`, `models/`, `eval/`, `explain/`, `dashboard/`
- Created `pyproject.toml` with all dependencies and dev tooling config (pytest, ruff, mypy)
- Created virtual environment (`.venv/`) and installed all dependencies including dev extras
- Created `configs/` with five YAML config files: `data.yaml`, `features.yaml`, `labels.yaml`, `models.yaml`, `eval.yaml`
- Created `docs/decisions.md` and `docs/progress.md`
- Created `.gitignore`, `README.md`, `AGENT_BUILD_PLAN.md`
- All modules have docstrings describing their purpose
- Created `data/raw/`, `data/interim/`, `data/processed/` with `.gitkeep` files
- Created `notebooks/` directory
- Created recorder CLI entry point (`recorder/__main__.py`)

### What was tested
- **`pytest` (7/7 passed)**: smoke tests for all 6 package imports + OrderBook basic operations
- **`ruff check .`**: all checks passed (0 errors)
- All packages importable from the installed editable package

### Known issues
- None

### Next step
- Phase 1: Recorder

---

## Phase 1: Recorder ✅

**Date**: 2026-10-11
**Status**: Complete

### What was done
- Implemented `recorder/stream.py`: Binance Futures WebSocket combined stream client with exponential backoff and local receive timestamping.
- Implemented `recorder/sync.py`: strict sequence synchronization according to official Binance Futures documentation (`u < lastUpdateId` discard, `U <= lastUpdateId <= u` first event, `pu == u_prev` sequential validation, and gap marker generation on mismatches).
- Implemented `recorder/writer.py`: PyArrow Parquet writer with batched flushing, date-based partitioning, hourly file rotation, and explicit gap marker rows.
- Implemented `recorder/__main__.py`: complete CLI runner supporting heartbeat logging (events/min, gap count, reconnect count).
- Added comprehensive unit tests in `tests/test_sync.py` and `tests/test_writer.py`.

### What was tested
- `tests/test_sync.py`: normal flow, buffering, snapshot alignment, gap detection, duplicate handling, out-of-order events, late snapshot, buffer overflow, state reset.
- `tests/test_writer.py`: stream classification, field extraction, gap row schema, batch flushing, and Parquet readability.

### Known issues
- Live recording requires active network connectivity to Binance Futures.

---

## Phase 2: Order-Book Reconstruction and Event Inference ✅

**Date**: 2026-10-11
**Status**: Complete

### What was done
- Implemented `book/orderbook.py`:
  - `OrderBook` maintaining sorted bids (descending) and asks (ascending).
  - Efficient level insertion, modification, and removal (`qty=0`).
  - Cache-backed top-N depth extraction and level lookup.
  - Property accessors (`best_bid`, `best_ask`, `mid_price`, `spread`) and crossed book detection.
  - Snapshot initialization and diff depth update application returning atomic change records.
  - Invalidation and gap-state tracking.
- Implemented `book/events.py`:
  - Per-level `BookEvent` derivation (`add`, `reduce`).
  - Heuristic fill vs cancel attribution matching against `aggTrade` timestamps and side conventions within a configurable window.
- Implemented `book/validate.py`:
  - Independent reconciliation between locally reconstructed book and REST snapshots, reporting mismatch rate by level.
- Implemented `book/replay.py`:
  - Deterministic replay of recorded raw Parquet logs through the book, propagating gap flags and generating inferred events.
- Created unit and property tests in `tests/test_orderbook.py`, `tests/test_events.py`, `tests/test_validate.py`, and `tests/test_replay.py`.

### What was tested
- All 65 tests in `tests/` passing (100% pass rate).
- Property tests: no crossed book on sequential valid updates, no negative sizes after updates.
- Validation logic against synthetic snapshots (exact match, quantity mismatch, level discrepancies).
- Replay pipeline testing with snapshot initialization and gap marker carrying.
- `ruff check .` passes cleanly with zero warnings/errors.

### Known issues
- None.

### Next step
- Phase 3: Features, Tensor and Labels

---

## Phase 3: Features, Tensor and Labels ✅

**Date**: 2026-10-11
**Status**: Complete

### What was done
- Implemented `features/tabular.py`:
  - Multi-level Order Flow Imbalance (OFI) based on Cont et al. formulation.
  - Top-N depth imbalance ($N \in [1, 5, 10]$).
  - Relative spread, multi-lag mid returns, realized volatility.
  - Signed trade volume, total trade volume, trade flow imbalance, and trade intensity.
  - Near-mid order book event metrics: add volume, cancel volume, cancel-to-add ratio.
  - Liquidity Center of Mass (CoM) for bids, asks, and relative book CoM, plus CoM velocity.
  - Strictly backward-looking causal rolling computations with no forward lookahead.
- Implemented `features/tensor.py`:
  - In-memory 3D heatmap tensor construction ($C \times P \times T$) with $C=4$ channels (`bid_size`, `ask_size`, `executed_buy_vol`, `executed_sell_vol`).
  - Price axis centered around reference mid-price in discrete ticks.
  - Non-linear `log1p` scaling.
  - `TensorNormalizer` enforcing strict train-only normalization statistics (mean, std) with serialization support.
- Implemented `features/walls.py`:
  - Liquidity wall detector identifying levels exceeding $5\times$ local median depth.
  - Stateful `WallTracker` recording wall lifecycle: age in ms, fill vs cancel attribution accumulation, rapid flicker detection within time windows, and price approach behavior.
  - Summary feature extractor producing aggregated wall imbalance, max wall sizes, and min wall distances.
- Implemented `features/labels.py`:
  - Triple-barrier labeling method with transaction fee and spread threshold barriers ($k \cdot (\text{half\_spread} + \text{fee\_bps} \cdot mid)$).
  - Vertical horizon cutoff (5 to 30 seconds).
  - Target labels: $+1$ (UP), $-1$ (DOWN), $0$ (NEUTRAL), with recorded `time_to_hit_ms` and realized return at hit.
  - Automatic invalidation of samples overlapping gap intervals or truncated at data edges.
  - `compute_class_balance_report` providing distribution diagnostics and recommendations.
- Added comprehensive unit tests in `tests/test_features_tabular.py`, `tests/test_features_tensor.py`, `tests/test_features_walls.py`, and `tests/test_features_labels.py`.

### What was tested
- All **82 tests** passing in `pytest` (100% pass rate).
- **Leakage acceptance test**: Verified that perturbing future data after time $t$ has zero effect on features at or before $t$.
- **Synthetic label tests**: Verified deterministic barrier hits on upward trends, downward trends, and range-bound flat paths.
- **Normalizer test**: Verified channel mean=0 and std=1 on training split, with zero test set leakage.
- `ruff check .` and `ruff format .` pass with 0 errors.

### Known issues
- None.

### Next step
- Phase 4: Baselines and Evaluation Framework

---

## Phase 4: Baselines and Evaluation Framework ✅

**Date**: 2026-10-12
**Status**: Complete

### What was done
- Implemented `eval/splits.py`:
  - `walk_forward_splits`: expanding-window time-ordered cross-validation splits with an explicit purge gap ($\ge 30\text{s}$) between training and validation periods.
  - `create_train_test_split`: partitions timeline into train/validation pool and a final untouched test holdout (default 20%) isolated by a purge gap.
- Implemented `models/baselines.py`:
  - **Model A**: Unified `TabularBaselineModel` wrapper supporting Logistic Regression (with `StandardScaler` and balanced class weights) and XGBoost multiclass classifier.
  - Label encoding and decoding between triple-barrier classes `[-1, 0, 1]` and internal indices `[0, 1, 2]`.
  - Comprehensive classification evaluation: balanced accuracy, macro F1, per-class F1, log loss, and multiclass Brier score.
  - Feature importances extraction.
- Implemented `eval/backtest.py`:
  - `EventDrivenBacktester`: realistic simulation with position limits (1.0 BTC unit), mark-to-market equity curve, and detailed trade logs (`TradeRecord`).
  - Transaction cost modeling: configurable `FeeScenario` (maker/taker fees) and adverse execution slippage penalty (1.0 bps per leg).
  - Stream gap enforcement: immediately forces position liquidation upon entering a gap with `exit_reason="gap"` and prohibits new trade entries during gap intervals.
  - Performance analytics: Net PnL, Gross PnL, Total Fees, Total Slippage, Win Rate, Profit Factor, Sharpe ratio, and Max Drawdown %.
- Implemented `eval/bootstrap.py`:
  - `moving_block_bootstrap`: stationary block resampling preserving temporal autocorrelation.
  - `compute_block_bootstrap_ci`: empirical percentile bootstrap (default 1000 resamples, 95% CI) for financial returns and evaluation metrics.
- Added comprehensive unit tests in `tests/test_splits.py`, `tests/test_baselines.py`, `tests/test_backtest.py`, and `tests/test_bootstrap.py`.

### What was tested
- All **92 tests** passing in `pytest` (100% pass rate).
- **Mandatory Sanity Check 1**: Verified that an "always NEUTRAL" strategy executes **0 trades** and yields 0.0 PnL and 0 fees.
- **Mandatory Sanity Check 2**: Verified that random trading on a flat price series incurs 0 gross PnL and strictly loses the exact transaction costs (fees + slippage).
- **Purge Gap Verification**: Verified that validation sets start $\ge \text{purge\_gap\_ms}$ after training periods end, with zero index overlap.
- **Model A Baseline Validation**: Verified Logistic Regression and XGBoost training, probability normalization ($\sum p_i = 1$), and metric calculation.
- **Block Bootstrap**: Verified 95% confidence intervals contain true parameters on synthetic series.
- `ruff check .` passes with 0 errors.

### Known issues
- None.

### Next step
- Phase 5: Wall Trust Model

---

## Phase 5: Wall Trust Model ✅

**Date**: 2026-10-12
**Status**: Complete

### What was done
- Defined ground-truth wall trust label in `docs/decisions.md` (Decision 007):
  - Outcome = 1 (Trustworthy): wall stands ($\ge 50\%$ size) or is traded into ($\ge 20\%$ fills) when price approaches within proximity ($N=10$ ticks).
  - Outcome = 0 (Low-Trust / Fleeting): wall pulled / cancelled before or upon approach.
- Implemented `models/trust.py`:
  - `extract_wall_feature_vector`: extracts causal features strictly known up to evaluation time: `[relative_size, distance_ticks, distance_bps, age_ms, flicker_count, cum_fill_qty, cum_cancel_qty, fill_ratio, approach_count, is_bid, book_imbalance]`.
  - `WallTrustModel`: gradient-boosted classifier wrapped in `CalibratedClassifierCV(method='sigmoid')` ensuring posterior calibration, mapping probabilities to an integer trust score in $[0, 100]$.
  - `compute_trust_decile_analysis`: validation decile reporting grouping walls into 10 score bins, computing empirical persistence rates, 95% Wilson score confidence intervals, and Spearman rank correlation answering research question RQ2.
- Integrated trust scores into tabular feature set via `compute_wall_trust_features` in `features/walls.py`:
  - `max_bid_wall_trust`, `max_ask_wall_trust`, `mean_bid_wall_trust`, `mean_ask_wall_trust`, and `wall_trust_imbalance`.
- Added comprehensive unit tests in `tests/test_trust.py`.

### What was tested
- All **96 tests** passing in `pytest` (100% pass rate).
- **Mandatory Acceptance Check**: Verified that wall features at time $t$ use ONLY data up to $t$, and are 100% invariant to future size spikes or cancels.
- **Calibrated Score Bounds**: Verified calibrated probabilities in $[0, 1]$ and trust scores strictly in $[0, 100]$.
- **RQ2 Decile Validation Report**: Confirmed positive rank correlation with Wilson score confidence intervals across all 10 deciles.
- `ruff check .` passes with 0 errors.

### Known issues
- None.

### Next step
- Phase 6: Deep Models

---

## Phase 6: Deep Models ✅

**Date**: 2026-10-12
**Status**: Complete

### What was done
- Implemented `models/datasets.py`:
  - `HeatmapDataset`: PyTorch dataset yielding `(tensor, label)` with $(4, P, T)$ tensors built on the fly, with `TensorNormalizer` integration and without saving images to disk.
  - `DeepLOBDataset`: PyTorch dataset yielding `(lob_matrix, label)` with $(1, 4 \cdot \text{levels}, T)$ order book matrices.
  - `FusionDataset`: PyTorch dataset yielding `(tensor, tabular, label)` for dual-branch architecture.
- Implemented `models/deeplob.py` (**Model B**):
  - Architecture adapted from Zhang et al. (2019) with level-by-level convolutions, Inception modules with multi-scale receptive fields, adaptive spatial pooling, LSTM temporal dynamics, and classification head.
  - Stabilized with convolutional batch normalization.
- Implemented `models/heatmap_cnn.py` (**Model C**):
  - 2D Convolutional Neural Network operating on $(4, P, T)$ heatmap tensor with batch norm, LeakyReLU, spatial pooling, and dense projection layers.
- Implemented `models/fusion.py` (**Model D**):
  - Dual-branch architecture fusing spatial and tabular representations:
    - **Spatial branch**: 2D-CNN processing the heatmap tensor with configurable price-band pooling variants:
      1. `band_gru` (default): adaptive pooling over price axis to 8 bands, retaining distance-from-mid structure over time, followed by a temporal GRU to dimension $d=64$.
      2. `gap_time`: global average pooling over time axis, keeping price bands.
      3. `full_gap`: global average pooling across both price and time axes.
    - **Tabular branch**: MLP projecting engineered order flow and wall trust features to dimension $d=64$.
    - **Fusion head**: Concatenation $(2d=128) \to \text{Dropout} \to \text{Dense} \to \text{Logits}(3)$.
- Added unit and overfitting sanity tests in `tests/test_deep_models.py`.

### What was tested
- All **102 tests** passing in `pytest` (100% pass rate).
- **Mandatory Overfitting Sanity Checks**: Verified that DeepLOB, HeatmapCNN, and DualBranchFusion (across all 3 pooling variants: `band_gru`, `gap_time`, `full_gap`) quickly overfit a tiny subset with steep loss reductions and 100% training accuracy.
- **Dataset Batching**: Verified shapes and label encodings through PyTorch DataLoaders.
- `ruff check .` passes with 0 errors.

### Known issues
- None.

### Next step
- Phase 7: Calibration and Abstention

---

## Phase 7: Calibration and Abstention ✅

**Date**: 2026-10-12
**Status**: Complete

### What was done
- Implemented `eval/calibration.py`:
  - `TemperatureScaler`: post-hoc probability calibrator fitting scalar parameter $T > 0$ on the **validation period exclusively** via bounded NLL minimization.
  - `compute_calibration_metrics`: computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Negative Log-Likelihood (NLL), Brier score, and bin-by-bin reliability statistics across confidence deciles.
- Implemented `eval/abstention.py`:
  - `compute_expected_value`: computes direction-specific expected values:
    $EV_{long} = P(up) \cdot G_{up} - P(down) \cdot L_{down} - \text{costs}$
    $EV_{short} = P(down) \cdot G_{down} - P(up) \cdot L_{up} - \text{costs}$
  - `apply_abstention_rule`: executes positions ONLY if $EV \ge \tau_{ev}$ AND confidence $P \ge \tau_{conf}$; otherwise abstains ($0$, NO TRADE).
  - `tune_abstention_thresholds`: grid-searches optimal $(\tau_{ev}, \tau_{conf})$ hurdles strictly on validation splits with zero test set leakage.
  - `compute_risk_coverage_curve`: generates risk-coverage diagnostics evaluating selective accuracy, coverage fraction, and selective returns across confidence thresholds.
  - `compare_with_and_without_abstention`: event-driven backtest comparison answering research question RQ3.
- Added comprehensive unit tests in `tests/test_calibration_abstention.py`.

### What was tested
- All **106 tests** passing in `pytest` (100% pass rate).
- **Mandatory Calibration Check**: Verified that Expected Calibration Error (ECE) substantially decreases on held-out test splits after temperature scaling.
- **Decision Rule Integrity**: Verified that weak or uncertain predictions are safely filtered into abstention ($0$).
- **RQ3 Backtest Comparison**: Confirmed that abstention reduces unprofitable low-conviction churn, cuts trade frequency, and improves risk-adjusted returns after costs.
- `ruff check .` passes with 0 errors.

### Known issues
- None.

### Next step
- Phase 8: Explainability

---

## Phase 8: Explainability ✅

**Date**: 2026-10-15
**Status**: Complete

### What was done
- Implemented `explain/shap_utils.py`:
  - `ShapExplanation`: container standardizing attribution tensors across models.
  - `compute_tabular_shap`: computes local and global SHAP attributions using `shap.TreeExplainer` for XGBoost and `shap.Explainer` for Logistic Regression pipelines.
  - `compute_tabular_branch_shap`: computes attributions for the MLP tabular branch of `DualBranchFusion` (Model D) with respect to engineered order-flow and wall trust features.
  - `get_top_features_for_sample`: ranks top contributing features for individual prediction instances and target classes (`[-1, 0, 1]`).
  - `compute_feature_importance_summary`: produces global mean-absolute-SHAP rankings across features.
- Implemented `explain/gradcam.py`:
  - `GradCAM`: hook-based gradient-weighted class activation mapping for 2D convolutional layers (`conv3` in `HeatmapCNN` and `spatial_branch.conv2` in `DualBranchFusion`).
  - Upsamples rectified feature maps to input tensor dimensions $(P, T)$ via bilinear interpolation, with per-sample min-max normalization in $[0, 1]$.
  - `identify_salient_price_bands`: collapses the temporal axis to extract a 1D price saliency profile, identifying which relative distance-to-mid ticks most heavily influenced logits.
- Implemented `explain/templates.py`:
  - `DecisionDiagnostic`: structured payload providing transparent model diagnostics.
  - `format_wall_explanation`: plain-language breakdown of wall persistence, fill ratio, flicker events, and trust scores.
  - `format_order_flow_explanation`: plain-language assessment of Order Flow Imbalance and depth accumulation.
  - `build_decision_diagnostic`: synthesizes model action, calibrated confidence, net EV, top SHAP drivers, and wall context.
  - Strict non-causal safeguards: strictly enforces diagnostic framing; prohibits forbidden words (`spoofing`, `manipulation`, `guaranteed profit`, `rigged`); attaches standardized disclaimer `STANDARD_DISCLAIMER`.
- Added comprehensive unit tests in `tests/test_explain.py`.

### What was tested
- All **112 tests** passing in `pytest` (100% pass rate).
- **Tabular SHAP Attribution**: Verified correct shape `(N, n_features, 3)` for XGBoost and Logistic Regression, correct top feature extraction, and sorted global importance.
- **Deep Tabular Branch SHAP**: Verified attribution generation on neural network tabular branches.
- **Grad-CAM Saliency Maps**: Verified shape `(P, T)` and `(B, P, T)`, normalized bounds $[0, 1]$, and price-band saliency extraction on both `HeatmapCNN` and `DualBranchFusion`.
- **Formatting and Safety Checks**: Verified absence of prohibited terms across all generated narrative outputs and presence of diagnostic disclaimers.
- `ruff check .` passes with 0 errors.

### Known issues
- None.

### Next step
- Phase 9: Replay Dashboard

---

## Phase 9: Replay Dashboard ✅

**Date**: 2026-10-15
**Status**: Complete

### What was done
- Implemented `dashboard/sample_data.py`:
  - Standalone deterministic generator producing 4 realistic demo scenarios saved in `data/sample_clips/` (Parquet, zstd compression):
    1. `bait_wall.parquet`: Fleeting ask wall (trust ~22/100) flickering and pulling before price reach; model discounts resistance and avoids short traps.
    2. `real_wall.parquet`: Authentic resting bid wall (trust ~88/100) absorbing 22 BTC of aggressive sales, sparking an upward rebound and confident long signal.
    3. `no_trade_abstention.parquet`: Choppy range-bound noise where net EV fails to clear the transaction fee hurdle, forcing explicit `NO TRADE (ABSTAIN)`.
    4. `failed_signal.parquet`: High-trust resting wall overwhelmed by an unexpected institutional market dump, demonstrating the probabilistic nature of diagnostics.
  - Automatic on-demand clip generation when missing.
- Implemented `dashboard/app.py`:
  - Streamlit application (`streamlit run dashboard/app.py`):
    - **Header & Metrics**: Replay scenario selector, live BTC/USDT mid price, net expected value metric, decision status badge (`TRADE LONG`, `TRADE SHORT`, or `NO TRADE (ABSTAIN)`), and calibrated probability stacked bar (`DOWN`, `NEUTRAL`, `UP`).
    - **Interactive Controls**: Play/pause toggle, variable playback speeds (1x, 5x, 20x), and interactive time scrubber slider.
    - **Perspective Modes**: Toggle between **Liquidity Trust View** (color-coded trust badges), **Plain Heatmap** (raw depth without trust overlay), and **Side-by-Side Comparison** (simultaneous synchronized dual columns).
    - **Scrolling Heatmap**: Plotly 2D order book depth heatmap with mid-price trajectory line and annotated active wall badges.
    - **Diagnostic 'Why?' Panel**: Human-readable narrative generated by `explain/templates.py`, horizontal SHAP feature attribution bar chart, active wall metrics table, and standardized system disclaimer (`STANDARD_DISCLAIMER`).
- Added comprehensive unit tests in `tests/test_dashboard.py`.

### What was tested
- All **115 tests** passing in `pytest` (100% pass rate).
- **Sample Clip Generation**: Verified schema validity, column completeness, strictly monotonic timestamps, and valid probability simplex ($\sum p_i = 1$) across all 4 scenarios.
- **Scenario Diagnostics**: Verified wall trust divergence between fleeting bait walls (<35) and authentic resting walls (>80), and verified strict abstention enforcement in the choppy scenario.
- **Plotly Figure Construction**: Verified heatmap, probability bar, and SHAP bar chart generation with valid data traces and dark theme styling.
- `ruff check .` passes with 0 errors.

### Known issues
- None.

### Next step
- Phase 10: Report and Packaging

---

## Phase 10: Report and Packaging ✅

**Date**: 2026-10-15
**Status**: Complete (All Project Phases 0 through 10 Complete)

### What was done
- Implemented `eval/report.py`:
  - `generate_ablation_benchmark_table`: compiles comparative metrics across Models A, B, C, D and pooling ablations (`band_gru`, `gap_time`, `full_gap`) addressing RQ1.
  - `generate_wall_trust_decile_report`: compiles 10-decile wall persistence validation report with 95% Wilson confidence intervals and Spearman rank correlation addressing RQ2.
  - `generate_calibration_report`: calculates Expected Calibration Error (ECE), Maximum Calibration Error (MCE), NLL, and Brier score before and after temperature scaling.
  - `generate_abstention_comparison_report`: compares backtest trading performance with vs. without abstention across VIP0 maker and taker fee regimes addressing RQ3.
  - `export_report_figures`: writes interactive Plotly HTML figures (`reliability_diagram.html`, `trust_deciles.html`) and CSV benchmarks to `docs/report/`.
  - `compile_full_research_report`: generates self-contained markdown evaluation report `docs/report/research_report.md`.
- Crafted comprehensive reproduction `README.md`:
  - System architecture diagram and pipeline walkthrough across all 10 phases.
  - Detailed answers to RQ1, RQ2, and RQ3 with full benchmark tables.
  - One-command quickstart reproduction commands for setup, tests, dashboard replay, live recording, and report compilation.
  - Rigorous non-causal compliance disclaimers and empirical limitations.
- Added unit tests in `tests/test_report.py`.

### What was tested
- All **120 tests** passing in `pytest` (100% pass rate).
- **Benchmark Compilation**: Verified ablation table extraction, decile persistence monotonicity ($r > 0.85$), and ECE improvement under temperature scaling.
- **Figure and Report Exports**: Verified that `research_report.md`, `reliability_diagram.html`, `trust_deciles.html`, and `ablation_benchmark.csv` are generated cleanly.
- `ruff check .` passes with 0 errors across the entire codebase.

### Known issues
- None.

### Project State
- **All phases (Phase 0 to Phase 10) are 100% complete, tested, and validated.**



