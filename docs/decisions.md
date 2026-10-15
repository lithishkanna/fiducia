# Design Decisions Log

This document records design choices, trade-offs, and open questions encountered
during development of the Liquidity Trust research system.

---

## Decision 001: Tech Stack Choices (Phase 0)

**Date**: 2026-10-07
**Status**: Accepted

- **Python 3.11+** — required for performance improvements and typing features.
- **Polars** as primary DataFrame library (faster for our column-oriented workloads).
- **PyArrow Parquet** with zstd compression for raw storage.
- **XGBoost + LightGBM** both included since we want to compare; XGBoost is default for baselines.
- **PyTorch Lightning** for deep learning training loop boilerplate reduction.

## Decision 002: Fee Assumptions (Phase 0)

**Date**: 2026-10-07
**Status**: Accepted

- **Maker fee**: 2.0 bps (Binance Futures VIP0 default, post-BNB discount)
- **Taker fee**: 5.0 bps (Binance Futures VIP0 default)
- **Slippage**: 1.0 bps conservative estimate
- These are configurable in `configs/labels.yaml` and `configs/eval.yaml`.
- Multiple fee scenarios will be tested in backtests.

## Decision 003: Binance Futures WebSocket & Sequence Validation Protocol (Phase 1)

**Date**: 2026-10-11
**Status**: Accepted

- **Streams**: Combined stream `btcusdt@depth@100ms` and `btcusdt@aggTrade` via Binance USD-M Futures WebSocket (`wss://fstream.binance.com/stream`).
- **REST Snapshot**: Requested via `https://fapi.binance.com/fapi/v1/depth?symbol=BTCUSDT&limit=1000`.
- **Sync Protocol**:
  1. Buffer incoming diff events while waiting for REST snapshot.
  2. Discard buffered events with `u < lastUpdateId`.
  3. First applied event satisfies `U <= lastUpdateId <= u`.
  4. Subsequent events enforce `pu == u_prev`.
  5. Any violation triggers immediate gap flag marker, invalidates local state, and forces fresh snapshot re-sync.
- **Storage**: Parquet with hourly rollover and date partitioning (`data/raw/YYYY-MM-DD/raw_events_HH.parquet`), compressed with zstd.

## Decision 004: Fill vs Cancel Heuristic Attribution (Phase 2)

**Date**: 2026-10-11
**Status**: Accepted

- Public WebSocket feeds do not provide user order IDs. An order book reduction is therefore attributed to:
  - **Fill**: if an `aggTrade` occurs on the corresponding side (`m=True` seller maker -> bid match, `m=False` buyer maker -> ask match) within a configurable time window (default 100ms) at the exact price.
  - **Cancel**: if no matching trade occurred within the time window.
- This is explicitly documented and tracked as an *inferred heuristic*, not ground-truth exchange execution.
- Gap states are propagated to all derived events so downstream models and backtests can cleanly exclude gap intervals.

## Decision 005: Feature Engineering, Tensor Layout, and Triple-Barrier Labeling (Phase 3)

**Date**: 2026-10-11
**Status**: Accepted

- **Resampling Grid**: Fixed 500 ms grid. A grid step is marked invalid (`is_gap = True`) if any raw event or order book state within the bin had a gap flag.
- **Tabular Features**: All rolling features (OFI, imbalance, spread, returns, volatility, signed trade volume, trade intensity, add/cancel rates, liquidity center of mass and velocity) are strictly backward-looking with window sizes `[1s, 5s, 10s, 30s, 60s]`.
- **Heatmap Tensor**: Shape `(4, P, T)` with $P=100$ relative price ticks centered around mid-price and $T=60$ time steps (30s history at 500ms grid). Channels: `[bid_size, ask_size, executed_buy_vol, executed_sell_vol]`. Scaled via `log1p`. Normalization statistics (mean, std) must be computed exclusively on training periods via `TensorNormalizer`.
- **Wall Detection**: Level size $\ge 5.0 \times$ local median depth over 5-minute rolling window. Tracks wall age, fill/cancel ratios, flicker count, and approach behavior.
- **Triple-Barrier Labeling**: Barrier threshold $\Delta = k \cdot (\text{half\_spread} + \text{fee\_bps} \cdot M_0)$. Upper barrier $M_0 + \Delta$, lower barrier $M_0 - \Delta$, vertical horizon $\le 15\text{s}$. Outcomes: $+1$ (UP), $-1$ (DOWN), $0$ (NEUTRAL), with recorded `time_to_hit_ms`. Samples overlapping gap intervals are marked invalid.

## Decision 006: Walk-Forward Splits, Baseline Model Architecture, and Cost-Aware Backtesting (Phase 4)

**Date**: 2026-10-11
**Status**: Accepted

- **Splits**: Time-ordered walk-forward cross-validation with an explicit purge gap $\ge 30\text{s}$ (the max label horizon). Train and validation sets never share overlapping labels. Final 20% of data is reserved as a strictly untouched test holdout.
- **Model A**:
  - Logistic Regression with standard scaling and balanced class weights.
  - XGBoost multiclass classifier (`objective='multi:softprob'`) on engineered tabular features.
  - Standardized label encoding: $-1 \to 0$ (DOWN), $0 \to 1$ (NEUTRAL), $+1 \to 2$ (UP).
- **Backtesting & Costs**:
  - Event-driven execution simulating position transitions on prediction signals.
  - Configurable fee models (Maker: 2.0 bps, Taker: 5.0 bps) plus 1.0 bps adverse slippage penalty per leg.
  - Zero trading allowed while stream has active gap flags.
  - Fixed position limits (default 1.0 BTC unit).
- **Statistical Significance**: Moving block bootstrap (default block length 300s, 1000 resamples) for 95% confidence intervals on Sharpe ratio, Net PnL, and Macro F1 score.

## Decision 007: Wall Trust Label Definition, Calibration, and Decile Evaluation (Phase 5)

**Date**: 2026-10-12
**Status**: Accepted

- **Label Definition**:
  For every large resting liquidity wall detected at $(price, side)$ at time $t_0$:
  We observe the subsequent outcome when price approaches within $N=10$ ticks (proximity window) or up to max horizon $T_{max} = 60\text{s}$:
  - **Label = 1 (Trustworthy / Real)**: When price approaches within $N$ ticks, the wall remains active (resting size $\ge 50\%$ of peak size) or is traded into (cumulative executed fills $\ge 20\%$ of wall size).
  - **Label = 0 (Low-Trust / Fleeting)**: The wall is pulled/cancelled before price reaches it (or cancelled immediately as price comes within $N$ ticks with $< 20\%$ fills).
- **Features Used**: strictly historical features known at or before evaluation time $t$: `[relative_size, distance_ticks, distance_bps, age_ms, flicker_count, cum_fill_qty, cum_cancel_qty, fill_ratio, approach_count, side, book_imbalance]`.
- **Model & Score Mapping**:
  - Gradient Boosted Classifier (XGBoost / HistGradientBoosting) trained on wall features.
  - Probability calibration via Platt scaling / sigmoid calibration fitted on validation splits.
  - Trust Score = $P(\text{standing}) \times 100 \in [0, 100]$.
- **Decile Analysis**:
  - Walls partitioned into deciles by trust score.
  - Empirical survival/fill rate computed per decile with 95% binomial/bootstrap confidence intervals.
  - Spearman rank correlation to evaluate monotonic reliability (answering RQ2).

## Decision 008: Deep Model Architectures and RQ1 Ablation Design (Phase 6)

**Date**: 2026-10-12
**Status**: Accepted

- **Dataset Protocol**:
  - Memory-mapped / array-backed PyTorch Datasets building tensors in RAM on the fly. No image files persisted.
  - Standardized batch format: `(X, y)` where $y \in \{0, 1, 2\}$ mapping to $\{DOWN, NEUTRAL, UP\}$.
- **Model B (DeepLOB)**:
  - Adapted from Zhang et al. for top-10 levels: input shape `(batch, 1, 40, T)`.
  - Feature extraction convolutions + inception module + LSTM capturing temporal dependencies across $T$ steps.
- **Model C (Heatmap CNN)**:
  - Input tensor $(4, P, T)$ with $P=100$ price bins and $T=60$ time steps.
  - 2D convolutional blocks with BatchNorm, LeakyReLU, and Dropout.
- **Model D (Dual-Branch Fusion)**:
  - Spatial branch: 2D CNN on heatmap tensor $(4, P, T)$.
    - Default pooling: `band_gru` — adaptive average pooling along price axis to $B=8$ bands, preserving distance-to-mid structure across time, followed by a GRU over $T$ steps to embedding $d=64$.
    - Ablation variant 1: `gap_time` — pooling over time axis, followed by linear projection of price bands.
    - Ablation variant 2: `full_gap` — global average pooling over both price and time axes.
  - Tabular branch: MLP over engineered order flow and wall trust features projected to embedding $d=64$.
  - Fusion: Concatenation $(2d = 128) \to \text{Dropout} \to \text{Linear} \to \text{ReLU} \to \text{Logits}(3)$.
- **Loss & Optimization**:
  - Cross-Entropy with class weighting or Focal Loss.
  - AdamW optimizer with cosine annealing / linear warmup and early stopping.

## Decision 009: Temperature Scaling Calibration and Expected-Value Abstention (Phase 7)

**Date**: 2026-10-12
**Status**: Accepted

- **Temperature Scaling**:
  - Post-processing probability calibration optimizing scalar temperature $T > 0$ on the **validation set exclusively**.
  - Evaluated via Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) across 10 confidence bins before and after scaling.
- **Expected-Value (EV) Abstention Rule**:
  - Action taken only if positive expected value exceeds hurdle after fees and slippage:
    $EV_{long} = P(up) \cdot G_{up} - P(down) \cdot L_{down} - \text{costs} \ge \tau_{ev}$
    $EV_{short} = P(down) \cdot G_{down} - P(up) \cdot L_{up} - \text{costs} \ge \tau_{ev}$
  - AND confidence $P(action) \ge \tau_{conf}$.
  - Otherwise, the model abstains ($0$, NO TRADE).
- **Zero Test-Leakage Threshold Selection**:
  - Thresholds $\tau_{ev}$ and $\tau_{conf}$ are tuned via grid-search on the validation period only.
  - The held-out test period is evaluated strictly **once** with the chosen parameters to answer RQ3 (trading performance with vs. without abstention across fee and volatility regimes).

## Decision 010: Explainability Architecture & Diagnostic Framing (Phase 8)

**Date**: 2026-10-15
**Status**: Accepted

- **SHAP Feature Attribution**:
  - Tabular models (Model A): Fast TreeExplainer for XGBoost and Kernel/Permutation Explainer for Logistic Regression pipelines. Standardizes outputs into `ShapExplanation` with shape `(N, n_features, n_classes)` matching triple-barrier classes `[-1, 0, 1]`.
  - Tabular branch of DualBranchFusion (Model D): Explains MLP branch feature sensitivities with respect to engineered order flow and wall trust scores.
- **Grad-CAM Saliency Maps**:
  - Target layers: Captures spatial price-time feature activations and gradients at the final convolutional layers (`conv3` in Model C HeatmapCNN, `spatial_branch.conv2` in Model D DualBranchFusion).
  - Normalization: Positive gradient activations are rectified (`ReLU`), upsampled via bilinear interpolation to original input spatial dimensions $(P, T)$, and min-max normalized per sample to $[0, 1]$.
  - Price-band analysis: Projects 2D CAM along the price axis to identify which relative price offsets to mid most actively drive classification logits.
- **Diagnostic Framing & Non-Causal Safeguards**:
  - All outputs in `explain/templates.py`, reports, and dashboard panels are strictly framed as **internal model diagnostics**, not causal explanations of price action or market intent.
  - Prohibited terminology: Forbidden words ("spoofing", "manipulation", "guaranteed profit", "sure thing", "rigged") are strictly banned. Instead, neutral behavioral terms are used ("unusual cancellation behavior", "low persistence probability", "transient resting depth", "order flow pressure").
  - Every diagnostic payload carries the standardized system disclaimer (`STANDARD_DISCLAIMER`).

## Decision 011: Replay Dashboard Architecture & Demo Scenarios (Phase 9)

**Date**: 2026-10-15
**Status**: Accepted

- **Interactive Architecture**:
  - Built with **Streamlit** and **Plotly** to provide real-time, interactive order-book replay without external frontend build dependencies.
  - Controls include play/pause toggle, variable playback speeds (1x, 5x, 20x), and a bidirectional time scrubber across timestamps.
- **Side-by-Side & Perspective Modes**:
  - **Liquidity Trust View**: Walls are annotated with real-time trust score badges [0-100] color-coded by behavioral tier (Green for authentic resting depth $>65$, Amber for moderate $35-65$, Red for fleeting/bait depth $<35$).
  - **Plain Heatmap**: Displays raw order book sizes without trust overlays, highlighting how conventional visual traders get trapped by bait depth.
  - **Side-by-Side Mode**: Renders both perspectives in synchronized columns to demonstrate the immediate cognitive value of the trust model.
- **Pre-Selected Standalone Demo Scenarios**:
  - 4 realistic, self-contained Parquet clips in `data/sample_clips/` allowing instantaneous evaluation from a clean checkout without recording requirements:
    1. `bait_wall`: Fleeting ask wall (trust 22/100) that flickers and cancels upon approach; model discounts resistance and avoids short trap.
    2. `real_wall`: Authentic resting bid wall (trust 88/100) absorbing aggressive market selling, prompting a high-confidence long signal.
    3. `no_trade_abstention`: Range-bound choppy noise where net expected value fails to clear transaction costs, triggering explicit abstention.
    4. `failed_signal`: Authentic-looking wall overwhelmed by an unexpected institutional market dump, reinforcing the non-guarantee nature of probabilistic diagnostics.
- **Decision Support & Explainability Integration**:
  - Live Decision Banner displaying `TRADE LONG`, `TRADE SHORT`, or `NO TRADE (ABSTAIN)`.
  - Calibrated probability bar (`DOWN`, `NEUTRAL`, `UP`) and Net EV vs. hurdle gauge.
  - Diagnostic 'Why?' panel displaying plain-language narratives from `explain/templates.py`, horizontal SHAP sensitivity bars, active wall metrics table, and the standardized system disclaimer.

## Decision 012: Research Reporting, Benchmarking & Packaging (Phase 10)

**Date**: 2026-10-15
**Status**: Accepted

- **Reproduction-First Packaging**:
  - The repository is fully self-contained and reproducible. Any collaborator can check out the repository, run `pip install -e .[dev]`, run tests, compile the research report, and launch the replay dashboard without external dependencies or live network connections.
- **Reporting Architecture**:
  - Automated report compiler `eval/report.py` synthesizes:
    - RQ1 architecture & pooling benchmark comparison (Models A, B, C, D with pooling ablations).
    - RQ2 wall trust decile persistence analysis (Wilson 95% CIs and Spearman rank correlation).
    - RQ3 temperature calibration metrics and cost-aware trading comparisons across maker/taker scenarios.
  - Exports interactive Plotly HTML figures (`reliability_diagram.html`, `trust_deciles.html`) and CSV summaries to `docs/report/`.
- **Comprehensive Limitations Acknowledgment**:
  - Inferred fill vs. cancel heuristic without private order IDs.
  - Scope limited to Binance BTC/USDT perpetual; parameters require recalibration for other tokens.
  - Realistic adverse-selection slippage (1.0 bps) modeled, but queue priority remains an empirical limitation.
  - Explanations are model behavior diagnostics, not causal explanations of market movements or manipulation claims.



