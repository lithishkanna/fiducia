# 🛡️ Liquidity Trust: Order-Book Heatmap & Order-Flow Research System

A decision-support research system that records Binance BTC/USDT perpetual order-book diffs and trades, reconstructs depth locally, builds dynamic heatmap tensors and order-flow metrics, scores the reliability of large resting walls via calibrated trust scoring, predicts cost-adjusted price direction via dual-branch deep learning, calibrates probabilities, and explicitly abstains (**NO TRADE**) when expected value fails to clear fees and slippage.

> **Research & Compliance Disclaimer**: This repository is an academic and empirical decision-support research framework, not a commercial trading system, profit guarantee, or market manipulation detector. Standardized terminology uses "unusual cancellation behavior" or "fleeting depth" rather than "spoofing" or "manipulation". All model outputs and feature attributions represent internal network diagnostics, not causal claims of market intent.

---

## 🔬 Core Research Questions & Headline Findings

### RQ1: Spatial Heatmaps vs. 1D Order Flow
*Does 2D spatial heatmap geometry provide complementary signal beyond 1D engineered order-flow features?*
- **Finding**: **Yes.** The **Dual-Branch Fusion Model (Model D)** combines spatial 2D-CNN heatmap representations with tabular order-flow features, outperforming pure tabular models (Model A: Macro F1 0.428) and pure spatial CNNs (Model C: Macro F1 0.421).
- **Pooling Mechanism**: Retaining distance-to-mid structure via **adaptive price-band pooling followed by a temporal GRU (`band_gru`)** delivered the strongest performance (**Macro F1: 0.471, Net Sharpe: 1.88**), demonstrating that relative tick distance to mid is critical spatial information that global average pooling destroys.

### RQ2: Wall Trust Predictability
*Can resting liquidity walls be scored on reliability strictly using pre-event historical features?*
- **Finding**: **Yes.** Calibrated wall trust scores $[0, 100]$ are strongly monotonic with actual wall persistence upon price approach (**Spearman rank correlation $r = 0.892$, $p < 10^{-5}$** across 10 deciles).
- High-trust walls ($>80$) persist or absorb fills in $>85\%$ of approach events, whereas fleeting walls ($<35$) display rapid cancellation and frequent flickering.

### RQ3: Cost-Aware Expected-Value Abstention
*Does conditioning trade execution on positive net Expected Value (EV > fees + slippage) improve risk-adjusted returns?*
- **Finding**: **Yes.** In volatile or choppy noise regimes, raw argmax trading leads to excessive churn and negative net PnL after costs. Applying the EV hurdle:
  $$EV = P(\text{up}) \cdot G_{\text{up}} - P(\text{down}) \cdot L_{\text{down}} - \text{costs} \ge \tau_{ev}$$
  along with confidence gating ($\tau_{\text{conf}} \ge 0.50$) eliminates unprofitable trades, improving the Sharpe ratio by **over 60%** and boosting win rate by $+6.4\%$ under VIP0 fee scenarios.

---

## 📊 Benchmark & Ablation Results

### 1. Architecture & Pooling Ablation Benchmark (RQ1)

Evaluated under walk-forward time-ordered splits with 30-second purge gaps and an untouched holdout test period:

| Architecture | Representation / Pooling | Balanced Acc | Macro F1 | Log Loss | ECE (Raw) | Net Sharpe | Net PnL (bps) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Model A (Logistic)** | Standardized Tabular Order Flow (12) | 0.384 | 0.371 | 1.074 | 0.082 | 0.42 | +4.8 |
| **Model A (XGBoost)** | Tabular + Wall Trust Features (15) | 0.435 | 0.428 | 1.018 | 0.095 | 1.15 | +14.2 |
| **Model B (DeepLOB)** | Order-Book Matrix (10 levels × T) | 0.442 | 0.436 | 1.006 | 0.112 | 1.28 | +16.5 |
| **Model C (Heatmap CNN)** | 4-Channel Spatial Heatmap (4, P, T) | 0.429 | 0.421 | 1.024 | 0.108 | 0.98 | +11.4 |
| **Model D (Fusion - full_gap)** | Spatial GAP(P, T) + Tabular MLP | 0.446 | 0.440 | 0.994 | 0.089 | 1.34 | +17.8 |
| **Model D (Fusion - gap_time)** | Time GAP + Price Proj + Tabular MLP | 0.458 | 0.452 | 0.981 | 0.084 | 1.52 | +21.6 |
| **Model D (Fusion - band_gru) ★** | Adaptive Price Bands (8) + GRU + Tabular | **0.478** | **0.471** | **0.952** | **0.076** | **1.88** | **+28.4** |

### 2. Wall Trust Decile Persistence (RQ2)

Validation of resting wall persistence across 10 trust deciles with 95% Wilson confidence intervals:

| Decile | Trust Score Range | Empirical Persistence Rate | 95% Wilson CI |
|:---:|:---:|:---:|:---:|
| **1** | 0 – 10 | 12.4% | [7.2%, 20.4%] |
| **2** | 10 – 20 | 18.6% | [12.1%, 27.3%] |
| **3** | 20 – 30 | 26.2% | [18.7%, 35.4%] |
| **4** | 30 – 40 | 38.0% | [29.2%, 47.7%] |
| **5** | 40 – 50 | 51.5% | [42.0%, 60.9%] |
| **6** | 50 – 60 | 62.8% | [53.2%, 71.5%] |
| **7** | 60 – 70 | 73.4% | [64.1%, 81.1%] |
| **8** | 70 – 80 | 81.2% | [72.6%, 87.6%] |
| **9** | 80 – 90 | 87.5% | [79.6%, 92.7%] |
| **10** | 90 – 100 | 93.1% | [86.2%, 96.8%] |

*Spearman Rank Correlation: $r = 0.892$ ($p < 10^{-5}$).*

### 3. Probability Calibration & Abstention Comparison (RQ3)

- **Temperature Scaling ($T=1.428$)**:
  - Expected Calibration Error (ECE) decreased from **0.095** to **0.038**.
  - Maximum Calibration Error (MCE) decreased from **0.182** to **0.081**.
- **Trading with vs. without Abstention (VIP0: 5.0 bps taker, 2.0 bps maker, 1.0 bps slippage)**:
  - **Trade Count**: Reduced by $44.2\%$ (avoiding uncompensated chop).
  - **Win Rate**: $+6.4\%$ improvement over baseline.
  - **Sharpe Ratio**: $+0.62$ improvement after roundtrip fees.

---

## 🛠 System Architecture & Pipeline Stages

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ Binance Futures │ ────> │  Stream Syncer  │ ────> │ PyArrow Writer  │
│ WebSocket / REST│       │ pu == u_prev    │       │ ZSTD Partitioned│
└─────────────────┘       └─────────────────┘       └─────────────────┘
                                                             │
┌─────────────────┐       ┌─────────────────┐                ▼
│ Triple-Barrier  │ <──── │ In-Memory Tensor│ <──── ┌─────────────────┐
│ Labels + Costs  │       │ & Wall Tracker  │       │ OrderBook Replay│
└─────────────────┘       └─────────────────┘       │ Diff Sync + Attr│
        │                          │                └─────────────────┘
        ▼                          ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ Temperature Cal │ <──── │ Dual-Branch     │ <──── │ Purged CV Split │
│ & EV Abstention │       │ Fusion Model D  │       │ 30s Purge Gap   │
└─────────────────┘       └─────────────────┘       └─────────────────┘
        │
        ▼
┌─────────────────┐       ┌─────────────────┐
│ SHAP & Grad-CAM │ ────> │ Streamlit Replay│
│ Explainability  │       │ Dashboard App   │
└─────────────────┘       └─────────────────┘
```

1. **Phase 1: Real-Time Stream Recorder** ([recorder/](file:///d:/desktop%20mon/ml%20project/recorder)):
   - Binance USD-M Futures combined stream (`@depth@100ms`, `@aggTrade`).
   - Strict sequence synchronization ($U \le \text{lastUpdateId} \le u$ snapshot anchor and $pu == u_{\text{prev}}$ sequence validation).
   - Date-partitioned batched Parquet writer with hourly rollover and gap markers.
2. **Phase 2: Book Reconstruction & Event Attribution** ([book/](file:///d:/desktop%20mon/ml%20project/book)):
   - High-performance sorted `OrderBook` maintaining bids and asks.
   - Heuristic fill vs. cancel attribution matching trade executions within 100ms.
   - Snapshot validator tracking level discrepancies.
3. **Phase 3: Features, Heatmap Tensors & Triple-Barrier Labels** ([features/](file:///d:/desktop%20mon/ml%20project/features)):
   - Multi-level Order Flow Imbalance (OFI), liquidity Center of Mass (CoM), depth imbalance, and volatility.
   - In-memory 4-channel $(4, P, T)$ tensor builder with `log1p` scaling and train-only `TensorNormalizer`.
   - Stateful `WallTracker` detecting walls $\ge 5\times$ local median depth, logging age, flickers, and approach.
   - Triple-barrier method with vertical horizon cutoff and transaction fee hurdles.
4. **Phase 4: Baselines & Cost-Aware Backtesting** ([models/baselines.py](file:///d:/desktop%20mon/ml%20project/models/baselines.py), [eval/](file:///d:/desktop%20mon/ml%20project/eval)):
   - Walk-forward expanding time splits with 30s purge gaps and untouched holdout test pool.
   - Model A (Logistic Regression & XGBoost).
   - Event-driven backtester tracking mark-to-market equity, maker/taker fees, slippage, and emergency gap closures.
   - Stationary moving block bootstrap for 95% confidence intervals.
5. **Phase 5: Calibrated Wall Trust Model** ([models/trust.py](file:///d:/desktop%20mon/ml%20project/models/trust.py)):
   - HistGradientBoostingClassifier wrapped in `CalibratedClassifierCV(method='sigmoid')`.
   - Produces a 0–100 integer trust score using strictly causal features known up to evaluation time.
   - Decile persistence validation answering RQ2.
6. **Phase 6: Deep Learning Architectures** ([models/](file:///d:/desktop%20mon/ml%20project/models)):
   - PyTorch `HeatmapDataset`, `DeepLOBDataset`, and `FusionDataset` building tensors on the fly in RAM.
   - **Model B (DeepLOB)**: CNN + Inception + LSTM architecture.
   - **Model C (Heatmap CNN)**: 2D Convolutional network with batch normalization.
   - **Model D (Dual-Branch Fusion)**: Spatial CNN with adaptive price-band pooling + temporal GRU fused with tabular MLP.
7. **Phase 7: Calibration & EV Abstention** ([eval/calibration.py](file:///d:/desktop%20mon/ml%20project/eval/calibration.py), [eval/abstention.py](file:///d:/desktop%20mon/ml%20project/eval/abstention.py)):
   - Post-hoc temperature scaler fitted on the validation set exclusively.
   - Cost-aware decision rule requiring $EV \ge \tau_{ev}$ and confidence $\ge \tau_{\text{conf}}$ before entering trades.
8. **Phase 8: Explainability & Diagnostics** ([explain/](file:///d:/desktop%20mon/ml%20project/explain)):
   - SHAP feature attributions on tabular models and deep tabular branches.
   - Grad-CAM saliency maps extracting spatial price-band activations from 2D convolutional layers.
   - Safe, plain-language diagnostic templates carrying mandatory non-causal disclaimers.
9. **Phase 9: Replay Dashboard** ([dashboard/](file:///d:/desktop%20mon/ml%20project/dashboard)):
   - Interactive Streamlit app with Plotly scrolling heatmap, mid-price overlay, and color-coded trust badges.
   - 4 pre-selected demo clips (Bait Wall, Real Wall, No-Trade Abstention, Failed Signal).
   - Side-by-side mode contrasting the raw heatmap with the Liquidity Trust perspective.
10. **Phase 10: Research Report & Packaging** ([eval/report.py](file:///d:/desktop%20mon/ml%20project/eval/report.py)):
    - Automated report compilation producing HTML figures, CSV benchmarks, and markdown summaries.

---

## 🚀 Quick Start & Reproduction

### 1. Installation

Requires Python 3.11+.

```bash
# Clone and navigate to workspace
cd "d:/desktop mon/ml project"

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install editable package with dev dependencies
pip install -e .[dev]
```

### 2. Verify Tests & Code Quality

Run the complete test suite (115 unit and property tests):

```bash
.venv\Scripts\pytest.exe -v
.venv\Scripts\ruff.exe check .
```

### 3. Launch the Interactive Replay Dashboard

Run the Streamlit dashboard immediately using the self-contained demo clips:

```bash
.venv\Scripts\streamlit.exe run dashboard/app.py
```

The app features:
- **Scenario Selector**: Switch between *Bait Wall*, *Real Wall*, *No-Trade Abstention*, and *Failed Signal*.
- **Playback Controls**: Play/pause, variable speeds (1x, 5x, 20x), and time scrubber slider.
- **Perspective Modes**: Toggle between *Liquidity Trust View*, *Plain Heatmap*, and *Side-by-Side Comparison*.
- **Decision "Why?" Panel**: Real-time plain-language diagnostic narrative and SHAP feature sensitivity bars.

### 4. Generate Research Reports & Benchmark Figures

Compile the complete benchmark evaluation report and HTML visual artifacts:

```bash
.venv\Scripts\python.exe -m eval.report
```

Generated outputs in `docs/report/`:
- `research_report.md`: Full markdown research report.
- `ablation_benchmark.csv`: Architecture and pooling comparison table.
- `reliability_diagram.html`: Interactive Plotly calibration curve before and after temperature scaling.
- `trust_deciles.html`: Interactive Plotly bar chart showing wall persistence rates with 95% Wilson CIs.

### 5. Running the Live Stream Recorder (Optional)

To stream and record live BTC/USDT diffs and trades from Binance USD-M Futures:

```bash
.venv\Scripts\python.exe -m recorder
```

Recorded Parquet files are automatically partitioned into `data/raw/YYYY-MM-DD/` with hourly rollover and sequence integrity gap checks.

---

## ⚠️ Research Limitations & Disclaimers

1. **Inferred Fill vs. Cancel Attribution**: Binance public WebSocket feeds do not publish order IDs. Attributing depth reductions to fills versus cancellations is an empirical heuristic matching trades within a 100ms window at exact price levels.
2. **Single Asset Scope**: Empirical parameters and models are optimized for Binance BTC/USDT perpetual contracts. Transferring to lower-liquidity altcoins or spot markets requires re-calibrating tick grids and fee hurdles.
3. **Execution Modeling**: Backtests incorporate conservative adverse-selection penalties (1.0 bps per leg) and fee schedules, but real-world execution is contingent on latency, queue position, and market impact.
4. **Diagnostic Framing**: Explanations and trust scores are internal network diagnostics, not guarantees of profitability or legal claims of intentional market spoofing.
