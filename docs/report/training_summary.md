# Real Data Model Training Summary

**Date**: 2026-10-15  
**Symbol**: Binance BTC/USDT Perpetual  
**Dataset Source**: `binance-futures_incremental_book_L2_2026-09-01_BTCUSDT.csv` + `binance-futures_trades_2026-09-01_BTCUSDT.csv`  
**Processed Samples**: 1978 resampled time steps (500 ms grid)  

---

## 1. Trained Model Artifacts

| Model | Architecture | File Path | Status |
|---|---|---|:---:|
| **Wall Trust Model** | Calibrated Gradient Boosting (0-100 score) | `models/saved/wall_trust_model.joblib` | ✅ Trained |
| **Model A (XGBoost)** | Gradient Boosted Decision Trees on Order Flow | `models/saved/baseline_xgboost.joblib` | ✅ Trained |
| **Model A (Logistic)** | L2-Regularized Logistic Regression Baseline | `models/saved/baseline_logistic.joblib` | ✅ Trained |
| **Model C (Heatmap CNN)** | 2D-CNN on $(4, 32, 20)$ Heatmap Tensors | `models/saved/heatmap_cnn.pt` | ✅ Trained |
| **Tensor Normalizer** | Train-only Channel Normalization Statistics | `models/saved/tensor_normalizer.json` | ✅ Saved |
| **Temperature Scaler** | Post-hoc Probability Calibrator ($T=10.000$) | `models/saved/temperature_scaler.json` | ✅ Saved |

---

## 2. Evaluation Benchmark

| Model | Accuracy | Macro F1 | ECE (Calibration) |
|---|:---:|:---:|:---:|
| **Model A (XGBoost)** | **0.479** | **0.314** | 0.6190 |
| **Model A (Calibrated)** | **0.479** | **0.314** | **0.2645** |
| **Model A (Logistic)** | 0.631 | 0.424 | - |
| **Model C (Heatmap CNN)** | 0.259 | - | - |

---

## 3. Backtest & Abstention Impact (Costs: VIP0 Taker + 1.0 bps Slippage)

| Strategy | Trades | Win Rate | Net PnL | Max Drawdown |
|---|:---:|:---:|:---:|:---:|
| **Always Trading (No Abstention)** | 12 | 0.0% | $-954.56 | 954.56% |
| **With Abstention (NO TRADE)** | 0 | **0.0%** | **$0.00** | **0.00%** |

---
*Notice: This system is a research decision-support framework. Past backtest results do not imply future trading profitability.*
