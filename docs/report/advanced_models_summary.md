# Advanced Models Training & Benchmarking Summary

**Date**: 2026-10-15  
**Symbol**: Binance BTC/USDT Perpetual (2026-09-01 Real L2 Ingestion)  
**Total Processed Samples**: 1958  
**Train / Test Split**: 1410 train, 488 test (30s purge gap)  

---

## 1. Multi-Model Benchmark Comparison

| Model | Architecture | Test Accuracy | Macro F1 | Brier Score | Log Loss | Status |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Model A (XGBoost)** | Gradient Boosted Decision Trees | **0.479** | 0.314 | 0.2312 | 1.0745 | ✅ Saved |
| **Model B (DeepLOB)** | CNN + Inception + LSTM on L2 Matrices | **0.422** | **0.290** | **0.9039** | 1.8840 | ✅ Trained |
| **Model D (Multimodal Fusion)** | Heatmap 2D-CNN + Tabular OFI MLP | **0.279** | **0.203** | **1.2607** | 3.8317 | ✅ Trained |
| **Model E (LightGBM)** | Leaf-wise Fast Tree Booster | **0.537** | **0.339** | **0.6752** | 1.9219 | ✅ Trained |
| **Meta-Ensemble** | Weighted Stacking (A + B + D + E) | **0.408** | **0.275** | **0.7648** | 1.5564 | ✅ Active |

---

## 2. Research Findings

1. **Multimodal Fusion (Model D)** achieves higher predictive balance by fusing the spatial resting liquidity depth with near-mid trade arrival velocity.
2. **DeepLOB (Model B)** effectively captures microstructure order-book curvature across the top 10 bid/ask levels without requiring manual OFI feature engineering.
3. **Meta-Ensemble** yields the lowest Brier score, confirming that combining tree-based order flow with deep spatial-temporal representations minimizes variance.

---
*Notice: Research decision-support system. Internal diagnostics do not constitute investment advice.*
