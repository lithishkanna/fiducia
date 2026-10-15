# AGENT_BUILD_PLAN.md: Liquidity Trust

> Put this file in the repo root (or rename it `CLAUDE.md` / `AGENTS.md`) so the coding agent reads it first.
> Work **one phase at a time**. Do not start a phase until the previous phase's acceptance checks pass.

---

## 0. Project in one paragraph

Build a research system that records Binance BTC/USDT perpetual order-book diffs and trades, rebuilds the book locally, derives a heatmap tensor plus engineered order-flow features, scores how trustworthy large liquidity walls are, predicts whether price will make a **tradable** move (larger than spread + fees), calibrates confidence, and **abstains (NO TRADE)** when a setup is not worth acting on. The central research question: **does a heatmap-based model beat engineered order-flow features after costs?** A negative result is acceptable. This is decision support, not a profit system.

---

## 1. Ground rules for the agent

1. **Correctness over speed.** Most bugs in this project are silent (sequence off-by-one, label leakage, optimistic fills). Write tests for every module.
2. **No look-ahead.** Features at time `t` may use only data with timestamp `<= t`. Labels use the future, but train/test splits must be time-ordered with a purge gap.
3. **Never save heatmaps as images.** Build tensors in memory in the Dataset class. Only the dashboard renders them.
4. **Gap intervals are excluded** from every training, validation and test window.
5. **Everything configurable** via YAML in `configs/`. No magic numbers in code.
6. **Reproducible.** Fixed seeds, logged configs, deterministic splits, pinned dependencies.
7. **Verify external APIs.** Check the current Binance Futures API docs before coding stream and snapshot logic. Do not rely on memory for endpoint names or sync rules.
8. **Never claim profitability.** Reports and UI text say "decision support". Use "unusual cancellation behavior", never "spoofing" or "manipulation".
9. **Keep scope small.** One asset (BTCUSDT perpetual), one horizon family. Do **not** add Redis, TimescaleDB, Kafka, Kubernetes or multi-asset support unless asked.
10. **Ask before big decisions.** If a design choice is ambiguous (label definition, fee assumptions), write the options and a recommendation in `docs/decisions.md`, then proceed with the recommendation.
11. After finishing each phase, update `docs/progress.md` with what was done, what was tested, known issues and the next step.

---

## 2. Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3.11+ |
| Capture | `websockets` or `aiohttp` (asyncio), `requests`/`httpx` for REST snapshots |
| Storage | Parquet via PyArrow, partitioned by date; DuckDB for queries |
| Processing | Polars (primary), NumPy, Numba, Pandas where convenient |
| ML baseline | scikit-learn, XGBoost or LightGBM |
| Deep learning | PyTorch + PyTorch Lightning |
| Calibration | scikit-learn, custom temperature scaling, optional MAPIE |
| Explainability | SHAP, Captum or pytorch-grad-cam |
| Stats | SciPy, statsmodels |
| Tracking | MLflow or Weights & Biases, YAML configs (Hydra optional) |
| Dashboard | Streamlit + Plotly |
| Quality | pytest, ruff, mypy (light), pre-commit |

---

## 3. Repository layout

```
liquidity-trust/
├── AGENT_BUILD_PLAN.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── data.yaml          # symbol, paths, stream settings
│   ├── features.yaml      # windows, bins, wall thresholds
│   ├── labels.yaml        # barrier k, horizon, fees
│   ├── models.yaml        # hyperparameters per model
│   └── eval.yaml          # splits, purge gap, fee scenarios
├── recorder/
│   ├── stream.py          # websocket client, buffering
│   ├── sync.py            # snapshot + sequence logic
│   └── writer.py          # Parquet writer with rotation
├── book/
│   ├── orderbook.py       # local book data structure
│   ├── events.py          # add / reduce / fill / cancel inference
│   └── validate.py        # compare against fresh snapshots
├── features/
│   ├── tensor.py          # heatmap tensor builder
│   ├── tabular.py         # OFI, imbalance, spread, volatility...
│   ├── walls.py           # wall detection + persistence features
│   └── labels.py          # triple-barrier labels
├── models/
│   ├── baselines.py       # logistic, XGBoost
│   ├── deeplob.py
│   ├── heatmap_cnn.py
│   ├── fusion.py          # dual-branch model
│   ├── trust.py           # wall trust model
│   └── datasets.py        # PyTorch Dataset (on-the-fly tensors)
├── eval/
│   ├── splits.py          # walk-forward + purge gap
│   ├── calibration.py     # temperature scaling, reliability, ECE
│   ├── abstention.py      # EV rule, thresholds, risk-coverage
│   ├── backtest.py        # event-driven backtester with costs
│   └── bootstrap.py
├── explain/               # shap_utils.py, gradcam.py, templates.py
├── dashboard/app.py       # Streamlit replay app
├── tests/
├── docs/                  # decisions.md, progress.md, report notes
├── notebooks/
└── data/                  # gitignored: raw/, interim/, processed/
```

---

## 4. Phases

Each phase lists **tasks** and **acceptance checks**. Do not skip the checks.

### Phase 0: Setup (day 1)

**Tasks**
- Create the repo, `pyproject.toml`, virtual environment, pre-commit with ruff, and pytest config.
- Create the folder layout above with empty modules and docstrings.
- Add `configs/*.yaml` with sensible defaults and comments.
- Create `docs/decisions.md` and `docs/progress.md`.

**Acceptance**
- `pytest` runs (even with zero tests), `ruff check` passes, and the package imports.

---

### Phase 1: Recorder (start this first; data cannot be backfilled for free)

**Tasks**
- Look up current Binance USD-M futures docs for: diff depth stream (`@depth@100ms`), aggregated trade stream (`@aggTrade`), and REST depth snapshot.
- Implement `recorder/stream.py`: an asyncio client that subscribes to both streams, stamps each message with local receive time, and auto-reconnects with backoff.
- Implement `recorder/sync.py` with this protocol:
  1. Open the diff stream and start **buffering** events.
  2. Request a REST snapshot and read its `lastUpdateId`.
  3. Discard buffered events with `u < lastUpdateId`.
  4. The first applied event must satisfy `U <= lastUpdateId <= u`.
  5. Every later event must satisfy `pu == u_prev`.
  6. On any violation: invalidate the book, **write a gap marker** to the log, fetch a new snapshot, and repeat from step 1.
  - Verify these rules against the current docs before finalizing.
- Implement `recorder/writer.py`: write raw messages to Parquet (batched, hourly rotation, date-partitioned, zstd compression). Include columns for exchange timestamp, receive timestamp, stream type, and the raw fields.
- Add a heartbeat log (messages per minute, gaps, reconnects) and a simple CLI: `python -m recorder.run --config configs/data.yaml`.

**Acceptance**
- Unit tests for sync logic using synthetic sequences: normal flow, gap, duplicate, out-of-order, and a snapshot that arrives late.
- Run for at least 1 hour against the live exchange with zero unhandled exceptions.
- Report measured data size per hour and per day in `docs/progress.md`. Adjust disk plans from this number.
- Leave the recorder running continuously from this point on (on a VM if possible).

---

### Phase 2: Order-book reconstruction and event inference

**Tasks**
- `book/orderbook.py`: bids and asks as sorted structures supporting `apply_update(price, qty, side)`, best bid/ask, mid, spread, size-at-level, and top-N depth. A quantity of 0 removes the level.
- Replay the recorded raw log through the book deterministically.
- `book/events.py`: derive **add** and **reduce** events per level. Attribute reductions to **fill** vs. **cancel** by matching against `aggTrade` at the same price and side within a configurable time window. Document this as an *inference*.
- `book/validate.py`: periodically fetch fresh REST snapshots and compare them against the reconstructed book at the closest sequence ID. Report the mismatch rate by depth level.

**Acceptance**
- Property tests: the book never has crossed bids/asks after valid updates, and no negative sizes.
- Mismatch rate versus snapshots reported. Investigate and fix if non-trivial.
- Replay of one day completes within a reasonable time (use Numba or batching if too slow).
- Gap intervals are carried through as flags.

---

### Phase 3: Features, tensor and labels

**Tasks**
- `features/tabular.py`: per time step, compute OFI, top-N depth imbalance, spread, mid return, short-term volatility, signed trade volume, trade intensity, add/cancel rates near mid, liquidity center of mass and its velocity.
- `features/tensor.py`: tensor of shape `C × P × T`. Channels: bid size, ask size, executed buy volume, executed sell volume. Price axis relative to mid (ticks or bps), `log1p` scaling, then normalization with statistics computed **only on the training period**.
- `features/walls.py`: detect walls (size above a multiple of the local median) and compute age, fill/cancel history, flicker count, approach behavior, relative size and distance.
- `features/labels.py`: triple-barrier labels. Upper/lower barrier = `mid ± k·(half_spread + fees)`, with a vertical barrier of 5 to 30 seconds. Record the label and the time-to-hit.
- Resample to a fixed grid (for example 500 ms or 1 s). Mark any grid step that overlaps a gap interval as invalid.

**Acceptance**
- Leakage tests: shifting future data must not change features at `t`. Add an explicit test that perturbs data after `t` and confirms features at `t` are unchanged.
- Label tests on synthetic price paths with known outcomes.
- Class balance report: UP / DOWN / NEUTRAL fractions per day. If NEUTRAL dominates heavily, tune `k` and the horizon, and log the decision.

---

### Phase 4: Baselines and evaluation framework

**Tasks**
- `eval/splits.py`: walk-forward splits with a **purge gap at least as long as the label horizon**, plus a final untouched test period.
- `models/baselines.py`: logistic regression and XGBoost/LightGBM on tabular features (**Model A**).
- `eval/backtest.py`: event-driven backtester with maker and taker fee scenarios, a slippage/latency penalty, no trading during gaps, position limits, and per-trade logging.
- `eval/bootstrap.py`: block bootstrap confidence intervals for key metrics.
- Log results to MLflow or W&B.

**Acceptance**
- A trivial "always NEUTRAL" strategy produces zero trades; a random strategy loses roughly the cost of its trades. These sanity checks catch backtester bugs.
- Model A metrics reported on validation with confidence intervals.
- Optional: reproduce a published-style baseline on FI-2010 as a pipeline sanity check.

---

### Phase 5: Wall trust model

**Tasks**
- Define the label precisely and record it in `docs/decisions.md`: for each wall, did it still stand (or was it traded into rather than cancelled) when price came within N ticks?
- `models/trust.py`: gradient-boosted classifier over wall features, calibrated, mapped to 0 to 100.
- Validation experiment: do low-trust walls get cancelled before being reached more often than high-trust walls? Report with confidence intervals on held-out time periods.
- Add the trust score and related features to the tabular feature set.

**Acceptance**
- A clear, statistically supported answer to RQ2 (even if negative), with a plot of outcome rate versus trust decile.
- A test that trust features for a wall use only data available up to the evaluation time.

---

### Phase 6: Deep models

**Tasks**
- `models/datasets.py`: PyTorch Dataset that builds tensors on the fly from the event log or cached arrays. Use memory-mapped arrays for speed, but do not export images.
- `models/deeplob.py`: DeepLOB (CNN + inception + LSTM) on top-N levels × time (**Model B**).
- `models/heatmap_cnn.py`: small 2D CNN on the tensor (**Model C**).
- `models/fusion.py` (**Model D**):
  - Spatial branch: small 2D-CNN, then **adaptive pooling over the price axis to a few price bands** (keeps distance-from-mid information) while retaining time, then a single-layer GRU or temporal conv to a vector of dimension `d`.
  - Tabular branch: MLP over engineered and trust features, projected to `d`.
  - Fusion: concatenate, dropout, linear head to 3 logits.
  - Config switch for pooling variants: (1) band pooling + GRU (default), (2) GAP over time only, (3) full GAP, optionally with coordinate channels. These are ablations.
- Use class-weighted or focal loss if needed, early stopping on a validation metric, and fixed seeds. Run at least 3 seeds for headline results.

**Acceptance**
- Each model overfits a tiny subset (sanity check) and then trains on full data with logged curves.
- Ablation table: A vs. B vs. C vs. D, and the pooling variants, with confidence intervals. This answers RQ1.

---

### Phase 7: Calibration and abstention

**Tasks**
- `eval/calibration.py`: temperature scaling fitted on the validation period only. Reliability diagrams and expected calibration error before and after.
- `eval/abstention.py`: decision rule `EV = P(up)·G_up − P(down)·L_down − fees − slippage`. Trade only if EV is clearly positive **and** confidence exceeds a threshold selected on validation. Produce risk-coverage curves (metric vs. fraction of times acting).
- Optional: conformal prediction for set-valued outputs.
- Run the full backtest with abstention on the untouched test period **once**.

**Acceptance**
- ECE improves after calibration (or document why not).
- Threshold selection uses validation data only, and the test period is used once.
- RQ3 answered: performance with vs. without abstention, under multiple fee scenarios and calm/volatile regimes.

---

### Phase 8: Explainability

**Tasks**
- SHAP on the tabular models and tabular branch.
- Grad-CAM (or Captum) on the CNN branch.
- `explain/templates.py`: plain-language reasons (for example "Sell wall at X has trust 31/100, so resistance is discounted").

**Acceptance**
- Explanations generated for a sample of decisions and reviewed manually for sanity.
- All UI and report text frames these as diagnostics of model behavior, not causal explanations.

---

### Phase 9: Replay dashboard

**Tasks**
- Streamlit app loading pre-selected clips (Parquet) with play/pause, speed (1x, 5x, 20x), time slider and clip selector.
- Panels: scrolling heatmap (Plotly) with mid-price line, walls annotated with trust scores, calibrated UP / DOWN / NEUTRAL probabilities, **TRADE / NO TRADE** state, "why" panel.
- Side-by-side mode: plain heatmap vs. Liquidity Trust view.
- Choose and save 3 to 4 demo clips: a bait wall, a real wall, a no-trade moment, and one failed signal.
- Provide a small sample dataset so the app runs without the full recording.

**Acceptance**
- App runs from a clean checkout with one command and the sample data.
- Record a 2 to 3 minute backup screen video of the demo.

---

### Phase 10: Report and packaging

**Tasks**
- Final README with setup, how to run each stage, and a results summary.
- Report figures: ablation table, reliability diagrams, trust-vs-outcome plot, risk-coverage curves, backtest equity curves with confidence bands.
- A limitations section: inferred fill/cancel split, short data period, single asset, fee sensitivity, explanations are diagnostics.

**Acceptance**
- Another person can reproduce the headline numbers from the README and configs.

---

## 5. Priority tiers (cut from the bottom if time runs short)

- **Must-have:** Phases 0 to 5, 7 (without conformal), 9 (basic).
- **Target:** DeepLOB and heatmap CNN (Phase 6, Models B and C), SHAP.
- **Stretch:** Fusion pooling ablations, Grad-CAM, conformal prediction, BTC-to-ETH transfer test.

---

## 6. Common mistakes to avoid

- Applying a diff event before the snapshot sync condition is satisfied, or ignoring a `pu` mismatch.
- Computing normalization statistics on the full dataset (leakage). Use the training period only.
- Random train/test splits on time series. Always time-ordered with a purge gap.
- Choosing the abstention threshold on the test set.
- Reporting accuracy without costs. Always report PnL after fees and slippage.
- Treating the CNN's win on one seed or one day as a result. Use multiple seeds and confidence intervals.
- Letting the backtester fill at prices that were not available (for example filling at mid or at a level that has already moved).
- Describing low-trust walls as "spoofing".

---

## 7. How the human will work with the agent

For each phase, the human will say: **"Do Phase N."** The agent should then:

1. Restate the phase tasks and acceptance checks in its own words.
2. List any assumptions or open questions (put them in `docs/decisions.md`).
3. Implement the code and tests.
4. Run the tests and acceptance checks, and show the results.
5. Update `docs/progress.md` and stop. Do not begin the next phase automatically.