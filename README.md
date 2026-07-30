# A Leakage-Free Evaluation Framework for Vietnamese Stock Market (VNIndex) Trend Prediction

Code and results accompanying the paper *"A Leakage-Free Evaluation Framework for Vietnamese Stock Market Trend Prediction Using Machine Learning"* (2014–2025).

> **TL;DR:** Classic technical indicators (RSI, MACD, Bollinger Bands...) carry almost no predictive power for next-day VNIndex direction (AUC ≈ 0.50) once data leakage is removed. A modest but statistically significant edge (cumulative AUC ≈ 0.55) comes mainly from **lagged global macro spillover** (S&P 500 / Nasdaq / VIX / DXY). This edge is heavily eroded by realistic transaction costs on a daily-rebalanced strategy, but survives when trading frequency is reduced via a persistence filter — beating Buy & Hold on a net-of-cost, risk-adjusted basis in 5 of 9 test years, mainly through drawdown reduction during downturns rather than upside capture during rallies. However, a strict true holdout test reveals that the filter band driving the best-performing model's headline result was selected in-sample: on genuinely unseen data, the linear model's edge does not hold up, while tree-based ensembles remain more stable.

## Repository structure

```
vnindex-repo/
├── .gitignore
├── CITATION.cff
├── README.md
├── requirements.txt
├── data/
│   ├── VNIndex_Raw.csv                    # Raw OHLCV + global market data (output of data_collection.py)
│   ├── VNIndex_Features.csv               # Early feature set (output of feature_engineering.py)
│   └── VNIndex_Features_v4.csv            # 68 clean features, used across the full pipeline (output of feature_engineering_v4.py)
├── notebooks/
│   └── VNIndex_Trend_Prediction.ipynb     # End-to-end pipeline, consolidated notebook
├── results/
│   ├── figures/
│   │   ├── fig4_ablation_study_labeled.png    # Fig. 4 — ablation by feature group
│   │   ├── fig5_yearly.png                    # Fig. 5 — annual Sharpe ratios
│   │   ├── fig6_equity.png                    # Fig. 6 — cumulative equity curve
│   │   └── FINAL_PAPER_RESULTS.png            # Combined 2-panel summary figure
│   └── tables/
│       ├── ablation_study_results.csv         # Table IV
│       ├── grid_filters_results_advanced.csv  # Section III-G — 60-band grid search
│       ├── grid_filters_final_audit.csv       # Table III
│       ├── oos_holdout_results.csv            # Section III-H — true 9-month holdout
│       ├── wfv_predictions_fix_v4.csv         # Per-fold walk-forward predictions, unfiltered
│       ├── wfv_predictions_final.csv          # Final filtered predictions, used for Fig. 5/6
│       └── wfv_results_CLEAN.csv              # Clean per-fold AUC (input to Fig. 3)
└── src/
    ├── data_collection.py                  # Phase 1 — vnstock / yfinance data pull → VNIndex_Raw.csv
    ├── feature_engineering.py              # Phase 2 — early feature set → VNIndex_Features.csv
    ├── feature_engineering_v4.py           # Phase 2 — builds VNIndex_Features_v4.csv, then runs the 16-fold walk-forward training (Phase 4/5) → wfv_results_CLEAN.csv, wfv_predictions_fix_v4.csv
    ├── AUC_feature.py                      # Section IV-B — AUC ablation by feature group
    ├── ablation_image.py                   # Renders Fig. 4
    ├── pipeline_grid_filters.py            # Section III-G — filter band grid search (60 trials)
    ├── pipeline_table3_audit.py            # Audit script — recomputes Table III
    ├── pipeline_backtest_final.py          # Main backtest — Fig. 6, FINAL_PAPER_RESULTS.png, Wilcoxon/permutation tests
    ├── Trading_Performance.py              # Table V, Fig. 5 — annual Sharpe ratios
    ├── pipeline_oos_holdout_test.py        # Section III-H — true holdout test
    ├── pipeline_dsr_calculation.py         # Section III-H — Deflated Sharpe Ratio
    ├── pipeline_spa_hac_test.py            # Section III-H — Newey-West HAC / Reality Check
    ├── pipeline_stress_test_optimized.py   # Section III-H — block-permutation test
    └── pipeline_ultimate_audit.py          # Combines all robustness audits
```
## Methodology summary

- **Task:** binary classification — will next-day VNIndex return exceed +0.1% (UP) or not (DOWN)?
- **Data:** VNIndex OHLCV (2014–2025, via `vnstock`), global markets (S&P 500, Nasdaq, Gold, Oil, DXY, VIX, HSI, Nikkei via `yfinance`), sector proxies (VCB/VHM/HPG), calendar effects — 2,793 trading sessions, 68 clean features across 4 groups.
- **Leakage controls:** three sources identified and fixed — (i) timezone mismatch between VN and Western/HK market closes, (ii) an accidentally retained future-return feature, (iii) feature scaler fit on the full dataset before splitting.
- **Validation:** walk-forward, 3-year rolling train / 6-month test, 16 folds (2017–2025) — no random shuffling of time-ordered data.
- **Models:** Random Forest, XGBoost, Logistic Regression (LSTM tested in an earlier iteration, dropped from the final pipeline for stability/interpretability reasons — see notebook history).
- **Ablation study:** features grouped into Technical / Money-flow / Macro-global / Calendar to isolate where the predictive signal actually comes from.
- **Backtest:** causal signal-to-return alignment (signal at day *T* applied to return at day *T+1*), transaction costs (0.2%/trade), and a persistence (hysteresis) filter to control turnover.
- **Robustness checks:** block-permutation test, Newey-West HAC test, Deflated Sharpe Ratio, White's Reality Check, and a strict true out-of-sample holdout (final 9 months, fully excluded from filter-band selection).

## Key results

| Model | Cumulative AUC (16 folds) | Net Sharpe (raw) | Net Sharpe (+ persistence filter) |
|---|---|---|---|
| Random Forest | 0.552 | 0.28 | 0.81 |
| XGBoost | 0.539 | 0.01 | 0.68 |
| Logistic Regression | 0.547 | -0.05 | **1.04** |
| MA Crossover (baseline) | 0.505 | — | 0.62 |
| Buy & Hold (baseline) | — | — | 0.51 |

Net Sharpe values above use the full-sample optimized filter band. **On a strict, never-seen 9-month holdout period**, Random Forest and XGBoost retain a positive edge (net Sharpe 0.72 and 0.58), while Logistic Regression's advantage does not hold up — its net Sharpe drops well below zero, indicating that its full-sample band selection was itself a hidden source of overfitting. See `results/tables/` for per-fold, per-year, ablation, and holdout results.

## Reproducing the results

The original notebook (`notebooks/VNIndex_Trend_Prediction.ipynb`) is outdated and kept for reference only. To reproduce the results, run the scripts below **in order, from inside `src/`**, with all input/output CSV and JSON files kept in the same working directory as the scripts (the current scripts use bare filenames, not the `data/` / `results/` paths shown in the repository structure above).

```bash
git clone https://github.com/HieuDZ609/vnindex-trend-prediction.git
cd vnindex-trend-prediction/src
pip install -r ../requirements.txt
```

1. `data_collection.py` — pulls raw OHLCV + global market data → `VNIndex_Raw.csv`
2. `feature_engineering.py` — builds the early feature set → `VNIndex_Features.csv`
3. `feature_engineering_v4.py` — builds the final 68-feature set (`VNIndex_Features_v4.csv`), then runs the 16-fold walk-forward training on RF / XGBoost / LogReg → `wfv_results_CLEAN.csv`, `wfv_predictions_fix_v4.csv`
4. `AUC_feature.py` — AUC ablation by feature group (Section IV-B)
5. `ablation_image.py` — renders Fig. 4
6. `pipeline_grid_filters.py` — filter band grid search, 60 trials (Section III-G)
7. `pipeline_table3_audit.py` — recomputes Table III
8. `pipeline_backtest_final.py` — main backtest, produces Fig. 6, `FINAL_PAPER_RESULTS.png`, Wilcoxon/permutation tests
9. `Trading_Performance.py` — Table V, Fig. 5 (annual Sharpe ratios)
10. `pipeline_oos_holdout_test.py` — true 9-month holdout test (Section III-H)
11. `pipeline_dsr_calculation.py` — Deflated Sharpe Ratio (Section III-H)
12. `pipeline_spa_hac_test.py` — Newey-West HAC / White's Reality Check (Section III-H)
13. `pipeline_stress_test_optimized.py` — block-permutation test (Section III-H)
14. `pipeline_ultimate_audit.py` — combines all robustness audits

## Data sources & disclaimer

- Vietnamese market data via [`vnstock`](https://github.com/thinh-vu/vnstock).
- Global market data via [`yfinance`](https://github.com/ranaroussi/yfinance) (unofficial Yahoo Finance API).
- Search-interest data via `pytrends` (tested, excluded from final features — correlation with target < 0.03).
- This repository is for **academic/research purposes only** and does not constitute investment advice.

## Limitations

- Predictive edge is modest (cumulative AUC ≈ 0.55) and not stable across all market regimes (fold-level AUC ranges roughly from 0.42 to 0.68).
- Backtest assumes no slippage beyond the stated transaction cost and no capacity constraints.
- The persistence-filter band for Logistic Regression was selected via a full-sample grid search; a strict holdout test shows this configuration does not generalize — its headline net Sharpe does not hold up on genuinely unseen data. Random Forest and XGBoost are more stable under the same test but still carry wide confidence intervals over the short (9-month) holdout window.
- The system is not yet validated for live trading; the natural next step is live paper trading on frozen, truly out-of-sample data.

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{trhieu2026vnindexleakage,
  title   = {A Leakage-Free Evaluation Framework for Vietnamese Stock Market Trend Prediction Using Machine Learning},
  author  = {Hieu, Tho},
  year    = {2026},
  journal = {TBD},
  url     = {https://github.com/HieuDZ609/vnindex-trend-prediction}
}
```

See also [`CITATION.cff`](./CITATION.cff).
