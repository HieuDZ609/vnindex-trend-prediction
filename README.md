# Predicting Vietnamese Stock Market (VNIndex) Trends Using Machine Learning

Code and results accompanying the paper *"Predicting Vietnamese Stock Market Trends Using Machine Learning Approaches"* (2014–2025).

> **TL;DR:** Classic technical indicators (RSI, MACD, Bollinger Bands...) carry almost no predictive power for next-day VNIndex direction (AUC ≈ 0.50). A modest but statistically significant edge (AUC ≈ 0.56–0.57) comes mainly from **lagged global market spillover** (S&P 500 / Nasdaq / VIX / DXY). This edge largely disappears once realistic transaction costs are applied to a daily-rebalanced strategy, but survives when trading frequency is reduced via a persistence filter — beating Buy & Hold on a net-of-cost, risk-adjusted basis in 6 of 9 test years, mainly through drawdown reduction during downturns rather than upside capture during rallies.

## Repository structure

```
├── notebooks/
│   └── VNIndex_Trend_Prediction.ipynb   # end-to-end pipeline (cleaned, final version)
├── src/                                  # modular scripts mirroring the notebook
│   ├── data_collection.py                # vnstock / yfinance data pull
│   ├── feature_engineering.py            # technical, money-flow, macro, calendar features
│   ├── walk_forward_validation.py        # 16-fold walk-forward evaluation (train 3y / test 6m)
│   └── backtest.py                       # equity curve, transaction costs, persistence filter
├── data/processed/
│   └── VNIndex_Features_v4.csv           # feature-engineered dataset used for the final results
├── results/
│   ├── figures/                          # paper-ready figures (300 dpi)
│   └── tables/                           # wfv_results, ablation_results, performance tables (.csv)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Methodology summary

- **Task:** binary classification — will next-day VNIndex return exceed +0.1% (UP) or not (DOWN)?
- **Data:** VNIndex OHLCV (2014–2025, via `vnstock`), global markets (S&P 500, Nasdaq, Gold, Oil, DXY, VIX, HSI, Nikkei via `yfinance`), sector proxies (VCB/VHM/HPG), calendar effects.
- **Leakage controls:** cross-market features lagged 1 day (VN closes before US markets); absolute price levels and next-day return excluded from features; stationarity verified via ADF test.
- **Validation:** walk-forward, 3-year rolling train / 6-month test, 16 folds (2017–2025) — no random shuffling of time-ordered data.
- **Models:** Random Forest, XGBoost, Logistic Regression (LSTM tested in an earlier iteration, dropped from the final pipeline for stability/interpretability reasons — see notebook history).
- **Ablation study:** features grouped into Technical / Money-flow / Macro-global / Calendar to isolate where the predictive signal actually comes from.
- **Backtest:** causal signal-to-return alignment (signal at day *T* applied to return at day *T+1*), transaction costs (0.2%/trade), and a persistence (hysteresis) filter to control turnover.

## Key results

| Model | AUC (mean ± std, 16 folds) | Net Sharpe (after 0.2% TC) |
|---|---|---|
| Random Forest | 0.556 ± 0.068 | 0.01 |
| XGBoost | 0.561 ± 0.062 | 0.04 |
| Logistic Regression | 0.572 ± 0.062 | 0.18 |
| **LogReg + persistence filter** | 0.572 | **0.72** |
| MA Crossover (baseline) | 0.510 | 0.58 |
| Buy & Hold (baseline) | — | 0.43 |

Full per-fold, per-year, and ablation results are in `results/tables/`.

## Reproducing the results

```bash
git clone https://github.com/HieuDZ609/vnindex-trend-prediction.git
cd vnindex-trend-prediction
pip install -r requirements.txt
jupyter notebook notebooks/VNIndex_Trend_Prediction.ipynb
```

The notebook reads/writes to `./data/` and `./results/` by default. A pre-computed feature file (`data/processed/VNIndex_Features_v4.csv`) is included so tables and figures can be reproduced exactly even as live market data continues to update.

## Data sources & disclaimer

- Vietnamese market data via [`vnstock`](https://github.com/thinh-vu/vnstock).
- Global market data via [`yfinance`](https://github.com/ranaroussi/yfinance) (unofficial Yahoo Finance API).
- Search-interest data via `pytrends` (tested, excluded from final features — correlation with target < 0.03).
- This repository is for **academic/research purposes only** and does not constitute investment advice.

## Limitations

- Predictive edge is modest (AUC ≈ 0.57) and not stable across all market regimes (fold-level AUC ranges from ~0.42 to ~0.68).
- Backtest assumes no slippage beyond the stated transaction cost and no capacity constraints.
- Persistence-filter band was selected via in-sample search over the test period; out-of-sample validation on a fresh period is recommended before any practical use.

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{krone2026vnindex,
  title   = {Predicting Vietnamese Stock Market Trends Using Machine Learning Approaches},
  author  = {Hieu, Tho},
  year    = {2026},
  journal = {TBD},
  url     = {https://github.com/HieuDZ609/vnindex-trend-prediction}
}
```

See also [`CITATION.cff`](./CITATION.cff).

## License

Code released under the [MIT License](./LICENSE). Third-party data remains subject to its original providers' terms.
