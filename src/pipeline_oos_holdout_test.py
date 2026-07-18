"""
pipeline_oos_holdout_test.py
==============================================================================
KIEM DINH OUT-OF-SAMPLE HOLDOUT THUC SU - Khac phuc Selection Bias khi chon
dai loc Persistence Filter.

VAN DE:
    Toan bo 20 dai loc trong pipeline_grid_filters.py duoc quet tren TOAN BO
    chuoi ngay (bao gom ca doan du lieu gan nhat). Viec chon dai "tot nhat"
    (vi du LogReg = [0.33, 0.67]) do do DA NHIN THAY du lieu tuong lai tai
    thoi diem ra quyet dinh -> vi pham nguyen tac walk-forward thuc su, du
    ban than xac suat du doan prob_{model} van hoan toan OOS (huan luyen
    walk-forward dung cach o buoc feature_engineering_v4.py).

GIAI PHAP:
    1. Cat HOLDOUT_MONTHS thang cuoi cung cua chuoi ra lam vung "chua tung
       bi dung toi" (true holdout).
    2. Chay lai grid search 20 dai CHI tren phan du lieu development (truoc
       holdout) -> band toi uu tim duoc o buoc nay mo phong dung nhung gi
       nha nghien cuu "biet" tai thoi diem do, khong co look-ahead.
    3. Ap dung CO DINH band do (khong tinh chinh gi them) len dung vung
       holdout -> do hieu nang thuc su ngoai mau.
    4. So sanh voi cach lam "ngay tho" (dung thang band da chon tu full-sample
       cu, vi du [0.33, 0.67]) ap len holdout, de thay chenh lech bao nhieu.

Dau vao: wfv_predictions_fix_v4.csv, VNIndex_Features_v4.csv
Dau ra : oos_holdout_results.csv (bang so sanh Dev vs Holdout moi model)
"""

import pandas as pd
import numpy as np
import warnings
from scipy.stats import skew, kurtosis

warnings.filterwarnings('ignore')

# ============================== CAU HINH ==============================
HOLDOUT_MONTHS   = 9          # So thang cuoi cung cach ly hoan toan (dieu chinh neu can)
TRANSACTION_COST = 0.002
FILTER_BANDS = [
    (0.50, 0.50), (0.49, 0.51), (0.48, 0.52), (0.47, 0.53), (0.46, 0.54),
    (0.45, 0.55), (0.44, 0.56), (0.43, 0.57), (0.42, 0.58), (0.41, 0.59),
    (0.40, 0.60), (0.39, 0.61), (0.38, 0.62), (0.37, 0.63), (0.36, 0.64),
    (0.35, 0.65), (0.34, 0.66), (0.33, 0.67), (0.32, 0.68), (0.30, 0.70)
]
# Band "day tho" da chon truoc do tu full-sample grid search (de doi chieu)
ORIGINAL_FULL_SAMPLE_BAND = {
    'RF':      (0.48, 0.52),
    'XGBoost': (0.41, 0.59),
    'LogReg':  (0.33, 0.67),
}

print("=" * 78)
print("START: OUT-OF-SAMPLE HOLDOUT TEST (Chong Selection Bias trong Filter Band)")
print("=" * 78)

# ============================== BUOC 1: NAP DU LIEU ==============================
try:
    pred_df = pd.read_csv("wfv_predictions_fix_v4.csv", parse_dates=["Date"])
    df_v4   = pd.read_csv("VNIndex_Features_v4.csv", parse_dates=["Date"])
except FileNotFoundError:
    raise FileNotFoundError("Thieu wfv_predictions_fix_v4.csv hoac VNIndex_Features_v4.csv.")

pred_df = pred_df.sort_values("Date").reset_index(drop=True)
df_v4   = df_v4.sort_values("Date").reset_index(drop=True)
pred_df = pred_df.drop(columns=['VN_Return'], errors='ignore')
pred_df = pred_df.merge(df_v4[['Date', 'VN_Return']], on='Date', how='left')

ret_next = pred_df['VN_Return'].shift(-1).fillna(0).values
n_days   = len(pred_df)

cutoff_date = pred_df['Date'].max() - pd.DateOffset(months=HOLDOUT_MONTHS)
dev_mask     = (pred_df['Date'] < cutoff_date).values
holdout_mask = (pred_df['Date'] >= cutoff_date).values

print(f"\nTong chuoi     : {pred_df['Date'].min().date()} -> {pred_df['Date'].max().date()} ({n_days} phien)")
print(f"Cutoff holdout : {cutoff_date.date()}  (giu lai {HOLDOUT_MONTHS} thang cuoi)")
print(f"Development    : {dev_mask.sum()} phien  |  Holdout: {holdout_mask.sum()} phien")

if holdout_mask.sum() < 60:
    print("\n[CANH BAO] Vung holdout qua ngan (<60 phien) de rut ra ket luan thong ke dang tin.")

# ============================== BUOC 2: HAM DANH GIA ==============================
def build_signal(prob, band_lo, band_hi):
    sigs = np.zeros(len(prob), dtype=int)
    current_pos = 0
    for i, p in enumerate(prob):
        if p > band_hi:
            current_pos = 1
        elif p < band_lo:
            current_pos = 0
        sigs[i] = current_pos
    return sigs

def evaluate(signals, returns_next, mask, tc_rate=0.002):
    """Danh gia hieu nang CHI tren cac phien nam trong mask (Dev hoac Holdout)."""
    sigs_m = signals[mask]
    ret_m  = returns_next[mask]
    if len(sigs_m) == 0:
        return dict(sharpe=0.0, n=0, turn=0.0, skew=0.0, kurt=0.0)

    pos_change = np.abs(np.diff(sigs_m, prepend=sigs_m[0]))
    strat_ret_net = sigs_m * ret_m - (pos_change * tc_rate)

    if len(strat_ret_net) < 5 or strat_ret_net.std() < 1e-8:
        return dict(sharpe=0.0, n=len(strat_ret_net), turn=0.0, skew=0.0, kurt=0.0)

    ann_ret = strat_ret_net.mean() * 252
    ann_vol = strat_ret_net.std() * np.sqrt(252)
    sh = ann_ret / ann_vol
    turn = (pos_change.sum() / len(sigs_m)) * 100
    return dict(sharpe=sh, n=len(strat_ret_net), turn=turn,
                skew=skew(strat_ret_net), kurt=kurtosis(strat_ret_net))

# ============================== BUOC 3: GRID SEARCH CHI TREN DEV ==============================
print("\n[BUOC A] Chay lai grid search 20 dai CHI tren vung Development (khong nhin thay Holdout)...")

dev_best = {}
for name in ['RF', 'XGBoost', 'LogReg']:
    prob = pred_df[f'prob_{name}'].values
    best_sh, best_band = -np.inf, None
    for lo, hi in FILTER_BANDS:
        sigs = build_signal(prob, lo, hi)
        res = evaluate(sigs, ret_next, dev_mask, TRANSACTION_COST)
        if res['sharpe'] > best_sh:
            best_sh, best_band = res['sharpe'], (lo, hi)
    dev_best[name] = {'band': best_band, 'dev_sharpe': best_sh}
    match = "TRUNG voi band full-sample cu" if best_band == ORIGINAL_FULL_SAMPLE_BAND[name] else "KHAC band full-sample cu"
    orig = ORIGINAL_FULL_SAMPLE_BAND[name]
    print(f"   -> {name:<7} | Band toi uu (chi Dev) = [{best_band[0]:.2f}, {best_band[1]:.2f}]  "
          f"| Dev Net Sharpe = {best_sh:.3f}  | {match} ([{orig[0]:.2f}, {orig[1]:.2f}])")

# ============================== BUOC 4: AP DUNG LEN HOLDOUT (2 CACH) ==============================
print("\n[BUOC B] Ap dung band CO DINH (khong tinh chinh) len vung Holdout that su...")
print("-" * 100)
print(f"{'Model':<9} {'Cach chon band':<28} {'Band':<14} {'Holdout SR':>11} {'Holdout n':>10} {'95% CI (SR)':>20}")
print("-" * 100)

results_rows = []
for name in ['RF', 'XGBoost', 'LogReg']:
    prob = pred_df[f'prob_{name}'].values

    for method, band in [
        ("Dev-only (dung chuan)", dev_best[name]['band']),
        ("Full-sample cu (doi chieu)", ORIGINAL_FULL_SAMPLE_BAND[name]),
    ]:
        sigs = build_signal(prob, band[0], band[1])
        res = evaluate(sigs, ret_next, holdout_mask, TRANSACTION_COST)

        # Khoang tin cay xap xi cho Sharpe holdout (Mertens/Lo, dieu chinh skew/kurt)
        n = res['n']
        if n > 5 and res['sharpe'] != 0.0:
            sr_daily = res['sharpe'] / np.sqrt(252)
            pearson_kurt = res['kurt'] + 3
            var_term = 1 - res['skew']*sr_daily + (pearson_kurt-1)/4*sr_daily**2
            se_daily = np.sqrt(max(1e-8, var_term) / (n-1))
            se_annual = se_daily * np.sqrt(252)
            ci = f"[{res['sharpe']-1.96*se_annual:.2f}, {res['sharpe']+1.96*se_annual:.2f}]"
        else:
            ci = "n/a"

        print(f"{name:<9} {method:<28} [{band[0]:.2f},{band[1]:.2f}]  {res['sharpe']:>10.3f}  {n:>9}  {ci:>20}")
        results_rows.append({
            'Model': name, 'Method': method, 'Band_lo': band[0], 'Band_hi': band[1],
            'Holdout_Sharpe': res['sharpe'], 'Holdout_N': n, 'Holdout_95CI': ci
        })

# Buy & Hold baseline tren dung vung Holdout de doi chieu
bnh_sigs = np.ones(n_days, dtype=int)
bnh_res = evaluate(bnh_sigs, ret_next, holdout_mask, tc_rate=0.0)
print("-" * 100)
print(f"{'Buy&Hold':<9} {'(baseline)':<28} {'---':<14} {bnh_res['sharpe']:>10.3f}  {bnh_res['n']:>9}")
print("=" * 100)

pd.DataFrame(results_rows).to_csv("oos_holdout_results.csv", index=False)
print("\nDa luu: oos_holdout_results.csv")
print("HOAN TAT kiem dinh Out-of-Sample Holdout.")
