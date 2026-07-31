"""
pipeline_spa_hac_test.py
==============================================================================
HAI BAI TEST BO SUNG - Muc do nghiem ngat cao nhat con lai trong audit.

TEST A: White's Reality Check / Stationary-Bootstrap SPA (Hansen 2005;
        Sullivan-Timmermann-White 1999)
    Kiem dinh TRUC TIEP cau hoi "co phai chien luoc tot nhat trong 60 to hop
    (20 dai x 3 model) chi la may man?" bang bootstrap thuc nghiem tren
    chinh phan phoi du lieu that (KHONG gia dinh Gauss nhu DSR, khong dinh
    loi don vi/kurtosis nhu lan truoc). Dung stationary bootstrap (Politis &
    Romano 1994) de bao toan dung cau truc tu tuong quan theo thoi gian.
    Tin hieu (sigs) cua ca 60 chien luoc duoc GIU CO DINH tu du lieu that;
    chi chuoi loi nhuan thi truong duoc resample -> dung dan vi day la
    kiem dinh dung, khong phai "xao tin hieu" nhu loi cu trong code goc.

TEST B: Newey-West HAC-adjusted significance cho chien luoc dau bang
        (LogReg [0.33, 0.67])
    Tinh lai sai so chuan (standard error) co dieu chinh tu tuong quan
    chuoi thoi gian bang Newey-West long-run variance (Bartlett kernel),
    thay vi gia dinh i.i.d. nhu cac CI truoc do. Bao gom "N hieu dung"
    (effective sample size) de thay ro muc do phinh to gia tao khi coi
    moi ngay la mot quan sat doc lap.

Dau vao: wfv_predictions_final.csv, VNIndex_Features_v4.csv
"""

import pandas as pd
import numpy as np
import warnings
from scipy.stats import norm

warnings.filterwarnings('ignore')

TRANSACTION_COST = 0.002
N_BOOTSTRAP      = 1000
MEAN_BLOCK_LEN   = 20     # ~ trung binh thoi gian giu lenh, cho stationary bootstrap
NW_LAGS          = 20     # do tre Newey-West (Bartlett kernel)
HEADLINE_KEY     = ('LogReg', 0.33, 0.67)

FILTER_BANDS = [
    (0.50, 0.50), (0.49, 0.51), (0.48, 0.52), (0.47, 0.53), (0.46, 0.54),
    (0.45, 0.55), (0.44, 0.56), (0.43, 0.57), (0.42, 0.58), (0.41, 0.59),
    (0.40, 0.60), (0.39, 0.61), (0.38, 0.62), (0.37, 0.63), (0.36, 0.64),
    (0.35, 0.65), (0.34, 0.66), (0.33, 0.67), (0.32, 0.68), (0.30, 0.70)
]

print("=" * 78)
print("START: WHITE'S REALITY CHECK (SPA) + NEWEY-WEST HAC SIGNIFICANCE TEST")
print("=" * 78)

# ============================== LOAD DATA ==============================
pred_df = pd.read_csv("wfv_predictions_final.csv", parse_dates=["Date"])
df_v4   = pd.read_csv("VNIndex_Features_v4.csv", parse_dates=["Date"])
pred_df = pred_df.sort_values("Date").reset_index(drop=True)
df_v4   = df_v4.sort_values("Date").reset_index(drop=True)
pred_df = pred_df.drop(columns=['VN_Return'], errors='ignore')
pred_df = pred_df.merge(df_v4[['Date', 'VN_Return']], on='Date', how='left')

ret_next = pred_df['VN_Return'].shift(-1).fillna(0).values
n_days = len(ret_next)


def build_signal(prob, lo, hi):
    sigs = np.zeros(len(prob), dtype=int)
    pos = 0
    for i, p in enumerate(prob):
        if p > hi:
            pos = 1
        elif p < lo:
            pos = 0
        sigs[i] = pos
    return sigs


def strat_daily_pnl(sigs, ret_arr, tc=TRANSACTION_COST):
    pos_change = np.abs(np.diff(sigs, prepend=sigs[0]))
    return sigs * ret_arr - pos_change * tc


# ================== DUNG LAI 60 TO HOP TU DU LIEU THAT ==================
print("\n[1/3] Dung lai 60 to hop (20 dai x 3 model) tu du lieu that...")
all_strats = {}
for name in ['RF', 'XGBoost', 'LogReg']:
    prob = pred_df[f'prob_{name}'].values
    for lo, hi in FILTER_BANDS:
        sigs = build_signal(prob, lo, hi)
        pnl = strat_daily_pnl(sigs, ret_next)
        all_strats[(name, lo, hi)] = {'sigs': sigs, 'pnl': pnl, 'mean': pnl.mean()}

keys = list(all_strats.keys())
means_real = np.array([all_strats[k]['mean'] for k in keys])
best_idx = int(np.argmax(means_real))
best_key = keys[best_idx]
observed_max_mean = means_real[best_idx]
print(f"   -> Chien luoc tot nhat (mean daily PnL that) = {best_key}")
print(f"   -> Observed max mean daily PnL = {observed_max_mean:.6f}")

# ============ TEST A: STATIONARY BOOTSTRAP REALITY CHECK ============
print(f"\n[2/3] TEST A: White's Reality Check qua Stationary Bootstrap "
      f"({N_BOOTSTRAP} lan, block~{MEAN_BLOCK_LEN} ngay)...")


def stationary_bootstrap_indices(n, mean_block_len, rng):
    p = 1.0 / mean_block_len
    idx = np.empty(n, dtype=int)
    cur = rng.integers(0, n)
    for i in range(n):
        idx[i] = cur
        if rng.random() < p:
            cur = rng.integers(0, n)
        else:
            cur = (cur + 1) % n
    return idx


rng = np.random.default_rng(42)
boot_max_stats = np.zeros(N_BOOTSTRAP)
sigs_matrix = np.array([all_strats[k]['sigs'] for k in keys])          # (60, n_days)
pos_change_matrix = np.abs(np.diff(sigs_matrix, axis=1,
                                    prepend=sigs_matrix[:, :1]))        # (60, n_days)
tc_cost_per_day = pos_change_matrix * TRANSACTION_COST                 # fixed regardless of resample

for b in range(N_BOOTSTRAP):
    idx = stationary_bootstrap_indices(n_days, MEAN_BLOCK_LEN, rng)
    ret_boot = ret_next[idx]
    pnl_boot = sigs_matrix * ret_boot[None, :] - tc_cost_per_day        # (60, n_days)
    boot_means = pnl_boot.mean(axis=1)
    boot_devs = boot_means - means_real                                # center at each strat's own real mean
    boot_max_stats[b] = boot_devs.max()

p_value_rc = (boot_max_stats >= observed_max_mean).mean()
print(f"   -> Ky vong nhieu (mean cua max bootstrap, N=60)     = {boot_max_stats.mean():.6f}")
print(f"   -> p-value White's Reality Check (dung cho ca 60 to hop) = {p_value_rc:.4f}")

# ====== TEST B: NEWEY-WEST HAC SIGNIFICANCE CHO CHIEN LUOC DAU BANG ======
print(f"\n[3/3] TEST B: Newey-West HAC-adjusted significance cho "
      f"{HEADLINE_KEY[0]} [{HEADLINE_KEY[1]:.2f},{HEADLINE_KEY[2]:.2f}]...")


def newey_west_lrv(x, lags):
    x = x - x.mean()
    T = len(x)
    gamma0 = np.dot(x, x) / T
    lrv = gamma0
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        gamma_l = np.dot(x[l:], x[:-l]) / T
        lrv += 2 * w * gamma_l
    return lrv


lr_pnl = all_strats[HEADLINE_KEY]['pnl']
naive_var = lr_pnl.var(ddof=1)
hac_lrv = newey_west_lrv(lr_pnl, NW_LAGS)
inflation = hac_lrv / naive_var
n_eff = n_days / inflation

se_naive_daily = np.sqrt(naive_var / n_days)
se_hac_daily = np.sqrt(hac_lrv / n_days)

t_naive = lr_pnl.mean() / se_naive_daily
t_hac = lr_pnl.mean() / se_hac_daily
p_naive = 2 * (1 - norm.cdf(abs(t_naive)))
p_hac = 2 * (1 - norm.cdf(abs(t_hac)))

print(f"   -> Variance thuong (gia dinh i.i.d.)          = {naive_var:.8f}")
print(f"   -> Long-run variance (Newey-West, lag={NW_LAGS})   = {hac_lrv:.8f}")
print(f"   -> He so lam phinh do tu tuong quan            = {inflation:.2f}x")
print(f"   -> N thuc te = {n_days}  ->  N hieu dung ~ {n_eff:.0f}")
print(f"   -> t-stat (chuan, gia dinh i.i.d.)              = {t_naive:.3f}  (p={p_naive:.4f})")
print(f"   -> t-stat (Newey-West HAC, dung)                = {t_hac:.3f}  (p={p_hac:.4f})")

print("\n" + "=" * 78)
print("HOAN TAT TEST A (White's Reality Check) + TEST B (Newey-West HAC).")
print("=" * 78)
