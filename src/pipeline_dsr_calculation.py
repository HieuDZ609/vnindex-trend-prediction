"""
pipeline_dsr_calculation.py
==============================================================================
DEFLATED SHARPE RATIO (DSR) - Bailey & Lopez de Prado (2014)

Tinh dung cong thuc goc, tranh 2 loi da phat hien trong ban tinh truoc do:
  1. Tron don vi: Sharpe ANNUALIZED dua thang vao cong thuc dung T o dang
     HANG NGAY (T = so phien giao dich). Script nay luon de-annualize truoc
     khi dua vao cong thuc, chi annualize lai SR* luc bao cao cuoi cung.
  2. Nham "excess kurtosis" (quy uoc mac dinh cua scipy.stats.kurtosis,
     chuan=0) voi "Pearson kurtosis" (quy uoc cong thuc DSR goc can,
     chuan=3). Script nay tu dong cong them 3.

Neu co du lieu that (wfv_predictions_fix_v4.csv + VNIndex_Features_v4.csv),
script se TU TINH T, skew, kurtosis, Sharpe truc tiep tu chuoi PnL that cua
chien luoc dau bang, thay vi go tay tu log cu (tranh sai so chep tay).
Neu khong co file, dung tam so lieu da duoc bao cao truoc do de doi chieu.

Dau vao (neu co) : wfv_predictions_fix_v4.csv, VNIndex_Features_v4.csv
"""

import pandas as pd
import numpy as np
import warnings
from scipy.stats import skew as sp_skew, kurtosis as sp_kurtosis, norm

warnings.filterwarnings('ignore')

# ============================== CAU HINH ==============================
N_TRIALS          = 60              # 20 dai loc x 3 model
TRANSACTION_COST  = 0.002
HEADLINE_MODEL    = 'LogReg'
HEADLINE_BAND     = (0.33, 0.67)
EULER_GAMMA       = 0.5772156649
PERIODS_PER_YEAR  = 252

print("=" * 78)
print("DEFLATED SHARPE RATIO (DSR) - Bailey & Lopez de Prado (2014)")
print("=" * 78)

# ============================== BUOC 1: LAY THAM SO ==============================
try:
    pred_df = pd.read_csv("wfv_predictions_final.csv", parse_dates=["Date"])
    df_v4   = pd.read_csv("VNIndex_Features_v4.csv", parse_dates=["Date"])
    pred_df = pred_df.sort_values("Date").reset_index(drop=True)
    df_v4   = df_v4.sort_values("Date").reset_index(drop=True)
    pred_df = pred_df.drop(columns=['VN_Return'], errors='ignore')
    pred_df = pred_df.merge(df_v4[['Date', 'VN_Return']], on='Date', how='left')
    ret_next = pred_df['VN_Return'].shift(-1).fillna(0).values

    prob = pred_df[f'prob_{HEADLINE_MODEL}'].values
    lo, hi = HEADLINE_BAND
    sigs = np.zeros(len(prob), dtype=int)
    pos = 0
    for i, p in enumerate(prob):
        if p > hi:
            pos = 1
        elif p < lo:
            pos = 0
        sigs[i] = pos
    pos_change = np.abs(np.diff(sigs, prepend=sigs[0]))
    daily_pnl = sigs * ret_next - pos_change * TRANSACTION_COST

    T = len(daily_pnl)
    skew_val = sp_skew(daily_pnl)
    excess_kurt_val = sp_kurtosis(daily_pnl)   # scipy default = EXCESS kurtosis (chuan = 0)
    SR_hat_annual = (daily_pnl.mean() * PERIODS_PER_YEAR) / (daily_pnl.std() * np.sqrt(PERIODS_PER_YEAR))

    print(f"\n[Du lieu that] Da tinh truc tiep tu {HEADLINE_MODEL} [{lo:.2f},{hi:.2f}]:")
    print(f"   T (so phien)        = {T}")
    print(f"   Skewness            = {skew_val:.4f}")
    print(f"   Excess Kurtosis     = {excess_kurt_val:.4f}")
    print(f"   Sharpe annualized   = {SR_hat_annual:.4f}")

except FileNotFoundError:
    print("\n[CANH BAO] Khong tim thay CSV that -> dung tam so lieu da bao cao truoc do de doi chieu.")
    T = 2314
    skew_val = -0.395
    excess_kurt_val = 8.650
    SR_hat_annual = 1.020
    print(f"   T={T}, skew={skew_val}, excess_kurt={excess_kurt_val}, SR_hat={SR_hat_annual}")


# ============================== BUOC 2: HAM DSR CHUAN ==============================
def deflated_sharpe_ratio(SR_hat_annual, T, skew_val, excess_kurt_val, N,
                           periods_per_year=PERIODS_PER_YEAR):
    """
    Tra ve dict gom SR* (nguong phat, annualized) va DSR (xac suat, 0-1).
    QUAN TRONG: moi phep tinh z-score deu thuc hien o don vi HANG NGAY
    (cung don vi voi T, skew, kurtosis), chi annualize lai SR* de bao cao.
    """
    pearson_kurt = excess_kurt_val + 3   # chuyen excess -> Pearson (chuan = 3)

    # --- SR_hat ve dang hang ngay ---
    SR_hat_daily = SR_hat_annual / np.sqrt(periods_per_year)

    # --- E[max SR_n] duoi H0 (SR=0 cho phan variance, quy uoc chuan) ---
    sigma_SR = np.sqrt(1 / (T - 1))
    z1 = norm.ppf(1 - 1/N)
    z2 = norm.ppf(1 - (1/N) * np.exp(-1))
    SR_star_daily = sigma_SR * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    SR_star_annual = SR_star_daily * np.sqrt(periods_per_year)

    # --- DSR z-score (PSR tai nguong SR_star, cung don vi hang ngay) ---
    denom = np.sqrt(1 - skew_val * SR_hat_daily + ((pearson_kurt - 1) / 4) * SR_hat_daily**2)
    z_dsr = (SR_hat_daily - SR_star_daily) * np.sqrt(T - 1) / denom
    dsr_prob = norm.cdf(z_dsr)

    return {
        'SR_hat_daily': SR_hat_daily,
        'SR_star_daily': SR_star_daily,
        'SR_star_annual': SR_star_annual,
        'z_dsr': z_dsr,
        'DSR': dsr_prob,
    }


# ============================== BUOC 3: TINH VA IN KET QUA ==============================
res = deflated_sharpe_ratio(SR_hat_annual, T, skew_val, excess_kurt_val, N_TRIALS)

print(f"\n[Ket qua DSR - N={N_TRIALS} lan thu]")
print(f"   SR* (nguong phat, annualized)  = {res['SR_star_annual']:.4f}")
print(f"   DSR z-score                     = {res['z_dsr']:.4f}")
print(f"   DSR (xac suat y nghia)          = {res['DSR']:.4f}  ({res['DSR']*100:.1f}%)")

if res['DSR'] >= 0.95:
    verdict = "DAT nguong chuan 95%"
elif res['DSR'] >= 0.90:
    verdict = "MARGINAL - gan nguong 90%, chua dat 95%"
else:
    verdict = "KHONG dat nguong y nghia thong ke thong thuong (95%)"
print(f"   Ket luan                        = {verdict}")

# ============ BUOC 4: MINH HOA LOI TRON DON VI (DE DOI CHIEU, KHONG DUNG) ============
print("\n[Doi chieu] Neu VO TINH dung SR_hat ANNUALIZED truc tiep vao cong thuc T-hang-ngay (loi tron don vi thuong gap):")
pearson_kurt = excess_kurt_val + 3
sigma_SR = np.sqrt(1 / (T - 1))
z1 = norm.ppf(1 - 1/N_TRIALS)
z2 = norm.ppf(1 - (1/N_TRIALS) * np.exp(-1))
SR_star_annual_ref = sigma_SR * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2) * np.sqrt(PERIODS_PER_YEAR)
denom_wrong = np.sqrt(1 - skew_val * SR_hat_annual + ((pearson_kurt - 1) / 4) * SR_hat_annual**2)
z_wrong = (SR_hat_annual - SR_star_annual_ref) * np.sqrt(T - 1) / denom_wrong
print(f"   DSR (SAI, minh hoa loi tron don vi) = {norm.cdf(z_wrong):.4f}  <- KHONG dung con so nay trong bai")

print("\n" + "=" * 78)
print(f"HOAN TAT. Dung dung SR*={res['SR_star_annual']:.3f} va DSR={res['DSR']:.3f} khi viet vao bai bao.")
print("=" * 78)
