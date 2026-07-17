"""
pipeline_stress_test_optimized.py
=================================
Hệ thống kiểm toán tối hậu dành riêng cho 3 mốc bộ lọc tối ưu (Grid Search Peak).
Mục tiêu: Bóc trần xem mốc Sharpe 1.020, 0.734 và 0.729 là giá trị thực hay lỗi logic.

Chạy 3 bài test: Ma trận Causal, Monte Carlo Return-Shuffling và Thống kê drawdown chi tiết.
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

TRANSACTION_COST = 0.002

print("====================================================================")
print("START OPERATION: STRESS TESTING THE OPTIMIZED PEAK FILTERS")
print("====================================================================")

# ====================================================================
# BƯỚC 1: NẠP DỮ LIỆU ĐÃ DỰ ĐOÁN
# ====================================================================
try:
    pred_df = pd.read_csv("wfv_predictions_final.csv", parse_dates=["Date"])
    df_v4   = pd.read_csv("VNIndex_Features_v4.csv", parse_dates=["Date"])
except FileNotFoundError:
    raise FileNotFoundError("Thiếu tệp dữ liệu sạch từ các bước trước.")

pred_df = pred_df.sort_values("Date").reset_index(drop=True)
df_v4   = df_v4.sort_values("Date").reset_index(drop=True)

pred_df = pred_df.drop(columns=['VN_Return'], errors='ignore')
pred_df = pred_df.merge(df_v4[['Date', 'VN_Return']], on='Date', how='left')

ret_next = pred_df['VN_Return'].shift(-1).fillna(0).values  # T+1 Return
n_days = len(pred_df)

# Cấu hình đỉnh tối ưu thu được từ Grid Search
best_configs = {
    'RF':      {'band_lo': 0.48, 'band_hi': 0.52, 'expected_sr': 0.729},
    'XGBoost': {'band_lo': 0.41, 'band_hi': 0.59, 'expected_sr': 0.734},
    'LogReg':  {'band_lo': 0.33, 'band_hi': 0.67, 'expected_sr': 1.020}
}

# ====================================================================
# HÀM TÍNH TOÁN HIỆU NĂNG VÀ ĐƯỜNG CONG VỐN CHUẨN
# ====================================================================
def run_backtest(signals, returns_next):
    pos_change = np.abs(np.diff(signals, prepend=signals[0]))
    strat_ret = signals * returns_next - (pos_change * TRANSACTION_COST)
    equity = np.cumprod(1 + strat_ret)
    
    ann_ret = strat_ret.mean() * 252
    ann_vol = strat_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-8 else 0.0
    max_dd = (equity / np.maximum.accumulate(equity) - 1).min()
    return sharpe, max_dd, int(pos_change.sum()), signals

# Sinh tín hiệu thực tế cho từng mô hình dựa trên mốc tối ưu tương ứng
actual_signals = {}
for name, cfg in best_configs.items():
    prob = pred_df[f'prob_{name}'].values
    sigs = np.zeros(n_days, dtype=int)
    current_pos = 0
    for i, p in enumerate(prob):
        if p > cfg['band_hi']:
            current_pos = 1
        elif p < cfg['band_lo']:
            current_pos = 0
        sigs[i] = current_pos
    actual_signals[name] = sigs

# ====================================================================
# TEST 1: KIỂM TOÁN MA TRẬN CAUSAL TRÊN TÍN HIỆU SAU LỌC
# ====================================================================
print("\n🚨 TEST 1: Kiểm toán ma trận tương quan của chuỗi TÍN HIỆU CUỐI CÙNG")
print("-" * 75)
print(f"{'Mô hình tối ưu':<15} | {'T-1 (Quá khứ)':<15} | {'T+0 (Cùng ngày)':<15} | {'T+1 (Ngày mai - Đích)'}")
print("-" * 75)
for name, sigs in actual_signals.items():
    c_minus1 = pd.Series(sigs).corr(pred_df['VN_Return'].shift(1))
    c_zero   = pd.Series(sigs).corr(pred_df['VN_Return'])
    c_plus1  = pd.Series(sigs).corr(pred_df['VN_Return'].shift(-1))
    
    status = "⚠️ NGUY CƠ OVERFIT" if abs(c_zero) > abs(c_plus1) else "✅ CAUSAL CHUẨN"
    print(f"{name:<15} | {c_minus1:>+15.4f} | {c_zero:>+15.4f} | {c_plus1:>+20.4f}  ➔ {status}")

# ====================================================================
# TEST 2: MONTE CARLO RETURN-SHUFFLING (PHÁ BỎ BIAS ĂN MAY)
# ====================================================================
print("\n🎲 TEST 2: Khởi chạy Monte Carlo Permutation (500 vòng xáo trộn dòng thời gian)...")
np.random.seed(42)

for name, sigs in actual_signals.items():
    sh_actual, mdd_actual, trades, _ = run_backtest(sigs, ret_next)
    
    # Tính tổng chi phí cố định để phạt tập ngẫu nhiên công bằng
    pos_change = np.abs(np.diff(sigs, prepend=sigs[0]))
    total_tc_cost = (pos_change * TRANSACTION_COST).sum()
    
    perm_sharpes = []
    for _ in range(500):
        shuffled_ret = np.random.permutation(ret_next)
        strat_ret = sigs * shuffled_ret
        ann_ret = strat_ret.mean() * 252 - (total_tc_cost / n_days * 252)
        ann_vol = strat_ret.std() * np.sqrt(252)
        perm_sharpes.append(ann_ret / ann_vol if ann_vol > 1e-8 else 0.0)
        
    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= sh_actual).mean()
    p95 = np.percentile(perm_sharpes, 95)
    
    print(f"   ➔ {name:<7} (Dải {str(best_configs[name]['band_lo'])}-{str(best_configs[name]['band_hi'])}):")
    print(f"         + Sharpe thực tế tính ra: {sh_actual:.3f} | Số lệnh: {trades} | Max Drawdown: {mdd_actual:.1%}")
    print(f"         + Nhiễu ngẫu nhiên kỳ vọng: Mean = {perm_sharpes.mean():.3f} ± {perm_sharpes.std():.3f} | Ngưỡng trần 95%: {p95:.3f}")
    print(f"         + GIÁ TRỊ P-VALUE: p = {p_value:.4f} ➔ {'✅ ĐẠT Ý NGHĨA KHÁCH QUAN' if p_value < 0.01 else '🔴 COI CHỪNG DATA SNOOPING'}")
    print("-" * 75)

print("\n====================================================================")
print("🏁 HỆ THỐNG TRA TẤN MỐC ĐỈNH HOÀN TẤT!")
print("====================================================================")