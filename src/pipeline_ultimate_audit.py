"""
pipeline_ultimate_audit.py
==========================
Hệ thống kiểm toán tối hậu gồm 6 bài toán tra tấn áp lực chuyên sâu 
dành cho file tín hiệu cuối cùng wfv_predictions_FINAL.csv.

Các bài toán:
1. Fold-by-Fold Stability (Độ bền 16 Folds)
2. Transaction Cost Sensitivity (Ngưỡng chịu phí 0.0% -> 0.5%)
3. Fat-Tail & Risk Metrics (Sortino, Calmar, Skewness, Kurtosis)
4. Position Holding Duration (Thống kê chuỗi lệnh và thời gian ôm vị thế)
5. Block-Permutation Bootstrap (Xáo trộn khối vị thế ngẫu nhiên)
6. Market Regime Asymmetry (Hiệu năng Uptrend vs Downtrend qua MA200)
"""

import pandas as pd
import numpy as np
import warnings
from scipy.stats import skew, kurtosis

warnings.filterwarnings('ignore')

print("====================================================================")
print("START OPERATION: ULTIMATE 6-STAGE AUDIT FOR FINAL PREDICTIONS")
print("====================================================================")

# ====================================================================
# BƯỚC 0: NẠP DỮ LIỆU FINAL VÀ ĐỒNG BỘ CAUSAL RETURN
# ====================================================================
try:
    pred_df = pd.read_csv("wfv_predictions_FINAL.csv", parse_dates=["Date"])
except FileNotFoundError:
    raise FileNotFoundError("Thiếu tệp wfv_predictions_FINAL.csv. Hãy chạy pipeline_backtest_final.py trước.")

pred_df = pred_df.sort_values("Date").reset_index(drop=True)
ret_next = pred_df['VN_Return'].shift(-1).fillna(0).values  # Causal T+1 Return
n_days = len(pred_df)

models = ['RF', 'XGBoost', 'LogReg']

# ====================================================================
# TEST 1: FOLD-BY-FOLD STABILITY TEST (ĐỘ BỀN THEO GIAI ĐOẠN)
# ====================================================================
print("\n🚨 TEST 1: Kiểm định độ bền kiểm thử chéo (16 Folds Stability)")
print("-" * 80)
# Chia chuỗi dữ liệu thành 16 phần bằng nhau tương ứng 16 Walk-Forward Folds
fold_size = n_days // 16
for name in models:
    sigs = pred_df[f'pred_{name}_filtered'].values
    fold_sharpes = []
    
    for f in range(16):
        idx_start = f * fold_size
        idx_end = (f + 1) * fold_size if f < 15 else n_days
        
        f_sigs = sigs[idx_start:idx_end]
        f_ret = ret_next[idx_start:idx_end]
        
        pos_change = np.abs(np.diff(f_sigs, prepend=f_sigs[0]))
        f_strat_ret = f_sigs * f_ret - (pos_change * 0.002)
        
        if f_strat_ret.std() > 1e-6:
            f_sr = (f_strat_ret.mean() * 252) / (f_strat_ret.std() * np.sqrt(252))
        else:
            f_sr = 0.0
        fold_sharpes.append(f_sr)
        
    fold_sharpes = np.array(fold_sharpes)
    print(f"   ➔ {name:<7} | Net Sharpe Trung bình Folds: {fold_sharpes.mean():>6.3f} | Độ lệch chuẩn (±Std): {fold_sharpes.std():.3f}")

# ====================================================================
# TEST 2: TRANSACTION COST SENSITIVITY TEST (SỰ NHẠY CẢM VỚI PHÍ GIAO DỊCH)
# ====================================================================
print("\n🚨 TEST 2: Thử nghiệm chi phí biến thiên (Transaction Cost Sensitivity)")
print("-" * 80)
costs_to_test = [0.000, 0.001, 0.002, 0.003, 0.005]
print(f"{'Mô hình':<10} | {'Phí 0.0%':<10} | {'Phí 0.1%':<10} | {'Phí 0.2% (Gốc)':<15} | {'Phí 0.3%':<10} | {'Phí 0.5%':<10}")
print("-" * 80)

for name in models:
    sigs = pred_df[f'pred_{name}_filtered'].values
    pos_change = np.abs(np.diff(sigs, prepend=sigs[0]))
    row_str = f"{name:<10} | "
    
    for tc in costs_to_test:
        strat_ret = sigs * ret_next - (pos_change * tc)
        sh = (strat_ret.mean() * 252) / (strat_ret.std() * np.sqrt(252)) if strat_ret.std() > 1e-6 else 0.0
        row_str += f"{sh:>8.3f}   | "
    print(row_str)

# ====================================================================
# TEST 3: FAT-TAIL & RISK METRICS AUDIT (RỦI RO ĐUÔI BÉO NÂNG CAO)
# ====================================================================
print("\n🚨 TEST 3: Phân tích phân phối đuôi béo và tỷ lệ rủi ro nâng cao (TC=0.2%)")
print("-" * 90)
print(f"{'Mô hình':<10} | {'Sortino Ratio':<15} | {'Calmar Ratio':<15} | {'Skewness':<12} | {'Kurtosis':<12}")
print("-" * 90)

for name in models:
    sigs = pred_df[f'pred_{name}_filtered'].values
    pos_change = np.abs(np.diff(sigs, prepend=sigs[0]))
    strat_ret = sigs * ret_next - (pos_change * 0.002)
    
    # 1. Sortino Ratio (Chỉ tính Volatility cho các ngày có lợi nhuận âm)
    downside_ret = strat_ret[strat_ret < 0]
    downside_vol = downside_ret.std() * np.sqrt(252) if len(downside_ret) > 0 else 1e-6
    ann_ret = strat_ret.mean() * 252
    sortino = ann_ret / downside_vol
    
    # 2. Calmar Ratio
    equity = np.cumprod(1 + strat_ret)
    max_dd = (equity / np.maximum.accumulate(equity) - 1).min()
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-6 else 0.0
    
    # 3. Độ lệch và độ nhọn phân phối
    sk = skew(strat_ret)
    kt = kurtosis(strat_ret)
    
    print(f"{name:<10} | {sortino:>13.3f}   | {calmar:>13.3f}   | {sk:>10.3f}   | {kt:>10.3f}")

# ====================================================================
# TEST 4: POSITION HOLDING DURATION TEST (THỐNG KÊ CHUỖI VỊ THẾ)
# ====================================================================
print("\n🚨 TEST 4: Kiểm định vị thế liên tục và chuỗi lệnh nắm giữ")
print("-" * 85)
print(f"{'Mô hình':<10} | {'Avg Hold Days':<15} | {'Cash Days (%)':<15} | {'Max Consecutive Wins':<22} | {'Max Consecutive Losses'}")
print("-" * 85)

for name in models:
    sigs = pred_df[f'pred_{name}_filtered'].values
    pos_change = np.abs(np.diff(sigs, prepend=sigs[0]))
    strat_ret = sigs * ret_next - (pos_change * 0.002)
    
    # Tính số ngày nắm giữ trung bình mỗi lệnh
    cash_days = np.sum(sigs == 0)
    cash_pct = (cash_days / n_days) * 100
    
    # Bóc tách độ dài chuỗi lệnh liên tiếp
    holding_periods = []
    current_hold = 0
    for s in sigs:
        if s == 1:
            current_hold += 1
        else:
            if current_hold > 0:
                holding_periods.append(current_hold)
                current_hold = 0
    if current_hold > 0: holding_periods.append(current_hold)
    avg_hold = np.mean(holding_periods) if len(holding_periods) > 0 else 0.0
    
    # Chuỗi thắng/thua liên tiếp dài nhất (Chỉ tính các ngày có vị thế Long)
    trade_days_ret = strat_ret[sigs == 1]
    max_win_streak = 0
    max_loss_streak = 0
    curr_win = 0
    curr_loss = 0
    
    for r in trade_days_ret:
        if r > 0:
            curr_win += 1
            max_loss_streak = max(max_loss_streak, curr_loss)
            curr_loss = 0
        elif r < 0:
            curr_loss += 1
            max_win_streak = max(max_win_streak, curr_win)
            curr_win = 0
            
    max_win_streak = max(max_win_streak, curr_win)
    max_loss_streak = max(max_loss_streak, curr_loss)
    
    print(f"{name:<10} | {avg_hold:>13.1f} phiên | {cash_pct:>13.1f}%   | {max_win_streak:>20} ngày   | {max_loss_streak:>21} ngày")

# ====================================================================
# TEST 5: BLOCK-PERMUTATION BOOTSTRAP TEST (XÁO TRỘN KHỐI VỊ THẾ)
# ====================================================================
print("\n🚨 TEST 5: Mô phỏng Monte Carlo xáo trộn khối vị thế (Block-Permutation)")
print("-" * 80)
np.random.seed(42)
block_size = 10  # Cắt chuỗi thành các khối 10 ngày để giữ nguyên tính liên tục của hệ thống lệnh

for name in models:
    sigs = pred_df[f'pred_{name}_filtered'].values
    pos_change = np.abs(np.diff(sigs, prepend=sigs[0]))
    total_tc_cost = (pos_change * 0.002).sum()
    
    # Lấy Sharpe thực tế để đối chiếu
    actual_ret = sigs * ret_next - (pos_change * 0.002)
    actual_sr = (actual_ret.mean() * 252) / (actual_ret.std() * np.sqrt(252)) if actual_ret.std() > 1e-6 else 0.0
    
    # Tách chuỗi return thành các khối liên tục
    n_blocks = n_days // block_size
    blocks = [ret_next[i*block_size : (i+1)*block_size] for i in range(n_blocks)]
    if n_days % block_size != 0:
        blocks.append(ret_next[n_blocks*block_size:])
        
    block_perm_sharpes = []
    for _ in range(500):
        # Xáo trộn ngẫu nhiên thứ tự các khối
        np.random.shuffle(blocks)
        shuffled_ret = np.concatenate(blocks)
        
        strat_ret = sigs * shuffled_ret
        ann_ret = strat_ret.mean() * 252 - (total_tc_cost / n_days * 252)
        ann_vol = strat_ret.std() * np.sqrt(252)
        block_perm_sharpes.append(ann_ret / ann_vol if ann_vol > 1e-6 else 0.0)
        
    block_perm_sharpes = np.array(block_perm_sharpes)
    p_val_block = (block_perm_sharpes >= actual_sr).mean()
    print(f"   ➔ {name:<7} | Net Sharpe Thực tế: {actual_sr:>5.3f} | Kỳ vọng nhiễu khối (Mean): {block_perm_sharpes.mean():>6.3f} | p-value khối: {p_val_block:.4f}")

# ====================================================================
# TEST 6: MARKET REGIME ASYMMETRY TEST (KIỂM ĐỊNH TRẠNG THÁI KHÁC BIỆT MA200)
# ====================================================================
print("\n🚨 TEST 6: Kiểm định tính bất đối xứng trạng thái thị trường (Regime Asymmetry qua MA200)")
print("-" * 95)
print(f"{'Mô hình':<10} | {'Hit Rate Uptrend (VN > MA200)':<30} | {'Hit Rate Downtrend (VN < MA200)':<30}")
print("-" * 95)

# Giả lập lại chuỗi giá đóng cửa để tính đường MA200
df_regime = pd.DataFrame({'Date': pred_df['Date'], 'VN_Return': pred_df['VN_Return']})
df_regime['Price'] = 1000 * (1 + df_regime['VN_Return'].fillna(0)).cumprod()
df_regime['MA200'] = df_regime['Price'].rolling(200).mean()

# Gán trạng thái thị trường tại ngày phát tín hiệu
is_uptrend = (df_regime['Price'] >= df_regime['MA200']).values

for name in models:
    sigs = pred_df[f'pred_{name}_filtered'].values
    
    # Định nghĩa ngày đoán đúng xu hướng khi Long: s=1 và return ngày mai > 0.001 (Theo Tau của Problem Formulation)
    correct_long = (sigs == 1) & (ret_next > 0.001)
    wrong_long   = (sigs == 1) & (ret_next <= 0.001)
    
    # Phân tách theo môi trường Uptrend
    up_correct = np.sum(correct_long & is_uptrend)
    up_total   = np.sum((sigs == 1) & is_uptrend)
    hit_rate_up = (up_correct / up_total * 100) if up_total > 0 else 0.0
    
    # Phân tách theo môi trường Downtrend
    down_correct = np.sum(correct_long & ~is_uptrend)
    down_total   = np.sum((sigs == 1) & ~is_uptrend)
    hit_rate_down = (down_correct / down_total * 100) if down_total > 0 else 0.0
    
    print(f"{name:<10} | {hit_rate_up:>5.1f}% (Đúng {up_correct}/{up_total} ngày phát lệnh Long)       | {hit_rate_down:>5.1f}% (Đúng {down_correct}/{down_total} ngày phát lệnh Long)")

print("\n" + "="*80)
print("🏁 HỆ THỐNG KIỂM TOÁN TỐI HẬU ĐÃ HOÀN TẤT TOÀN BỘ 6 BÀI TRA TẤN!")
print("====================================================================")