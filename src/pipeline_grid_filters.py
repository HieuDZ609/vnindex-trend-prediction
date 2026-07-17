"""
pipeline_grid_filters.py
========================
Quét hàng loạt (Grid Search) 20 bộ lọc dải trung tính cho RF, XGBoost, LogReg.
Bổ sung kiểm toán: Sharpe_gross (Chưa trừ phí) và Turn% (Tỷ lệ vòng quay/ngày).

Đầu vào local : wfv_predictions_fix_v4.csv VÀ VNIndex_Features_v4.csv
Đầu ra local : Bảng ma trận so sánh nâng cao grid_filters_results.csv
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

TRANSACTION_COST = 0.002  # Phí giao dịch 0.2% áp dụng tại HOSE

print("====================================================================")
print("START OPERATION: GRID SEARCH 20 FILTERS WITH TURN% & GROSS SHARPE")
print("====================================================================")

# ====================================================================
# BƯỚC 1: NẠP DỮ LIỆU ĐÃ DỰ ĐOÁN
# ====================================================================
try:
    pred_df = pd.read_csv("wfv_predictions_fix_v4.csv", parse_dates=["Date"])
    df_v4   = pd.read_csv("VNIndex_Features_v4.csv", parse_dates=["Date"])
except FileNotFoundError:
    raise FileNotFoundError("Thiếu tệp wfv_predictions_fix_v4.csv hoặc VNIndex_Features_v4.csv. Hãy chạy feature_engineering_v4.py trước.")

pred_df = pred_df.sort_values("Date").reset_index(drop=True)
df_v4   = df_v4.sort_values("Date").reset_index(drop=True)

# Đồng bộ lợi nhuận thực tế hàng ngày
pred_df = pred_df.drop(columns=['VN_Return'], errors='ignore')
pred_df = pred_df.merge(df_v4[['Date', 'VN_Return']], on='Date', how='left')

ret_next = pred_df['VN_Return'].shift(-1).fillna(0).values  # Causal Return ngày T+1
n_days = len(pred_df)

# ====================================================================
# BƯỚC 2: ĐỊNH NGHĨA HÀM TÍNH TOÁN CHI TIẾT (GROSS/NET SHARPE + TURN%)
# ====================================================================
def evaluate_comprehensive_strategy(signals, returns_next, tc_rate=0.002):
    # Tính số lần đảo vị thế (0->1 hoặc 1->0)
    pos_change = np.abs(np.diff(signals, prepend=signals[0]))
    n_trades = int(pos_change.sum())
    
    # 1. Tính Turnover Rate (%) trung bình mỗi phiên
    # Định nghĩa Turn% = (Tổng số lần đổi vị thế / Tổng số phiên) * 100
    turn_pct = (n_trades / len(signals)) * 100
    
    # 2. Tính Gross Performance (Chưa trừ phí)
    strat_ret_gross = signals * returns_next
    ann_ret_gross = strat_ret_gross.mean() * 252
    ann_vol_gross = strat_ret_gross.std() * np.sqrt(252)
    sh_gross = ann_ret_gross / ann_vol_gross if ann_vol_gross > 1e-8 else 0.0
    
    # 3. Tính Net Performance (Đã trừ phí 0.2%)
    strat_ret_net = signals * returns_next - (pos_change * tc_rate)
    ann_ret_net = strat_ret_net.mean() * 252
    ann_vol_net = strat_ret_net.std() * np.sqrt(252)
    sh_net = ann_ret_net / ann_vol_net if ann_vol_net > 1e-8 else 0.0
    
    return sh_gross, sh_net, turn_pct, n_trades

# ====================================================================
# BƯỚC 3: ĐỊNH NGHĨA DANH SÁCH 20 BỘ LỌC VÙNG TRUNG TÍNH (LO, HI)
# ====================================================================
filter_bands = [
    (0.50, 0.50),  # 1. Không lọc (Raw mô hình gốc)
    (0.49, 0.51), (0.48, 0.52), (0.47, 0.53), (0.46, 0.54),
    (0.45, 0.55), (0.44, 0.56), (0.43, 0.57), (0.42, 0.58), (0.41, 0.59),
    (0.40, 0.60),  # 11. Dải gốc trong paper
    (0.39, 0.61), (0.38, 0.62), (0.37, 0.63), (0.36, 0.64),
    (0.35, 0.65), (0.34, 0.66), (0.33, 0.67), (0.32, 0.68), (0.30, 0.70)
]

# ====================================================================
# BƯỚC 4: KHỞI CHẠY QUÉT GRID SEARCH MATRIX
# ====================================================================
print(f"📡 Đang quét qua 20 bộ lọc với hệ thống số liệu mở rộng...")

grid_records = []

for band_lo, band_hi in filter_bands:
    band_label = f"Raw (0.5)" if band_lo == 0.5 else f"[{band_lo:.2f}, {band_hi:.2f}]"
    row_record = {'Filter Band': band_label}
    
    for name in ['RF', 'XGBoost', 'LogReg']:
        prob = pred_df[f'prob_{name}'].values
        signals = np.zeros(n_days, dtype=int)
        current_pos = 0
        
        for i, p in enumerate(prob):
            if p > band_hi:
                current_pos = 1
            elif p < band_lo:
                current_pos = 0
            signals[i] = current_pos
            
        sh_gross, sh_net, turn_pct, trades = evaluate_comprehensive_strategy(signals, ret_next, TRANSACTION_COST)
        
        row_record[f'{name}_ShGross'] = sh_gross
        row_record[f'{name}_ShNet']   = sh_net
        row_record[f'{name}_Turn%']   = turn_pct
        row_record[f'{name}_Trades']  = trades
        
    grid_records.append(row_record)

grid_df = pd.DataFrame(grid_records)

# Tính Buy & Hold làm mốc chuẩn so sánh (Phí = 0)
bnh_signals = np.ones(n_days)
_, sh_bnh, _, _ = evaluate_comprehensive_strategy(bnh_signals, ret_next, tc_rate=0.0)

# ====================================================================
# BƯỚC 5: XUẤT BẢNG IN TỔNG HỢP SIÊU SCANNABLE
# ====================================================================
print("\n" + "="*115)
print("📋 BẢNG KIỂM TOÁN NÂNG CAO: GRID SEARCH METRICS (TURN% + SHARPE GROSS VS NET)")
print("="*115)
print(f"{'Dải Lọc':<13} | {'RF (Gross / Net / Turn%)':<27} | {'XGBoost (Gross / Net / Turn%)':<29} | {'LogReg (Gross / Net / Turn%)':<27}")
print("-" * 115)

for _, row in grid_df.iterrows():
    rf_txt  = f"{row['RF_ShGross']:>5.2f} / {row['RF_ShNet']:>5.2f} / {row['RF_Turn%']:>4.1f}%"
    xgb_txt = f"{row['XGBoost_ShGross']:>5.2f} / {row['XGBoost_ShNet']:>5.2f} / {row['XGBoost_Turn%']:>4.1f}%"
    lr_txt  = f"{row['LogReg_ShGross']:>5.2f} / {row['LogReg_ShNet']:>5.2f} / {row['LogReg_Turn%']:>4.1f}%"
    
    # Đánh dấu sao nếu Net Sharpe ăn được Buy & Hold
    if row['RF_ShNet'] > sh_bnh: rf_txt += "★"
    if row['XGBoost_ShNet'] > sh_bnh: xgb_txt += "★"
    if row['LogReg_ShNet'] > sh_bnh: lr_txt += "★"
    
    print(f"{row['Filter Band']:<13} | {rf_txt:<27} | {xgb_txt:<29} | {lr_txt:<27}")

print("-" * 115)
print(f"Mốc so sánh đối chiếu: Buy & Hold Baseline Sharpe = {sh_bnh:.3f} (Turn% = 0.0%)")
print("Cấu trúc cột: Sharpe_Gross (Chưa phí) / Sharpe_Net (Đã trừ phí) / Turn% (Tỷ lệ đảo vị thế ngày)")
print("=" * 115)

# Lưu tệp CSV local mới đầy đủ cột
grid_df.to_csv("grid_filters_results_advanced.csv", index=False)
print("💾 Đã xuất tệp nâng cao thành công: grid_filters_results_advanced.csv")