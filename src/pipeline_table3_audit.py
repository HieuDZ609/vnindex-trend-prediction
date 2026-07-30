import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score

# Cấu hình tham số giao dịch chuẩn HOSE
TRANSACTION_COST = 0.002  # Phí ma sát 0.2% mỗi vòng khớp/đổi vị thế

print("====================================================================")
print("PHASE 6a AUDIT: ENGINE TÍNH TOÁN SHARPE, TURNOVER RATE VÀ MEAN AUC")
print("====================================================================")

# 1. NẠP DỮ LIỆU ĐẦU VÀO V4
if not os.path.exists("wfv_predictions_fix_v4.csv") or not os.path.exists("VNIndex_Features_v4.csv"):
    raise FileNotFoundError("🔴 Lỗi: Thiếu file dữ liệu v4. Hãy chạy file Feature Engineering trước!")

pred_df = pd.read_csv("wfv_predictions_fix_v4.csv", parse_dates=["Date"])
df_v4   = pd.read_csv("VNIndex_Features_v4.csv", parse_dates=["Date"])

# Đồng bộ hóa chuỗi lợi nhuận thực tế hàng ngày và giá VNIndex
pred_df = pred_df.drop(columns=['VN_Return'], errors='ignore')
pred_df = pred_df.merge(df_v4[['Date', 'VN_Return', 'Price_vs_MA20', 'Price_vs_MA50']], on='Date', how='left')
pred_df = pred_df.sort_values("Date").reset_index(drop=True)

# Khởi tạo mảng dữ liệu gốc
y_true = pred_df['y_true'].values
ret_next = pred_df['VN_Return'].shift(-1).fillna(0).values  # Causal Return ngày T+1
n_days = len(pred_df)

# 2. THUẬT TOÁN ĐÁNH GIÁ CHIẾN LƯỢC ĐỊNH LƯỢNG (CORE MATHEMATICS)
def evaluate_comprehensive_strategy(signals, returns_next, tc_rate=0.002):
    pos_change = np.abs(np.diff(signals, prepend=signals[0]))
    n_trades = int(pos_change.sum())
    turn_pct = (n_trades / len(signals)) * 100
    
    # Gross Sharpe
    strat_ret_gross = signals * returns_next
    ann_ret_gross = strat_ret_gross.mean() * 252
    ann_vol_gross = strat_ret_gross.std() * np.sqrt(252)
    sh_gross = ann_ret_gross / ann_vol_gross if ann_vol_gross > 1e-8 else 0.0
    
    # Net Sharpe (Đã trừ 0.2% phí HOSE)
    strat_ret_net = signals * returns_next - (pos_change * tc_rate)
    ann_ret_net = strat_ret_net.mean() * 252
    ann_vol_net = strat_ret_net.std() * np.sqrt(252)
    sh_net = ann_ret_net / ann_vol_net if ann_vol_net > 1e-8 else 0.0
    
    # Max Drawdown
    cum_net_ret = np.cumsum(strat_ret_net)
    cum_wealth = np.exp(cum_net_ret)
    running_max = np.maximum.accumulate(cum_wealth)
    drawdowns = (cum_wealth - running_max) / running_max
    max_dd = drawdowns.min()
    
    return sh_gross, sh_net, turn_pct, n_trades, max_dd

summary_records = []

# 3. CHẠY TÍNH TOÁN CHO CÁC MÔ HÌNH THÔ (RAW)
models_to_test = ['RF', 'XGBoost', 'LogReg']
filter_bands = {
    'RF': (0.42, 0.58),
    'XGBoost': (0.38, 0.62),
    'LogReg': (0.33, 0.67)
}

for name in models_to_test:
    prob = pred_df[f'prob_{name}'].values
    raw_signals = (prob >= 0.5).astype(int)
    auc_score = roc_auc_score(y_true, prob)
    sh_gross, sh_net, turn_pct, trades, max_dd = evaluate_comprehensive_strategy(raw_signals, ret_next, TRANSACTION_COST)
    summary_records.append({
        'Model Setup': f'Raw {name}', 
        'Mean_AUC': auc_score, 
        'Sharpe_Gross': sh_gross, 
        'Sharpe_Net': sh_net, 
        'Max_DD': max_dd, 
        'Trades': trades, 
        'Turn%': turn_pct
    })

# 4. CHẠY TÍNH TOÁN CHO CÁC MÔ HÌNH SẠCH (FILTERED)
for name in models_to_test:
    prob = pred_df[f'prob_{name}'].values
    band_lo, band_hi = filter_bands[name]
    
    # Tính trực tiếp auc_score cho từng mô hình trong vòng lặp Filter
    auc_score = roc_auc_score(y_true, prob)
    
    filtered_signals = np.zeros(n_days, dtype=int)
    current_pos = 0
    for i, p in enumerate(prob):
        if p > band_hi:
            current_pos = 1
        elif p < band_lo:
            current_pos = 0
        filtered_signals[i] = current_pos
        
    sh_gross, sh_net, turn_pct, trades, max_dd = evaluate_comprehensive_strategy(filtered_signals, ret_next, TRANSACTION_COST)
    summary_records.append({
        'Model Setup': f'{name} + Filter [{band_lo}, {band_hi}]', 
        'Mean_AUC': auc_score,  # Đã có auc_score chuẩn xác
        'Sharpe_Gross': sh_gross, 
        'Sharpe_Net': sh_net, 
        'Max_DD': max_dd, 
        'Trades': trades, 
        'Turn%': turn_pct
    })

# ====================================================================
# 5. TÍNH TOÁN BASELINE MA CROSSOVER (MA20 > MA50) 
# ====================================================================
raw_ma_pos = (pred_df['Price_vs_MA20'] < pred_df['Price_vs_MA50']).astype(int).values
ma_signals = np.roll(raw_ma_pos, 1)
ma_signals[0] = 0  

ma_auc = roc_auc_score(y_true[1:], ma_signals[1:])
sh_gross_ma, sh_net_ma, turn_ma, trades_ma, max_dd_ma = evaluate_comprehensive_strategy(
    ma_signals, ret_next, tc_rate=TRANSACTION_COST
)

summary_records.append({
    'Model Setup': 'MA Crossover Baseline (20/50)', 
    'Mean_AUC': ma_auc, 'Sharpe_Gross': sh_gross_ma, 
    'Sharpe_Net': sh_net_ma, 'Max_DD': max_dd_ma, 
    'Trades': trades_ma, 'Turn%': turn_ma
})

# Tín hiệu Persistence
persistence_signals = (pred_df['VN_Return'] > 0).astype(int).values
persistence_signals_shifted = np.roll(persistence_signals, 1)
auc_persistence = roc_auc_score(y_true[1:], persistence_signals_shifted[1:])
print(f"Persistence AUC: {auc_persistence:.3f}")

# 6. TÍNH TOÁN BASELINE BUY & HOLD
_, sh_bnh, _, _, max_dd_bnh = evaluate_comprehensive_strategy(np.ones(n_days, dtype=int), ret_next, tc_rate=0.0)
summary_records.append({
    'Model Setup': 'Buy & Hold Baseline', 'Mean_AUC': np.nan, 'Sharpe_Gross': sh_bnh, 
    'Sharpe_Net': sh_bnh, 'Max_DD': max_dd_bnh, 'Trades': 1, 'Turn%': 0.0
})

# 7. IN KẾT QUẢ ĐỐI CHIẾU SIÊU TƯỜNG MINH
audit_df = pd.DataFrame(summary_records)
print("\n" + "="*122)
print("📋 KẾT QUẢ KIỂM TOÁN THỰC NGHIỆM: CHI TIẾT HIỆU NĂNG MÔ HÌNH VÀ CHIẾN LƯỢC BỘ LỌC")
print("="*122)
print(f"{'Cấu Hình Hệ Thống':<32} | {'Mean AUC':<8} | {'Sharpe Gross':<12} | {'Sharpe Net':<10} | {'Max DD':<9} | {'N.Trades':<8} | {'Turnover/Day':<12}")
print("-" * 122)
for _, row in audit_df.iterrows():
    auc_str = f"{row['Mean_AUC']:>8.3f}" if not pd.isna(row['Mean_AUC']) else f"{'---':>8}"
    print(f"{row['Model Setup']:<32} | {auc_str} | {row['Sharpe_Gross']:>12.3f} | {row['Sharpe_Net']:>10.3f} | {row['Max_DD']:>9.3f} | {row['Trades']:>8} | {row['Turn%']:>11.1f}%")
print("="*122)

audit_df.to_csv("grid_filters_final_audit.csv", index=False)
print("💾 Đã lưu cấu trúc kiểm toán thực tế tại file: grid_filters_final_audit.csv")