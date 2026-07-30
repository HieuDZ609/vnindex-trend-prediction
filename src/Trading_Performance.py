import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os

warnings.filterwarnings('ignore')

print("====================================================================")
print("START RUNNING: AUTOMATIC SHARPE CALCULATOR FROM OUT-OF-SAMPLE LOGS")
print("====================================================================")

file_pred = "wfv_predictions_FINAL.csv"
file_feat = "VNIndex_Features_v4.csv"

# 1. Kiểm tra sự tồn tại của file trước khi xử lý
if not os.path.exists(file_pred) or not os.path.exists(file_feat):
    raise FileNotFoundError("🔴 Lỗi: Thiếu file wfv_predictions_FINAL.csv hoặc VNIndex_Features_v4.csv trong thư mục!")

# 2. ĐỌC VÀ LÀM SẠCH DỮ LIỆU ĐẦU VÀO ĐỘNG
pred_df = pd.read_csv(file_pred, parse_dates=["Date"])
df_feat = pd.read_csv(file_feat, parse_dates=["Date"])

pred_df = pred_df.sort_values("Date").reset_index(drop=True)
pred_df = pred_df.drop(columns=['VN_Return'], errors='ignore')
pred_df = pred_df.merge(df_feat[['Date', 'VN_Return']], on='Date', how='left')
pred_df['Year'] = pred_df['Date'].dt.year

# Đồng bộ toán học: Lấy chuỗi tỷ suất sinh lời dịch chuyển T+1
ret_next = pred_df['VN_Return'].shift(-1).fillna(0).values

# 3. HÀM CỐT LÕI TÍNH ANNUAL SHARPE KHỚP 100% logic VỚI CELL 23_FIX
def annual_sharpe_fixed(sig, ret_arr):
    sig = np.array(sig)
    # Tính chi phí giao dịch ròng 0.2% dựa trên hành vi đảo vị thế lệnh
    cost = np.abs(np.diff(sig, prepend=sig[0])) * 0.002
    daily_ret = sig * ret_arr - cost

    ann_ret = daily_ret.mean() * 252
    ann_vol = daily_ret.std() * np.sqrt(252)
    return ann_ret / ann_vol if ann_vol > 1e-8 else 0.0

results_yr = []
years = sorted(pred_df['Year'].unique())

# Duyệt qua từng năm để bóc tách dữ liệu tính toán
for yr in years:
    mask = pred_df['Year'] == yr
    ret_yr = ret_next[mask]
    if len(ret_yr) < 50:
        continue

    sig_lrf = pred_df.loc[mask, 'pred_LogReg_filtered'].values
    sig_bnh = np.ones(len(ret_yr))
    if 'MA_Signal' in pred_df.columns:
        sig_ma = pred_df.loc[mask, 'MA_Signal'].values
    else:
        sig_ma = (df_feat.loc[df_feat['Date'].dt.year == yr, 'Price_vs_MA20'] > 0).astype(int).values[:len(ret_yr)]

    results_yr.append({
        'Year': int(yr),
        'LogReg+Filter': annual_sharpe_fixed(sig_lrf, ret_yr),
        'MA Crossover':  annual_sharpe_fixed(sig_ma,  ret_yr),
        'Buy & Hold':    annual_sharpe_fixed(sig_bnh, ret_yr),
        'Market ann%':   ret_yr.mean() * 252 * 100
    })

yr_df = pd.DataFrame(results_yr)

# ====================================================================
# 4. IN MA TRẬN KẾT QUẢ ĐẦU RA RA TERMINAL
# ====================================================================
print("\nYEAR-BY-YEAR SHARPE — EXTRACTED & FIXED (mean×252 / std×√252)")
print("=" * 68)
print(f"{'Year':>6} {'LogReg+Filter':>14} {'MA Crossover':>14} {'Buy & Hold':>12} {'Market%':>9}")
print("-" * 68)
for _, row in yr_df.iterrows():
    flag = "★" if row['LogReg+Filter'] > row['Buy & Hold'] else " "
    print(f"{int(row['Year']):>6} {row['LogReg+Filter']:>14.3f} {row['MA Crossover']:>14.3f} {row['Buy & Hold']:>12.3f} {row['Market ann%']:>8.1f}% {flag}")
print("=" * 68)

beat = (yr_df['LogReg+Filter'] > yr_df['Buy & Hold']).sum()
print(f"Beats B&H: {beat}/{len(yr_df)} years ({beat/len(yr_df)*100:.0f}%)\n")

# ====================================================================
# 5. VẼ ĐỒ THỊ KÉP (DÙNG constrained_layout ĐỂ TRÁNH LỖI TIGHT_LAYOUT)
# ====================================================================
fig, axes = plt.subplots(2, 1, figsize=(11, 12), layout='constrained')
fig.patch.set_facecolor('white')

x = np.arange(len(yr_df))
w = 0.26
yr_labels = [str(int(y)) for y in yr_df['Year']]
C = {'logreg':'#1D9E75', 'ma':'#5DCAA5', 'bnh':'#B4B2A9', 'pos':'#1D9E75', 'neg':'#D85A30'}

# --- PANEL E — ANNUAL SHARPE RATIO BY YEAR ---
ax = axes[0]
rects1 = ax.bar(x - w, yr_df['LogReg+Filter'], w, color=C['logreg'], alpha=0.9, label='LogReg+Filter')
rects2 = ax.bar(x,     yr_df['MA Crossover'],  w, color=C['ma'],     alpha=0.9, label='MA Crossover')
rects3 = ax.bar(x + w, yr_df['Buy & Hold'],    w, color=C['bnh'],    alpha=0.9, label='Buy & Hold')
ax.axhline(0, color='black', lw=0.8, alpha=0.5)

if 2022 in list(yr_df['Year']):
    i22 = list(yr_df['Year']).index(2022)
    ax.axvspan(i22 - 0.45, i22 + 0.45, alpha=0.08, color='red')
    ax.text(i22, ax.get_ylim()[1] - 0.6, '2022\nCrash', ha='center', va='top', fontsize=10, color='red', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(yr_labels, fontsize=11, fontweight='bold')
ax.set_title('(E) Annual Sharpe Ratio Breakdown (TC = 0.2%)', fontsize=13, fontweight='bold', pad=10)
ax.set_ylabel('Sharpe Ratio (Annualized)', fontsize=12)
ax.set_ylim(-2.0, 5.0)
ax.legend(fontsize=10, loc='lower left', frameon=True, facecolor='white', edgecolor='#BDC3C7')
ax.grid(True, alpha=0.3, linestyle=':', axis='y')

# --- PANEL F — DELTA SHARPE (LÀM TRÒN LÊN 3 CHỮ SỐ ĐỂ KHÓP ĐẠI SỐ 100%) ---
ax2 = axes[1]
# Fix lệch 0.001 bằng cách lấy giá trị đã hiển thị trên Panel E để trừ
sharpe_lrf = np.round(yr_df['LogReg+Filter'], 3)
sharpe_bnh = np.round(yr_df['Buy & Hold'], 3)
diff = sharpe_lrf - sharpe_bnh

bar_colors = [C['pos'] if v > 0 else C['neg'] for v in diff]
rects_f = ax2.bar(x, diff, color=bar_colors, alpha=0.9, edgecolor='black', lw=0.4)
ax2.axhline(0, color='black', lw=1, alpha=0.6)

for rect, v in zip(rects_f, diff):
    va_dir = 'bottom' if v >= 0 else 'top'
    offset = 5 if v >= 0 else -14
    ax2.annotate(f'{v:+.3f}',
                xy=(rect.get_x() + rect.get_width() / 2, v),
                xytext=(0, offset),
                textcoords="offset points",
                ha='center', va=va_dir, fontsize=10, fontweight='black', color='#2C3E50')

ax2.set_xticks(x)
ax2.set_xticklabels(yr_labels, fontsize=11, fontweight='bold')
ax2.set_title('(F) LogReg+Filter ΔSharpe vs Buy & Hold', fontsize=13, fontweight='bold', pad=10)
ax2.set_ylabel('ΔSharpe (Strategy − Benchmark)', fontsize=12)
ax2.set_ylim(-5.2, 2.5)
ax2.legend(handles=[
    mpatches.Patch(color=C['pos'], alpha=0.9, label='Outperform Market'),
    mpatches.Patch(color=C['neg'], alpha=0.9, label='Underperform Market')
], fontsize=10, loc='lower right', frameon=True, facecolor='white', edgecolor='#BDC3C7')
ax2.grid(True, alpha=0.3, linestyle=':', axis='y')

fig.suptitle('Year-by-Year Quantitative Performance Audit (Fixed Core Math Approach)\n'
             'Transaction Costs = 0.2% per trade | Signal Day t Applied to Return Day t+1',
             fontsize=13, fontweight='bold')

# Xuất ảnh trực tiếp ra file
output_filename = "fig5_yearly.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"💾 Đã xuất thành công đồ thị tại: {output_filename}")
print("====================================================================")