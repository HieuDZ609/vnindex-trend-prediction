import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Thiết lập hệ số ma trận kết quả thực tế từ log chạy định lượng
data = {
    'Feature Group': ['G1 — Technical', 'G2 — Money Flow', 'G3 — Macro/Global', 'G4 — Calendar', 'G_ALL — Full Matrix'],
    'RF':      [0.489, 0.496, 0.566, 0.515, 0.552],
    'XGBoost': [0.477, 0.489, 0.541, 0.511, 0.539],
    'LogReg':  [0.474, 0.503, 0.567, 0.513, 0.547]
}

df_plot = pd.DataFrame(data)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(df_plot['Feature Group']))
width = 0.24  # Tăng nhẹ độ rộng để hiển thị số đẹp hơn

# Vẽ các cụm cột dữ liệu
rects1 = ax.bar(x - width, df_plot['RF'], width, label='Random Forest', color="#69A1E2", edgecolor='black', linewidth=0.5)
rects2 = ax.bar(x, df_plot['XGBoost'], width, label='XGBoost', color="#90EB93", edgecolor='black', linewidth=0.5)
rects3 = ax.bar(x + width, df_plot['LogReg'], width, label='Logistic Regression', color="#6CCEEA", edgecolor='black', linewidth=0.5)

# Vẽ đường mốc ngẫu nhiên 0.50
ax.axhline(y=0.50, color='#E74C3C', linestyle='--', linewidth=1.5, label='Random Guessing Baseline (0.50)')

# Hàm helper để add label số lên đầu từng cột cột
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=13, fontweight='bold')

# Thực hiện add label số lên đầu cả 3 mô hình
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

ax.set_title('Ablation Study: Cumulative Out-of-Sample AUC by Feature Group', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Feature Groups Partitioned', fontsize=12, labelpad=12)
ax.set_ylabel('Cumulative Out-of-Sample AUC-ROC', fontsize=12, labelpad=12)
ax.set_xticks(x)
ax.set_xticklabels(df_plot['Feature Group'], fontsize=11)
ax.set_ylim(0.35, 0.65)

ax.grid(axis='y', linestyle=':', alpha=0.6, color='#BDC3C7')

ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='#BDC3C7', fontsize=10)

plt.tight_layout()
output_filename = 'fig4_ablation_study_labeled.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.close()
print("Lưu đồ thị thành công!")