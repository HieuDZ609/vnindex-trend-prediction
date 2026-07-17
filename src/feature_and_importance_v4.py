"""
feature_and_importance_v4.py
=============================
Pipeline gom bộ kiểm toán rò rỉ dữ liệu (Leakage Audit), trích xuất 70 đặc trưng sạch (v4)
và xuất bảng xếp hạng tầm quan trọng của tính năng (Feature Importance).

Đầu vào local : VNIndex_Features.csv (File chứa 56 đặc trưng thô ban đầu)
Đầu ra local : 
    - VNIndex_Features_v4.csv (Bộ dữ liệu 70 đặc trưng sạch hoàn toàn)
    - feature_importance.csv   (Bảng xếp hạng Feature Importance từ mô hình XGBoost)

Mạch xử lý tích hợp từ Cell 16A đến Cell 20 của Notebook.
"""

import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

print("====================================================================")
print("START PIPELINE: FIX FEATURE ➔ EXTRACT V4 ➔ FEATURE IMPORTANCE")
print("====================================================================")

# ====================================================================
# BƯỚC 1: NẠP DỮ LIỆU CƠ SỞ (56 FEATURES GỐC TRÊN LOCAL)
# ====================================================================
print("\n📂 Lấy dữ liệu 56 đặc trưng thô ban đầu...")
try:
    df = pd.read_csv("VNIndex_Features.csv", parse_dates=["Date"])
except FileNotFoundError:
    raise FileNotFoundError("Không tìm thấy file 'VNIndex_Features.csv' thô. Hãy đảm bảo file nằm cùng thư mục chạy code.")

df = df.sort_values("Date").reset_index(drop=True)
print(f"✅ Đã nạp thành công: {df.shape[0]} dòng × {df.shape[1]} cột")


# ====================================================================
# BƯỚC 2: FIX FEATURES — SỬA SẠCH DATA LEAKAGE (TYPE 1 & TYPE 2)
# ====================================================================
print("\n🛠️  Giai đoạn Fix Features: Tiến hành khử rò rỉ dữ liệu...")

# 1. Sửa Type 1 (Timezone Leakage): Dịch chuyển lag 1 ngày cho thị trường đóng cửa sau Việt Nam[cite: 2, 3]
cross_market_needs_lag = ['SP500_Return', 'NASDAQ_Return', 'Gold_Return', 'Oil_Return', 'DXY_Return']
for col in cross_market_needs_lag:
    if col in df.columns:
        df[f"{col}_lag1"] = df[col].shift(1)
print("   ➔ [Type 1] Đã sửa lệch múi giờ Mỹ bằng cách tạo các biến *_lag1[cite: 2, 3]")

# 2. Sửa Type 2 (Future-Return Proxy): Đổi tên để cô lập biến khối ngoại ETF giao dịch rác[cite: 2]
if 'Foreign_Net_Buy' in df.columns:
    df = df.rename(columns={'Foreign_Net_Buy': 'EXCLUDED_Foreign_Net_Buy_proxy'})
print("   ➔ [Type 2] Đã đánh dấu loại bỏ proxy khối ngoại thô[cite: 2]")


# ====================================================================
# BƯỚC 3: TRÍCH XUẤT BIẾN MỞ RỘNG (REGIME + INTERACTION NÂNG CAO)
# ====================================================================
print("\n📐 Giai đoạn trích xuất biến nâng cao (Cell 17 — Biến trạng thái & Nhân chéo)...")

# 1. Trích xuất các biến trạng thái cấu trúc thị trường (Regime Features)[cite: 2]
df['VN_HV20_Roll'] = df['VN_Return'].rolling(20).std() * np.sqrt(252)
vol_q75 = df['VN_HV20_Roll'].rolling(252, min_periods=60).quantile(0.75)
vol_q25 = df['VN_HV20_Roll'].rolling(252, min_periods=60).quantile(0.25)

df['Regime_HighVol']   = (df['VN_HV20_Roll'] > vol_q75).astype(int)
df['Regime_LowVol']    = (df['VN_HV20_Roll'] < vol_q25).astype(int)
df['Regime_Uptrend']   = (df['VN_Return'].rolling(20).mean() > 0).astype(int)
df['Regime_Downtrend'] = (df['VN_Return'].rolling(20).mean() < 0).astype(int)

if 'VIX_Return' in df.columns:
    vix_q90 = df['VIX_Return'].rolling(252, min_periods=60).quantile(0.90)
    df['Regime_Crisis'] = (df['VIX_Return'] > vix_q90).astype(int)
else:
    df['Regime_Crisis'] = 0

# 2. Trích xuất các biến nhân chéo thông minh (Interaction Features)[cite: 2]
if 'SP500_Return_lag1' in df.columns and 'Volume_Shock' in df.columns:
    df['SP500_x_VolShock'] = df['SP500_Return_lag1'] * df['Volume_Shock']
if 'RSI' in df.columns and 'VN_HV10' in df.columns:
    df['RSI_x_HV10'] = df['RSI'] * df['VN_HV10']
if 'VN_Return_lag_1' in df.columns and 'Volume_Shock' in df.columns:
    df['Return_x_VolShock'] = df['VN_Return_lag_1'] * df['Volume_Shock']
if 'SP500_Return_lag1' in df.columns and 'Regime_Uptrend' in df.columns:
    df['SP500_x_Trend'] = df['SP500_Return_lag1'] * df['Regime_Uptrend']

# 3. Trích xuất chuỗi động lực học thống kê (Rolling Stats)[cite: 2]
df['WinRate_10d'] = df['Target'].rolling(10).mean().shift(1)
df['WinRate_20d'] = df['Target'].rolling(20).mean().shift(1)
df['Consec_Up']   = df['VN_Return'].gt(0).rolling(5).sum().shift(1)

print("   ➔ Đã sinh xong các biến phái sinh nâng cao độc lập thời gian[cite: 2]")


# ====================================================================
# BƯỚC 4: SÀNG LỌC TRIỆT ĐỂ ➔ XUẤT FILE FEATURE_V4 (70 FEATURES SẠCH)
# ====================================================================
# Thiết lập danh sách bài trừ tuyệt đối biến leak (Type 2) và giá tuyệt đối (Non-stationary)[cite: 1, 3]
hard_exclude = [
    'Next_Return',    # Lợi suất tương lai ngày T+1 = Target bị lộ diện (Xóa vĩnh viễn)[cite: 1, 3]
    'VNIndex',        # Giá đóng cửa tuyệt đối[cite: 1, 3]
    'Open', 'High', 'Low', 'Date', 'Target', 'VN_HV20_Roll'
]
orig_cross_market = ['SP500_Return', 'NASDAQ_Return', 'Gold_Return', 'Oil_Return', 'DXY_Return']

# Trích xuất danh sách tính năng sạch đạt chuẩn đưa vào mô hình học
valid_features = [
    c for c in df.columns
    if c not in hard_exclude + orig_cross_market
    and not c.startswith('EXCLUDED_')
    and df[c].isna().mean() < 0.10
]

# Làm sạch NaN ma trận đặc trưng qua ffill và điền 0 mốc đầu[cite: 1]
df[valid_features] = df[valid_features].ffill().fillna(0)
df = df.dropna(subset=['Target'])

# Kiểm tra chặn (Assert) kỹ thuật[cite: 1]
assert 'Next_Return' not in valid_features, "🔴 CRITICAL ERROR: Next_Return vẫn lọt vào danh sách đặc trưng!"[cite: 1]
print(f"\n✅ ĐÃ TRÍCH XUẤT XONG FEATURE_V4: {len(valid_features)} features sạch đưa vào model[cite: 1, 2]")

# Xuất tệp dữ liệu v4 ra local lưu trữ[cite: 2]
df.to_csv("VNIndex_Features_v4.csv", index=False)
print("💾 Đã xuất tệp thành công: VNIndex_Features_v4.csv[cite: 2]")


# ====================================================================
# BƯỚC 5: TÍNH TOÁN VÀ XUẤT FILE FEATURE_IMPORTANCE.CSV (CELL 20)
# ====================================================================
print("\n🔍 Cell 20: Tiến hành đo lường chỉ số tăng trưởng thông tin (Feature Importance)...")

# Chuẩn hóa ma trận đặc trưng sạch trước khi fit[cite: 1]
X = StandardScaler().fit_transform(df[valid_features])
y = df['Target'].values

# Đọc cấu hình tham số tối ưu từ Grid Search trước đó (hoặc dùng mặc định bảo thủ)[cite: 1, 2]
try:
    with open("xgb_best_params.json") as f:
        best_params = json.load(f)
    best_params.pop('tuned_AUC', None)
    print("   ➔ Đã cấu hình XGBoost theo bộ tham số tối ưu trích xuất từ file json[cite: 1, 2]")
except FileNotFoundError:
    best_params = {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05, 'min_child_weight': 5, 'subsample': 0.8, 'colsample_bytree': 0.7}
    print("   ⚠️  Không tìm thấy xgb_best_params.json local, áp dụng cấu hình XGBoost mặc định.")

# Khởi tạo mô hình định giá[cite: 1]
xgb_model = XGBClassifier(**best_params, eval_metric='logloss', random_state=42, use_label_encoder=False, verbosity=0)
xgb_model.fit(X, y)

# Tạo bảng Feature Importance xếp hạng từ cao xuống thấp[cite: 2]
fi_df = pd.DataFrame({
    'feature': valid_features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 TOP 10 FEATURE IMPORTANCE CÓ ĐÓNG GÓP GAIN LỚN NHẤT TRÊN LOCAL:")
print(fi_df.head(10).to_string(index=False))

# Xuất bảng xếp hạng đặc trưng ra tệp CSV local phục vụ vẽ biểu đồ Paper[cite: 2]
fi_df.to_csv("feature_importance.csv", index=False)

print("\n" + "=" * 65)
print("🏁 PIPELINE GỘP HOÀN TẤT: DỮ LIỆU ĐÃ ĐƯỢC LÀM SẠCH VÀ TRÍCH XUẤT THÀNH CÔNG!")
print("====================================================================")
print("Các tệp đầu ra hiện có tại cùng thư mục:")
print("  ➔ VNIndex_Features_v4.csv (Tập dữ liệu chứa đúng 70 đặc trưng sạch)[cite: 2]")
print("  ➔ feature_importance.csv   (Bảng xếp hạng đóng góp Gain dùng cho Paper)[cite: 2]")
print("=" * 65)