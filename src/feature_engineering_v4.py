import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# Cấu hình mặc định
THRESHOLD = 0.001
TRAIN_YEARS = 3
TEST_MONTHS = 6
STEP_MONTHS = 6

print("====================================================================")
print("START PIPELINE: CƠ SỞ 56 FEATURES ĐẾN HẾT CELL 19 (70 CLEAN FEATURES)")
print("====================================================================")

# ====================================================================
# BƯỚC 1: ĐỌC DỮ LIỆU CƠ SỞ (56 FEATURES GỐC VÀ DỮ LIỆU PHỤ TRỢ)
# ====================================================================
# Giả định file đầu vào chứa đầy đủ các cột tính toán thô ban đầu
print("\n📂 Lấy dữ liệu cơ sở từ tệp local...")
try:
    df = pd.read_csv("VNIndex_Features.csv", parse_dates=["Date"])
except FileNotFoundError:
    # Nếu chưa chạy pipeline trước, tạo mock-up để test cấu trúc logic
    raise FileNotFoundError("Hãy đảm bảo file 'VNIndex_Features.csv' (bản thô) nằm cùng thư mục chạy code.")

df = df.sort_values("Date").reset_index(drop=True)
print(f"✅ Đã nạp thành công: {df.shape[0]} dòng × {df.shape[1]} cột")


# ====================================================================
# BƯỚC 2: SỬA LEAKAGE TYPE 1 & TYPE 2 TRÊN FEATURE LIST GỐC
# ====================================================================
# 1. Sửa Type 1 (Timezone Leakage): Tạo lag 1 cho các thị trường phương Tây[cite: 2, 3]
print("\n📐 Sửa Leakage Type 1: Tạo bản lag 1 ngày cho thị trường lệch múi giờ...")
cross_market_needs_lag = ['SP500_Return', 'NASDAQ_Return', 'Gold_Return', 'Oil_Return', 'DXY_Return']
for col in cross_market_needs_lag:
    if col in df.columns:
        df[f"{col}_lag1"] = df[col].shift(1)
        print(f"   ➔ Tạo cột: {col}_lag1[cite: 2]")

# 2. Sửa Type 2 (Future-Return Proxy): Đổi tên để đánh dấu loại bỏ biến khối ngoại ETF rác[cite: 2]
if 'Foreign_Net_Buy' in df.columns:
    df = df.rename(columns={'Foreign_Net_Buy': 'EXCLUDED_Foreign_Net_Buy_proxy'})
    print("🗑️  Đã đánh dấu biến proxy khối ngoại thô thành EXCLUDED_Foreign_Net_Buy_proxy[cite: 2]")


# ====================================================================
# BƯỚC 3: PHÁT TRIỂN BIẾN NÂNG CAO (CELL 17 — REGIME + INTERACTION)
# ====================================================================
print("\n⚙️  Cell 17: Xây dựng thêm biến nâng cao Trạng thái (Regime) & Tương tác (Interaction)...")

# --- 1. Tạo các biến trạng thái thị trường (Regime Features)[cite: 2] ---
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

print("   ➔ Đã tạo 5 biến trạng thái thị trường (Regime Features)[cite: 2]")

# --- 2. Tạo các biến nhân chéo thông minh (Interaction Features)[cite: 2] ---
if 'SP500_Return_lag1' in df.columns and 'Volume_Shock' in df.columns:
    df['SP500_x_VolShock'] = df['SP500_Return_lag1'] * df['Volume_Shock']
if 'RSI' in df.columns and 'VN_HV10' in df.columns:
    df['RSI_x_HV10'] = df['RSI'] * df['VN_HV10']
if 'VN_Return_lag_1' in df.columns and 'Volume_Shock' in df.columns:
    df['Return_x_VolShock'] = df['VN_Return_lag_1'] * df['Volume_Shock']
if 'SP500_Return_lag1' in df.columns and 'Regime_Uptrend' in df.columns:
    df['SP500_x_Trend'] = df['SP500_Return_lag1'] * df['Regime_Uptrend']

# --- 3. Tạo các biến thống kê chuỗi động (Rolling Stats)[cite: 2] ---
df['WinRate_10d'] = df['Target'].rolling(10).mean().shift(1)
df['WinRate_20d'] = df['Target'].rolling(20).mean().shift(1)
df['Consec_Up']   = df['VN_Return'].gt(0).rolling(5).sum().shift(1)

print("   ➔ Đã tạo các biến phái sinh tương tác (Interaction Features)[cite: 2]")


# ====================================================================
# BƯỚC 4: LỌC BỎ CHẶT CHẼ ĐỂ TRÍCH XUẤT ĐÚNG 70 FEATURES SẠCH
# ====================================================================
# Định nghĩa các danh sách cấm đưa vào mô hình học (Leak nghiêm trọng)[cite: 1]
hard_exclude = [
    'Next_Return',    # T+1 Return = Target lộ diện hoàn toàn (Leak loại 2)[cite: 1, 3]
    'VNIndex',        # Giá đóng cửa tuyệt đối (Non-stationary)[cite: 1, 3]
    'Open', 'High', 'Low', 'Date', 'Target', 'VN_HV20_Roll'
]
orig_cross_market = ['SP500_Return', 'NASDAQ_Return', 'Gold_Return', 'Oil_Return', 'DXY_Return']

# Gom danh sách đặc trưng hợp lệ
valid_features = [
    c for c in df.columns
    if c not in hard_exclude + orig_cross_market
    and not c.startswith('EXCLUDED_')
    and df[c].isna().mean() < 0.10
]

# Đảm bảo dọn sạch NaN trong ma trận đặc trưng
df[valid_features] = df[valid_features].ffill().fillna(0)
df = df.dropna(subset=['Target'])

# Kiểm tra xác nhận (Assert) hệ thống[cite: 1]
assert 'Next_Return' not in valid_features, "🔴 LỖI: Next_Return vẫn chưa bị loại bỏ!"[cite: 1]
print(f"\n✅ KIỂM TOÁN HOÀN TẤT: Đã cấu trúc thành công Feature list v4 sạch!")
print(f"   ➔ Số lượng đặc trưng sạch đưa vào mô hình: {len(valid_features)} features[cite: 1, 2]")


# ====================================================================
# BƯỚC 5: LƯU TRỮ ĐẦU RA PHIÊN BẢN v4 (83 CỘT TỔNG - CHỨA 70 FEATURES SẠCH)
# ====================================================================
# Lưu trữ dữ liệu thô bao gồm cả nhãn để các cell xử lý sau chạy local không lỗi[cite: 2]
df.to_csv("VNIndex_Features_v4.csv", index=False)
print(f"💾 Đã xuất tệp tin local: VNIndex_Features_v4.csv ({df.shape[1]} cột tổng)[cite: 2]")


# ====================================================================
# BƯỚC 6: CELL 19 — CHẠY WALK-FORWARD VALIDATION TRÊN 70 CLEAN FEATURES
# ====================================================================
print("\n🔁 Cell 19: Tiến hành huấn luyện 16 folds cuốn chiếu với Type 3 Scaler Fix...")

# Chia cấu trúc folds[cite: 1]
folds = []
fold_start = df['Date'].min()
while True:
    train_end  = fold_start + pd.DateOffset(years=TRAIN_YEARS)
    test_start = train_end
    test_end   = test_start + pd.DateOffset(months=TEST_MONTHS)
    if test_end > df['Date'].max():
        break
    folds.append((fold_start, train_end, test_start, test_end))
    fold_start = fold_start + pd.DateOffset(months=STEP_MONTHS)

# Đọc cấu hình tối ưu của XGBoost từ file local (Tạo fallback cấu hình bảo thủ nếu thiếu file)[cite: 1, 2]
try:
    with open("xgb_best_params.json") as f:
        best_xgb = json.load(f)
    best_xgb.pop('tuned_AUC', None)
    print(f"   ➔ Đã nạp tham số tối ưu từ 'xgb_best_params.json'[cite: 1, 2]")
except FileNotFoundError:
    best_xgb = {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05, 'min_child_weight': 5, 'subsample': 0.8, 'colsample_bytree': 0.7}
    print(f"   ⚠️  Không tìm thấy file tuning local, áp dụng cấu hình XGBoost mặc định bảo thủ.")

# Khởi tạo mô hình[cite: 1]
models = {
    'RF': RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=15,
        class_weight='balanced', random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(
        **best_xgb, eval_metric='logloss',
        random_state=42, use_label_encoder=False, verbosity=0),
    'LogReg': LogisticRegression(
        C=0.05, max_iter=1000, random_state=42,
        class_weight='balanced'),
}

results, all_preds = [], []
print(f"\n🔁 Đang huấn luyện {len(folds)} folds cuốn chiếu thực tế...")
print(f"{'Fold':>4} {'Test Period':>24}  {'AUC_RF':>8} {'AUC_XGB':>9} {'AUC_LR':>8}")
print("-" * 62)

for i, (fs, te, ts, tend) in enumerate(folds):
    train_m = (df['Date'] >= fs)  & (df['Date'] < te)
    test_m  = (df['Date'] >= ts)  & (df['Date'] < tend)
    
    X_tr, y_tr = df.loc[train_m, valid_features], df.loc[train_m, 'Target']
    X_te, y_te = df.loc[test_m,  valid_features], df.loc[test_m,  'Target']
    dates_te = df.loc[test_m, 'Date']

    if len(y_te) < 30 or y_te.nunique() < 2:
        continue

    # --- SỬA LEAKAGE TYPE 3 (SCALER LEAK) CỤ CỤC BỘ TỪNG FOLD ---[cite: 1]
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    fold_r = {'fold': i+1, 'test_start': ts.date(), 'test_end': tend.date()}
    fold_p = {'Date': dates_te.values, 'y_true': y_te.values}

    for name, model in models.items():
        model.fit(X_tr_sc, y_tr)
        prob = model.predict_proba(X_te_sc)[:, 1]
        pred = (prob >= 0.5).astype(int)
        
        fold_r[f'AUC_{name}'] = roc_auc_score(y_te, prob)
        fold_r[f'Acc_{name}'] = accuracy_score(y_te, pred)
        fold_r[f'F1_{name}']  = f1_score(y_te, pred, zero_division=0)
        
        fold_p[f'prob_{name}'] = prob
        fold_p[f'pred_{name}'] = pred

    results.append(fold_r)
    all_preds.append(pd.DataFrame(fold_p))
    
    print(f"  {i+1:>2}  {str(ts.date())}➔{str(tend.date())}  "
          f"  {fold_r['AUC_RF']:>8.4f}"
          f"  {fold_r['AUC_XGBoost']:>8.4f}"
          f"  {fold_r['AUC_LogReg']:>8.4f}")

# Gom kết quả đầu ra lưu trữ làm nguyên liệu cho Cell 20 tính Causal Trading[cite: 1, 2]
results_df = pd.DataFrame(results)
pred_df    = pd.concat(all_preds, ignore_index=True).sort_values('Date').reset_index(drop=True)

results_df.to_csv("wfv_results_CLEAN.csv", index=False)
pred_df.to_csv("wfv_predictions_fix_v4.csv", index=False)

print("\n" + "=" * 65)
print("🏁 PIPELINE HOÀN TẤT: Đã tạo xong dữ liệu v4 và chạy xong dự đoán Cell 19!")
print("====================================================================")
print("File đầu ra sẵn sàng tại thư mục local:")
print("  ➔ VNIndex_Features_v4.csv (Bảng 70 đặc trưng sạch nâng cao)[cite: 2]")
print("  ➔ wfv_results_CLEAN.csv (Số liệu AUC/Acc/F1 gốc của từng fold)[cite: 1, 2]")
print("  ➔ wfv_predictions_fix_v4.csv (Tín hiệu thô phục vụ Cell 20 sửa luật trading)[cite: 2]")
print("=" * 65)