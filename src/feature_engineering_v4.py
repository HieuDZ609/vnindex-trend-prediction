import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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
print("START PIPELINE: CƠ SỞ 56 FEATURES ĐẾN HẾT CELL 19 (68 CLEAN FEATURES)")
print("====================================================================")

# ====================================================================
# BƯỚC 1: ĐỌC DỮ LIỆU CƠ SỞ
# ====================================================================
print("\n📂 Lấy dữ liệu cơ sở từ tệp local...")
try:
    df = pd.read_csv("VNIndex_Features.csv", parse_dates=["Date"])
except FileNotFoundError:
    raise FileNotFoundError("Hãy đảm bảo file 'VNIndex_Features.csv' (bản thô) nằm cùng thư mục chạy code.")

df = df.sort_values("Date").reset_index(drop=True)
print(f"✅ Đã nạp thành công: {df.shape[0]} dòng × {df.shape[1]} cột")


# ====================================================================
# BƯỚC 2: SỬA LEAKAGE TYPE 1 & TYPE 2 
# ====================================================================
print("\n📐 Sửa Leakage Type 1: Tạo bản lag 1 ngày cho thị trường lệch múi giờ...")
cross_market_needs_lag = ['SP500_Return', 'NASDAQ_Return', 'Gold_Return', 'Oil_Return', 'DXY_Return']
for col in cross_market_needs_lag:
    if col in df.columns:
        df[f"{col}_lag1"] = df[col].shift(1)

if 'Foreign_Net_Buy' in df.columns:
    df = df.rename(columns={'Foreign_Net_Buy': 'FUEVFVND_Volume'})
    print("✅ Đã đổi tên proxy khối ngoại thành FUEVFVND_Volume")

if 'HSI_Return' in df.columns:
    df = df.rename(columns={'HSI_Return': 'HSI_Return_lag1'})

# ====================================================================
# BƯỚC 3: PHÁT TRIỂN BIẾN NÂNG CAO
# ====================================================================
print("\n⚙️  Cell 17: Xây dựng thêm biến nâng cao Trạng thái (Regime) & Tương tác (Interaction)...")

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

if 'SP500_Return_lag1' in df.columns and 'Volume_Shock' in df.columns:
    df['SP500_x_VolShock'] = df['SP500_Return_lag1'] * df['Volume_Shock']
if 'RSI' in df.columns and 'VN_HV10' in df.columns:
    df['RSI_x_HV10'] = df['RSI'] * df['VN_HV10']
if 'VN_Return_lag_1' in df.columns and 'Volume_Shock' in df.columns:
    df['Return_x_VolShock'] = df['VN_Return_lag_1'] * df['Volume_Shock']
if 'SP500_Return_lag1' in df.columns and 'Regime_Uptrend' in df.columns:
    df['SP500_x_Trend'] = df['SP500_Return_lag1'] * df['Regime_Uptrend']

df['WinRate_10d'] = df['Target'].rolling(10).mean().shift(1)
df['WinRate_20d'] = df['Target'].rolling(20).mean().shift(1)
df['Consec_Up']   = df['VN_Return'].gt(0).rolling(5).sum().shift(1)

# ====================================================================
# BƯỚC 4: GÁN TRỰC TIẾP CHUẨN 68 CLEAN FEATURES ĐỒNG BỘ 100% THỨ TỰ
# ====================================================================
feature_groups = {
    'G1_Technical': [
        'VN_Return', 'VN_Return_5d', 'VN_Return_20d', 'RSI', 'MACD_DIFF', 
        'BB_WIDTH', 'BB_PCT', 'ATR14_PCT', 'STOCH_K', 'STOCH_D', 'OBV_Trend', 
        'Price_vs_MA20', 'Price_vs_MA50', 'Volume_Shock', 'Volume_Shock_5',
        'VN_Return_lag_1', 'VN_Return_lag_2', 'VN_Return_lag_3', 
        'VN_Return_lag_5', 'VN_Return_lag_7', 'HL_Range'
    ],
    'G2_MoneyFlow': [
        'Foreign_Net_Ratio', 'Foreign_Net_MA5', 
        'Sector_Momentum', 'Sector_Divergence', 
        'WinRate_10d'
    ],
    'G3_MacroGlobal': [
        'SP500_Return_lag1', 'NASDAQ_Return_lag1', 'Gold_Return_lag1', 'Oil_Return_lag1', 'DXY_Return_lag1', 
        'HSI_Return_lag1', 'Nikkei_Return', 'VIX_Return', 'VIX_Change', 
        'VN_VIX_Signal', 'VN_HV5', 'VN_HV10', 'FX_Stress', 'FX_Stress_20d', 'Asia_Sentiment',
        'Regime_HighVol', 'Regime_LowVol', 'Regime_Uptrend', 'Regime_Downtrend', 'Regime_Crisis',
        'SP500_x_VolShock', 'RSI_x_HV10', 'Return_x_VolShock', 'SP500_x_Trend'
    ],
    'G4_Calendar': [
        'Is_Earnings_Season', 'Is_Earnings_Peak', 'Is_Deriv_Expiry', 'Is_Pre_Expiry', 
        'Is_Quarter_End', 'Is_Quarter_End_Week', 'Is_Near_Tet', 'Is_Tet_Zone', 
        'Is_Monday', 'Is_Friday', 'Is_Month_Start', 'Is_Month_End', 
        'Month_Sin', 'Month_Cos', 'DOW_Sin', 'DOW_Cos', 
        'WinRate_20d', 'Consec_Up'
    ]
}

# Gom đúng 68 features theo đúng thứ tự mảng của Ablation Study
valid_features = [f for g in feature_groups.values() for f in g]

# Điền tạm các NaN rải rác để tránh lỗi, Imputer sẽ xử lý sâu hơn bên dưới
df[valid_features] = df[valid_features].ffill().fillna(0)
df = df.dropna(subset=['Target'])

assert len(valid_features) == 68, f"🔴 LỖI: Số lượng features là {len(valid_features)}, chưa tròn 68!"
print(f"\n✅ KIỂM TOÁN HOÀN TẤT: Đã đồng bộ thứ tự 68 features sạch!")

# ====================================================================
# BƯỚC 5: LƯU TRỮ ĐẦU RA PHIÊN BẢN v4
# ====================================================================
df.to_csv("VNIndex_Features_v4.csv", index=False)


# ====================================================================
# BƯỚC 6: CELL 19 — CHẠY WALK-FORWARD VALIDATION TRÊN 68 CLEAN FEATURES
# ====================================================================
print("\n🔁 Cell 19: Tiến hành huấn luyện 16 folds cuốn chiếu với Imputer & Scaler Fix...")

folds = []
fold_start = df['Date'].min()
while True:
    train_end   = fold_start + pd.DateOffset(years=TRAIN_YEARS)
    test_start = train_end
    test_end   = test_start + pd.DateOffset(months=TEST_MONTHS)
    if test_end > df['Date'].max():
        break
    folds.append((fold_start, train_end, test_start, test_end))
    fold_start = fold_start + pd.DateOffset(months=STEP_MONTHS)

try:
    with open("xgb_best_params.json") as f:
        best_xgb = json.load(f)
    best_xgb.pop('tuned_AUC', None)
except FileNotFoundError:
    best_xgb = {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05, 'min_child_weight': 5, 'subsample': 0.8, 'colsample_bytree': 0.7}

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

    # ĐỒNG BỘ IMPUTER VÀ SCALER VỚI ABLATION SCRIPT
    imputer = SimpleImputer(strategy='median')
    X_tr_imp = imputer.fit_transform(X_tr)
    X_te_imp = imputer.transform(X_te)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_imp)
    X_te_sc = scaler.transform(X_te_imp)

    X_tr_sc = np.nan_to_num(X_tr_sc)
    X_te_sc = np.nan_to_num(X_te_sc)

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

results_df = pd.DataFrame(results)
pred_df    = pd.concat(all_preds, ignore_index=True).sort_values('Date').reset_index(drop=True)

results_df.to_csv("wfv_results_CLEAN.csv", index=False)
pred_df.to_csv("wfv_predictions_fix_v4.csv", index=False)

print("\n" + "=" * 65)
print("🏁 PIPELINE HOÀN TẤT: Đã tạo xong dữ liệu v4 và chạy xong dự đoán!")
print("====================================================================")