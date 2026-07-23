import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

print("====================================================================")
print("START OPERATION: ABLATION STUDY - CUMULATIVE AUC BY FEATURE GROUP")
print("====================================================================")

# 1. NẠP DỮ LIỆU ĐÃ LÀM SẠCH V4
if not os.path.exists("VNIndex_Features_v4.csv"):
    raise FileNotFoundError("🔴 Lỗi: Thiếu file dữ liệu VNIndex_Features_v4.csv. Hãy chạy pipeline v4 trước!")

df = pd.read_csv("VNIndex_Features_v4.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# 2. PHÂN TÁCH DANH SÁCH FEATURES THEO 4 NHÓM GỐC
hard_exclude = ['Date', 'VNIndex', 'Open', 'High', 'Low', 'Volume', 'Next_Return', 'Target']

valid_features = [c for c in df.columns if c not in hard_exclude]

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
valid_features = [f for g in feature_groups.values() for f in g]
all_features = [f for g in feature_groups.values() for f in g]

# Thêm nhóm G_ALL bao gồm toàn bộ các biến hợp lệ
feature_groups['G_ALL_Full_Matrix'] = all_features


total_assigned = sum(len(v) for v in feature_groups.values())
print(f"✅ Xác thực cấu trúc dữ liệu đầu vào thành công:")
print(f"   ➔ Tổng số đặc trưng làm sạch phát hiện được: {len(valid_features)} features")
print(f"   ➔ Tổng số đặc trưng đã phân phối vào 4 nhóm: {total_assigned} features")

# Cấu hình bước nhảy Folds cuốn chiếu
TRAIN_YEARS = 3
TEST_MONTHS = 6
STEP_MONTHS = 6

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

# Đọc cấu hình XGBoost
try:
    with open("xgb_best_params.json") as f:
        best_xgb = json.load(f)
    best_xgb.pop('tuned_AUC', None)
except FileNotFoundError:
    best_xgb = {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.7}

def get_fresh_models():
    return {
        'RF': RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=15, class_weight='balanced', random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(**best_xgb, eval_metric='logloss', random_state=42, use_label_encoder=False, verbosity=0),
        'LogReg': LogisticRegression(C=0.05, max_iter=1000, class_weight='balanced', random_state=42)
    }

ablation_results = []

# 3. TIẾN HÀNH QUÉT CỐT LÕI
for g_name, g_features in feature_groups.items():
    g_features = [c for c in g_features if c in df.columns]
    if len(g_features) == 0:
        print(f"⚠️  Nhóm {g_name} không tìm thấy đặc trưng phù hợp trong tệp tin, bỏ qua.")
        continue
        
    print(f"\n📡 Đang bóc tách thực nghiệm nhóm: {g_name} ({len(g_features)} features)...")
    
    all_y_true = []
    preds_dict = {'RF': [], 'XGBoost': [], 'LogReg': []}
    
    for fs, te, ts, tend in folds:
        train_m = (df['Date'] >= fs)  & (df['Date'] < te)
        test_m  = (df['Date'] >= ts)  & (df['Date'] < tend)
        
        X_tr, y_tr = df.loc[train_m, g_features], df.loc[train_m, 'Target']
        X_te, y_te = df.loc[test_m,  g_features], df.loc[test_m,  'Target']
        
        if len(y_te) < 30 or y_te.nunique() < 2:
            continue
            
        # FIX TRIỆT ĐỂ LỖI NaN: Imputer + Scaler per fold (Chống Leakage)
        imputer = SimpleImputer(strategy='median')
        X_tr_imp = imputer.fit_transform(X_tr)
        X_te_imp = imputer.transform(X_te)
        
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_imp)
        X_te_sc = scaler.transform(X_te_imp)
        
        # Đảm bảo không còn NaN nào lọt qua
        X_tr_sc = np.nan_to_num(X_tr_sc)
        X_te_sc = np.nan_to_num(X_te_sc)
        
        models = get_fresh_models()
        all_y_true.extend(y_te.values)
        
        for m_name, model in models.items():
            model.fit(X_tr_sc, y_tr)
            prob = model.predict_proba(X_te_sc)[:, 1]
            preds_dict[m_name].extend(prob)
            
    # 4. TÍNH CUMULATIVE AUC
    row_res = {'Feature Group': g_name, 'Num_Features': len(g_features)}
    for m_name in ['RF', 'XGBoost', 'LogReg']:
        row_res[f'Cumulative_AUC_{m_name}'] = roc_auc_score(all_y_true, preds_dict[m_name])
    
    ablation_results.append(row_res)

# Xuất ma trận kết quả tổng hợp
ablation_df = pd.DataFrame(ablation_results)
print("\n" + "="*80)
print("📋 BẢNG KIỂM TOÁN ABLATION STUDY: CUMULATIVE AUC THEO TỪNG NHÓM ĐẶC TRƯNG")
print("="*80)
print(ablation_df.to_string(index=False))
print("="*80)

ablation_df.to_csv("ablation_study_results.csv", index=False)
print("💾 Đã lưu file kết quả bóc tách tại: ablation_study_results.csv")