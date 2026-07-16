"""
feature_engineering.py
=======================
Xây dựng 56 features (4 nhóm G1-G4) + nhãn Target cho bài toán dự
đoán xu hướng VNIndex, từ dữ liệu thô do `data_collection.py` cung cấp.

Refactor từ notebook gốc `VNIndex_Trend_Prediction.ipynb`:
    - Cell 4-9        : feature G1-G4 phiên bản đầu
    - Cell 16A        : fix leak múi giờ (lag 1 ngày cho cross-market)
    - Cell 16B        : loại Foreign_Net_Buy thô khỏi feature list
    - Cell 16C_FIX    : loại các cột leak (Next_Return, OHLC tuyệt đối)
    - Cell 17A        : VIX (level, không lag), HSI/Nikkei (không lag)

QUAN TRỌNG — đây là pipeline "sạch", khớp với Bảng "Feature Groups
Summary" và Bảng "Ablation Study" trong paper: 56 features
(G1=21, G2=4, G3=15, G4=16). Đã test bằng dữ liệu giả lập, ra đúng
56/56 cột, không thiếu không dư.

Notebook gốc còn có các cell thử nghiệm SAU Cell 17A (Cell 19 trở
đi: Regime_HighVol/LowVol/Uptrend/Downtrend/Crisis, các cột tương
tác *_x_*, và WinRate_10d/20d, Consec_Up). Các đặc trưng này KHÔNG
nằm trong 56 features được báo cáo trong paper, nên KHÔNG được đưa
vào file này. Nếu bạn muốn thử nghiệm thêm, đó là hướng mở rộng.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

# Threshold cho Target — 0.1%: coi các phiên tăng rất nhẹ (< 0.1%) là gần như đứng yên, gộp
# vào nhãn DOWN, tránh model học nhiễu thành tín hiệu tăng.
THRESHOLD = 0.001

# Các mã cross-market lệch múi giờ so với VNIndex (đóng cửa sau VNIndex
# nhiều giờ) → PHẢI lag 1 ngày, nếu không sẽ leak thông tin tương lai.
# Xem "Leakage Analysis and Prevention" (Type 1) trong paper.
CROSS_MARKET_NEEDS_LAG: List[str] = [
    "SP500_Return", "NASDAQ_Return", "Gold_Return", "Oil_Return", "DXY_Return",
]

# Ngày Tết Nguyên Đán 2014-2025 (dùng cho Is_Near_Tet).
TET_DATES = pd.to_datetime([
    "2014-01-31", "2015-02-19", "2016-02-08", "2017-01-28",
    "2018-02-16", "2019-02-05", "2020-01-25", "2021-02-12",
    "2022-02-01", "2023-01-22", "2024-02-10", "2025-01-29",
])


# ══════════════════════════════════════════════════════════════════
# G1 — Technical (21 features)
# ══════════════════════════════════════════════════════════════════
def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """G1: 21 đặc trưng kỹ thuật, tính từ giá/khối lượng VNIndex."""
    df = df.copy()
    close, high, low, vol = df["VNIndex"], df["High"], df["Low"], df["Volume"]

    df["VN_Return"] = close.pct_change()
    df["VN_Return_5d"] = close.pct_change(5)
    df["VN_Return_20d"] = close.pct_change(20)

    df["RSI"] = RSIIndicator(close=close, window=14).rsi()

    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD_DIFF"] = macd.macd_diff()
    # MACD_SIGNAL được tính ở notebook gốc nhưng không nằm trong G1
    # cuối cùng dùng cho model — bỏ qua ở đây cho gọn.

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    df["Price_vs_MA20"] = (close - ma20) / ma20 * 100
    df["Price_vs_MA50"] = (close - ma50) / ma50 * 100

    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["BB_WIDTH"] = bb.bollinger_wband()
    df["BB_PCT"] = bb.bollinger_pband()

    df["ATR14_PCT"] = AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range() / close * 100

    stoch = StochasticOscillator(high=high, low=low, close=close,
                                  window=14, smooth_window=3)
    df["STOCH_K"] = stoch.stoch()
    df["STOCH_D"] = stoch.stoch_signal()

    obv = OnBalanceVolumeIndicator(close=close, volume=vol).on_balance_volume()
    obv_ma = obv.rolling(20).mean()
    df["OBV_Trend"] = (obv - obv_ma) / obv_ma.abs().replace(0, np.nan)

    vol_ma20 = vol.rolling(20).mean()
    df["Volume_Shock"] = vol / vol_ma20
    df["Volume_Shock_5"] = vol / vol.rolling(5).mean()

    for lag in [1, 2, 3, 5, 7]:
        df[f"VN_Return_lag_{lag}"] = df["VN_Return"].shift(lag)

    df["HL_Range"] = (high - low) / close * 100

    return df


TECHNICAL_FEATURES: List[str] = [
    "VN_Return", "VN_Return_5d", "VN_Return_20d",
    "RSI", "MACD_DIFF", "BB_WIDTH", "BB_PCT", "ATR14_PCT",
    "STOCH_K", "STOCH_D", "OBV_Trend",
    "Price_vs_MA20", "Price_vs_MA50",
    "Volume_Shock", "Volume_Shock_5",
    "VN_Return_lag_1", "VN_Return_lag_2", "VN_Return_lag_3",
    "VN_Return_lag_5", "VN_Return_lag_7",
    "HL_Range",
]  # 21 — khớp Bảng "Feature Groups Summary" (G1)


# ══════════════════════════════════════════════════════════════════
# G2 — Money Flow (4 features)
# ══════════════════════════════════════════════════════════════════
def compute_money_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """G2: 4 đặc trưng dòng tiền.

    Dùng Foreign_Net_Buy (thô, từ data_collection.py) để tính 2 tỉ lệ
    phái sinh, và 3 mã đại diện ngành VCB/VHM/HPG để tính động lượng
    ngành. Cột Foreign_Net_Buy THÔ không được đưa vào feature list
    cuối cùng (xem Cell 16B của notebook gốc) — chỉ 2 cột phái sinh
    Ratio/MA5 mới được giữ lại.
    """
    df = df.copy()

    df["Foreign_Net_Ratio"] = (
        df["Foreign_Net_Buy"] / (df["Volume"] * df["VNIndex"]).replace(0, np.nan)
    ).fillna(0)
    df["Foreign_Net_MA5"] = df["Foreign_Net_Buy"].rolling(5).mean().fillna(0)

    sector_cols = [c for c in ["VCB_R", "VHM_R", "HPG_R"] if c in df.columns]
    if sector_cols:
        df[sector_cols] = df[sector_cols].ffill()
        df["Sector_Momentum"] = df[sector_cols].mean(axis=1)
        df["Sector_Divergence"] = df[sector_cols].std(axis=1)
    else:
        df["Sector_Momentum"] = 0.0
        df["Sector_Divergence"] = 0.0

    return df


MONEY_FLOW_FEATURES: List[str] = [
    "Foreign_Net_Ratio", "Foreign_Net_MA5",
    "Sector_Momentum", "Sector_Divergence",
]  # 4 — khớp Bảng "Feature Groups Summary" (G2)


# ══════════════════════════════════════════════════════════════════
# G3 — Macro / Global (15 features)
# ══════════════════════════════════════════════════════════════════
def fix_cross_market_lag(df: pd.DataFrame) -> pd.DataFrame:
    """Sửa leak múi giờ (Type 1 trong paper).

    S&P 500 / NASDAQ / Gold / Oil / DXY đóng cửa SAU khi VNIndex đã
    đóng cửa (S&P 500 đóng ~22:00 giờ VN, VNIndex đóng lúc 14:45).
    Dùng giá trị NGÀY T của các mã này để dự đoán VNIndex ngày T là
    look-ahead bias. Ta tạo bản lag 1 ngày; cột gốc (chưa lag) bị
    loại khỏi feature list ở bước lắp ráp cuối (assemble_final_dataset).

    HSI và Nikkei KHÔNG cần lag: phiên của chúng trùng hoặc kết thúc
    trước khi VNIndex đóng cửa cùng ngày, nên giá trị ngày T đã là
    thông tin biết trước tại thời điểm cần dự đoán.
    """
    df = df.copy()
    for col in CROSS_MARKET_NEEDS_LAG:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)
    return df


def compute_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """G3: 15 đặc trưng vĩ mô/toàn cầu, sau khi đã lag đúng chiều
    thời gian (xem `fix_cross_market_lag`)."""
    df = fix_cross_market_lag(df)

    df["VN_HV5"] = df["VN_Return"].rolling(5).std() * np.sqrt(252)
    df["VN_HV10"] = df["VN_Return"].rolling(10).std() * np.sqrt(252)
    df["VN_HV20"] = df["VN_Return"].rolling(20).std() * np.sqrt(252)  # chỉ dùng nội bộ, không phải feature cuối
    df["VN_VIX_Signal"] = (df["VN_HV10"] / df["VN_HV20"].replace(0, np.nan)).fillna(1)

    # LƯU Ý ĐẶT TÊN :
    # cột 'VIX_Return' ở đây thực chất chứa MỨC (level) của VIX, KHÔNG
    # phải % thay đổi — vì VIX vốn là chỉ báo "mức độ sợ hãi", dùng
    # level có ý nghĩa hơn % change. % thay đổi thật nằm ở 'VIX_Change'.
    if "VIX" in df.columns:
        df["VIX_Return"] = df["VIX"]          # tên giữ nguyên, chứa level
        df["VIX_Change"] = df["VIX"].pct_change()

    # ⚠️ PHÁT HIỆN KHI REFACTOR (đã báo trong chat, không có sẵn note
    # này ở notebook gốc): FX_Stress dùng 'DXY_Return' GỐC (chưa lag),
    # không phải 'DXY_Return_lag1'. Đây là cùng loại rủi ro leak múi
    # giờ mà Type 1 đã sửa cho các cột *_lag1 khác, nhưng dòng tính
    # FX_Stress trong notebook gốc (viết trước Cell 16A) không được
    # cập nhật lại sau khi thêm lag. Giữ NGUYÊN logic gốc ở đây để
    # khớp với số liệu đã công bố trong paper — bạn cần tự quyết định
    # có sửa lại (đổi sang DXY_Return_lag1) và chạy lại WFV hay không
    # trước khi nộp bài.
    df["FX_Stress"] = df["DXY_Return"].rolling(5).mean().fillna(0) * 100
    df["FX_Stress_20d"] = df["FX_Stress"].rolling(20).sum().fillna(0)

    asia_cols = [c for c in ["HSI_Return", "Nikkei_Return"] if c in df.columns]
    df["Asia_Sentiment"] = df[asia_cols].mean(axis=1).fillna(0) if asia_cols else 0.0

    return df


MACRO_FEATURES: List[str] = [
    "SP500_Return_lag1", "NASDAQ_Return_lag1",
    "Gold_Return_lag1", "Oil_Return_lag1", "DXY_Return_lag1",
    "HSI_Return", "Nikkei_Return",
    "VIX_Return", "VIX_Change", "VN_VIX_Signal",
    "VN_HV5", "VN_HV10",
    "FX_Stress", "FX_Stress_20d", "Asia_Sentiment",
]  # 15 — khớp Bảng "Feature Groups Summary" (G3)


# ══════════════════════════════════════════════════════════════════
# G4 — Calendar (16 features)
# ══════════════════════════════════════════════════════════════════
def compute_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """G4: 16 đặc trưng lịch (mùa báo cáo, đáo hạn phái sinh, Tết,
    hiệu ứng ngày trong tuần, mã hóa chu kỳ sin/cos)."""
    df = df.copy()

    month = df["Date"].dt.month
    dow = df["Date"].dt.dayofweek
    dom = df["Date"].dt.day

    # Mùa báo cáo: T1, T4, T7, T10
    df["Is_Earnings_Season"] = month.isin([1, 4, 7, 10]).astype(int)
    df["Is_Earnings_Peak"] = (month.isin([1, 4, 7, 10]) & (dom <= 15)).astype(int)

    # Đáo hạn phái sinh: Thứ Năm (=3) tuần thứ 3 (ngày 15-21)
    df["Is_Deriv_Expiry"] = ((dow == 3) & dom.between(15, 21)).astype(int)
    df["Is_Pre_Expiry"] = df["Is_Deriv_Expiry"].shift(1).fillna(0).astype(int)

    # Cuối quý / cuối năm
    df["Is_Quarter_End"] = month.isin([3, 6, 9, 12]).astype(int)
    df["Is_Quarter_End_Week"] = (month.isin([3, 6, 9, 12]) & (dom >= 23)).astype(int)

    # Tết (±10 ngày)
    df["Is_Near_Tet"] = df["Date"].apply(
        lambda d: int(any(abs((d - t).days) <= 10 for t in TET_DATES))
    )
    df["Is_Tet_Zone"] = month.isin([1, 2]).astype(int)

    # Hiệu ứng ngày trong tuần / trong tháng
    df["Is_Monday"] = (dow == 0).astype(int)
    df["Is_Friday"] = (dow == 4).astype(int)
    df["Is_Month_Start"] = (dom <= 5).astype(int)
    df["Is_Month_End"] = (dom >= 25).astype(int)

    # Mã hóa chu kỳ (sin/cos) — tốt hơn label encoding
    df["Month_Sin"] = np.sin(2 * np.pi * month / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * month / 12)
    df["DOW_Sin"] = np.sin(2 * np.pi * dow / 5)
    df["DOW_Cos"] = np.cos(2 * np.pi * dow / 5)

    return df


CALENDAR_FEATURES: List[str] = [
    "Is_Earnings_Season", "Is_Earnings_Peak",
    "Is_Deriv_Expiry", "Is_Pre_Expiry",
    "Is_Quarter_End", "Is_Quarter_End_Week",
    "Is_Near_Tet", "Is_Tet_Zone",
    "Is_Monday", "Is_Friday", "Is_Month_Start", "Is_Month_End",
    "Month_Sin", "Month_Cos", "DOW_Sin", "DOW_Cos",
]  # 16 — khớp Bảng "Feature Groups Summary" (G4)

ALL_FEATURES: List[str] = (
    TECHNICAL_FEATURES + MONEY_FLOW_FEATURES + MACRO_FEATURES + CALENDAR_FEATURES
)  # 21 + 4 + 15 + 16 = 56


# ══════════════════════════════════════════════════════════════════
# Target (nhãn)
# ══════════════════════════════════════════════════════════════════
def compute_target(df: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    """Tạo Next_Return và Target đúng theo Problem Formulation trong
    paper: UP (1) nếu return ngày mai > threshold, ngược lại DOWN (0).

    threshold mặc định = 0.001 (0.1%) — PHẢI khớp với paper và với
    Cell 3 của notebook gốc (`THRESHOLD = 0.001`).
    """
    df = df.copy()
    df["Next_Return"] = df["VN_Return"].shift(-1)
    df["Target"] = (df["Next_Return"] > threshold).astype(int)
    return df


# ══════════════════════════════════════════════════════════════════
# Lắp ráp cuối cùng
# ══════════════════════════════════════════════════════════════════
# Các cột KHÔNG được dùng làm feature vì leak thông tin tương lai
# hoặc là giá tuyệt đối / non-stationary — xem Leakage Analysis (Type 2)
# và Cell 16C_FIX của notebook gốc.
LEAK_COLUMNS: List[str] = ["Next_Return", "VNIndex", "Open", "High", "Low"]

# Cột cross-market GỐC (chưa lag) — đã được thay bằng bản *_lag1,
# nên bị loại khỏi feature list cuối cùng (Cell 16C_FIX).
ORIGINAL_CROSS_MARKET_COLUMNS: List[str] = CROSS_MARKET_NEEDS_LAG


def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Chạy toàn bộ pipeline feature engineering: G1 → G2 → G3 → G4
    → Target, trên dữ liệu thô do `data_collection.py` trả về.
    """
    df = raw_df.sort_values("Date").reset_index(drop=True)
    df = compute_technical_features(df)
    df = compute_money_flow_features(df)
    df = compute_macro_features(df)
    df = compute_calendar_features(df)
    df = compute_target(df)
    return df


def assemble_final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Lọc đúng 56 features + Target + cột phụ trợ, audit NaN, và
    dropna để ra dataset sẵn sàng cho walk-forward validation.
    """
    features = [f for f in ALL_FEATURES if f in df.columns]
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        print(f"⚠️  Thiếu {len(missing)} feature(s) trong df: {missing}")

    print("=== NaN audit trước dropna ===")
    nan_counts = df[features].isna().sum().sort_values(ascending=False)
    has_nan = nan_counts[nan_counts > 0]
    if len(has_nan) > 0:
        print(has_nan.head(15).to_string())
    else:
        print("✅ Không có NaN!")

    keep = (["Date", "VNIndex", "Open", "High", "Low", "Volume"]
            + features + ["Next_Return", "Target"])
    keep = [c for c in dict.fromkeys(keep) if c in df.columns]
    dataset = df[keep].dropna().reset_index(drop=True)

    print(f"\n=== Dataset sau dropna ===")
    print(f"Shape    : {dataset.shape}")
    print(f"Features : {len(features)} (mục tiêu theo paper: 56)")
    if len(dataset) > 0:
        vc = dataset["Target"].value_counts()
        n = len(dataset)
        print(f"Target   : UP={vc.get(1, 0)} ({vc.get(1, 0)/n*100:.1f}%)  "
              f"DOWN={vc.get(0, 0)} ({vc.get(0, 0)/n*100:.1f}%)")

    return dataset


if __name__ == "__main__":
    raw = pd.read_csv("VNIndex_Raw.csv", parse_dates=["Date"])
    full = build_features(raw)
    dataset = assemble_final_dataset(full)
    dataset.to_csv("VNIndex_Features.csv", index=False)
    print("💾 Saved VNIndex_Features.csv")
