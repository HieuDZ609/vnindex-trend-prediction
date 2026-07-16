"""
data_collection.py
===================
Thu thập dữ liệu thô cho bài toán dự đoán xu hướng VNIndex.

Nguồn dữ liệu:
    - VNIndex OHLCV           : vnstock (v4.x, nguồn KBS/VCI, fallback vnstock 3.x)
    - Global markets          : yfinance (S&P 500, NASDAQ, Gold, Oil, DXY,
                                VIX, Hang Seng, Nikkei 225, US 3M T-bill)
    - Foreign net buy         : vnstock API (fallback: proxy FUEVFVND ETF)
    - Sector proxies          : yfinance (VCB, VHM, HPG)

Refactor từ notebook gốc `VNIndex_Trend_Prediction.ipynb`
(Cell 1 → Cell 4, và Cell 17A cho phần VIX/HSI/Nikkei).

File này CHỈ thu thập + merge dữ liệu thô theo đúng ngày (Date).
Toàn bộ việc tính đặc trưng (kể cả sửa lệch múi giờ / lag) nằm ở
`feature_engineering.py` — xem docstring ở đó để biết chi tiết.
"""

from __future__ import annotations

import warnings
from functools import reduce
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────
# Cấu hình mặc định (khớp với notebook gốc)
# ────────────────────────────────────────────────────────────────────
START = "2014-01-01"   # Từ 2014: VN30F đã có, data tương đối đầy đủ
END = "2025-06-01"

# Ticker toàn cầu lấy qua yfinance (Cell 3 gốc)
GLOBAL_TICKERS: Dict[str, str] = {
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Gold": "GC=F",
    "Oil": "CL=F",
    "DXY": "DX-Y.NYB",
    "VIX": "^VIX",
    "HSI": "^HSI",
    "Nikkei": "^N225",
    "US3M": "^IRX",
}
# Lưu ý: US3M (^IRX) được fetch để tham khảo (đúng như notebook gốc)
# nhưng KHÔNG nằm trong 56 features cuối cùng dùng cho model — xem
# MACRO_FEATURES trong feature_engineering.py.

SECTOR_PROXIES: Dict[str, str] = {
    "VCB_R": "VCB.VN",
    "VHM_R": "VHM.VN",
    "HPG_R": "HPG.VN",
}


def _normalize_date(series: pd.Series) -> pd.Series:
    """Chuẩn hóa cột ngày: bỏ timezone, cắt giờ về 00:00:00.

    Bắt buộc phải làm bước này: VNIndex (giờ Việt Nam) và các mã
    global (giờ Mỹ/HK/Nhật, timezone khác nhau) phải được đưa về
    cùng một "ngày lịch" trước khi merge — nếu không, lệch ngày sẽ
    tạo NaN hàng loạt sau khi join hai bên.
    """
    s = pd.to_datetime(series)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s.dt.normalize()


def fetch_vnindex(start: str = START, end: str = END) -> pd.DataFrame:
    """Lấy OHLCV của VNIndex từ vnstock.

    Thử vnstock 4.x (`vnstock.api.quote.Quote`) với nguồn KBS rồi VCI
    trước; nếu cả hai lỗi, fallback về vnstock 3.x (`vnstock.Quote`).
    Đúng thứ tự thử như Cell 3 của notebook gốc.

    Returns
    -------
    pd.DataFrame
        Cột: Date, Open, High, Low, VNIndex, Volume.
    """
    raw = None
    for source in ["KBS", "VCI"]:
        try:
            from vnstock.api.quote import Quote as QuoteNew
            q = QuoteNew(symbol="VNINDEX", source=source)
            raw = q.history(start=start, end=end, interval="1D")
            print(f"✅ vnstock.api.quote source={source} OK")
            break
        except Exception as e:
            print(f"  vnstock.api {source}: {e}")

    if raw is None:
        try:
            from vnstock import Quote
            q = Quote(source="kbs", symbol="VNINDEX")
            raw = q.history(start=start, end=end, interval="1D")
            print("✅ vnstock 3.x Quote OK")
        except Exception as e:
            raise RuntimeError(f"Không lấy được VNIndex: {e}")

    raw = raw.rename(columns={
        "time": "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "VNIndex", "volume": "Volume",
    })
    raw["Date"] = _normalize_date(raw["Date"])

    df = raw.sort_values("Date").reset_index(drop=True)
    df = df[["Date", "Open", "High", "Low", "VNIndex", "Volume"]].copy()
    for col in ["Open", "High", "Low", "VNIndex", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["VNIndex"]).reset_index(drop=True)

    assert len(df) > 500, "Quá ít phiên giao dịch lấy được!"
    print(f"✅ VNIndex: {len(df)} phiên | "
          f"{df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


def _yf_close_and_return(ticker: str, col: str, start: str,
                          end: str) -> Optional[pd.DataFrame]:
    """Tải giá đóng cửa + % thay đổi của 1 mã từ yfinance.

    Trả về DataFrame với cột Date, <col> (giá đóng cửa), <col>_Return
    (% thay đổi ngày). Trả về None nếu tải lỗi hoặc quá ít dữ liệu
    (< 100 phiên) — hàm gọi sẽ tự điền 0.0, đúng hành vi Cell 3 gốc.
    """
    try:
        raw = yf.download(ticker, start=start, end=end,
                           progress=False, auto_adjust=True)
        if len(raw) < 100:
            raise ValueError(f"Ít dữ liệu ({len(raw)})")
        s = raw[["Close"]].copy()
        s.columns = [col]
        s[f"{col}_Return"] = s[col].pct_change()
        s.index = _normalize_date(pd.Series(s.index)).values
        s.index.name = "Date"
        print(f"  ✅ {ticker:15s} ({col}): {len(raw)} ngày")
        return s.reset_index()
    except Exception as e:
        print(f"  ⚠️  {ticker:15s} ({col}): {e}")
        return None


def fetch_global_markets(df: pd.DataFrame, start: str = START,
                          end: str = END) -> pd.DataFrame:
    """Merge dữ liệu thị trường toàn cầu vào `df` theo cột Date.

    QUAN TRỌNG: hàm này chỉ TẢI VÀ GHÉP dữ liệu thô (giá đóng cửa +
    % thay đổi thô, cùng ngày lịch T — CHƯA xử lý lệch múi giờ).
    Việc lag 1 ngày cho SP500/NASDAQ/Gold/Oil/DXY (bắt buộc để tránh
    look-ahead bias — xem Leakage Analysis Type 1 trong paper) được
    thực hiện ở `feature_engineering.py::fix_cross_market_lag`, KHÔNG
    làm ở đây, để giữ đúng ranh giới "thu thập dữ liệu" và
    "kỹ thuật đặc trưng".

    Returns
    -------
    pd.DataFrame
        `df` đã merge thêm cột <Ticker> và <Ticker>_Return cho mỗi
        mã trong GLOBAL_TICKERS.
    """
    df = df.copy()
    print("📡 Fetching global markets từ yfinance...")
    for col, ticker in GLOBAL_TICKERS.items():
        merged = _yf_close_and_return(ticker, col, start, end)
        if merged is not None:
            df = df.merge(merged, on="Date", how="left")
        else:
            df[col] = 0.0
            df[f"{col}_Return"] = 0.0

    return_cols = (list(GLOBAL_TICKERS.keys())
                   + [f"{k}_Return" for k in GLOBAL_TICKERS])
    exist_cols = [c for c in return_cols if c in df.columns]
    df[exist_cols] = df[exist_cols].ffill()
    return df


def fetch_foreign_net_buy(df: pd.DataFrame, start: str = START,
                           end: str = END) -> pd.DataFrame:
    """Lấy dữ liệu khối ngoại mua ròng (Foreign Net Buy).

    Thử vnstock API mới trước (nguồn KBS rồi VCI). Nếu không lấy
    được, fallback dùng % thay đổi giá của ETF FUEVFVND làm proxy —
    đây là quyết định có chủ đích, được ghi rõ trong paper (proxy có
    khoảng 84% giá trị bằng 0 trước năm 2019, do ETF ít giao dịch
    thời gian đầu). Nếu cả hai đều thất bại, cột được điền 0.0.

    Cột `Foreign_Net_Buy` trả về ở đây là dữ liệu THÔ — cột này
    KHÔNG được dùng trực tiếp làm feature (xem
    `feature_engineering.py::MONEY_FLOW_FEATURES`); chỉ 2 cột phái
    sinh Foreign_Net_Ratio / Foreign_Net_MA5 mới được đưa vào model.
    """
    df = df.copy()
    foreign_ok = False

    for source in ["KBS", "VCI"]:
        try:
            from vnstock.api.quote import Quote as QuoteNew
            q_foreign = QuoteNew(symbol="HOSE", source=source)
            fn = q_foreign.foreign(start=start, end=end)
            fn["Date"] = _normalize_date(pd.to_datetime(fn.get("time", fn.index)))
            net_col = next((c for c in fn.columns
                             if "net" in c.lower() and "value" in c.lower()), None)
            if net_col:
                fn = fn.rename(columns={net_col: "Foreign_Net_Buy"})
                fn = fn[["Date", "Foreign_Net_Buy"]].dropna()
                df.drop(columns=["Foreign_Net_Buy"], errors="ignore", inplace=True)
                df = df.merge(fn, on="Date", how="left")
                df["Foreign_Net_Buy"] = df["Foreign_Net_Buy"].fillna(0)
                foreign_ok = True
                print(f"✅ Foreign Net Buy: vnstock API mới ({source})")
                break
        except Exception as e:
            print(f"  vnstock API {source}: {e}")

    if not foreign_ok:
        try:
            etf = yf.download("FUEVFVND.VN", start=start, end=end,
                               progress=False, auto_adjust=True)
            if len(etf) > 200:
                s = etf[["Close"]].pct_change().copy()
                s.columns = ["Foreign_Net_Buy"]
                s.index = _normalize_date(pd.Series(etf.index)).values
                s.index.name = "Date"
                df.drop(columns=["Foreign_Net_Buy"], errors="ignore", inplace=True)
                df = df.merge(s.reset_index(), on="Date", how="left")
                df["Foreign_Net_Buy"] = df["Foreign_Net_Buy"].ffill().fillna(0)
                foreign_ok = True
                n_match = (df["Foreign_Net_Buy"] != 0).sum()
                print(f"✅ Foreign Net Buy: proxy FUEVFVND ({n_match} ngày khớp)")
            else:
                raise ValueError("Ít dữ liệu")
        except Exception as e2:
            print(f"⚠️  FUEVFVND lỗi: {e2}")

    if not foreign_ok:
        df["Foreign_Net_Buy"] = 0.0

    return df


def fetch_sector_proxies(start: str = START, end: str = END) -> pd.DataFrame:
    """Lấy return của 3 mã đại diện ngành (VCB, VHM, HPG) từ yfinance.

    Dùng để tính Sector_Momentum / Sector_Divergence trong
    `feature_engineering.py`. Trả về DataFrame chỉ có cột Date (rỗng)
    nếu không mã nào tải được — bước feature engineering sẽ tự điền 0.
    """
    print("📡 Fetching sector proxies (VCB, VHM, HPG)...")
    frames = {}
    for col, ticker in SECTOR_PROXIES.items():
        try:
            raw = yf.download(ticker, start=start, end=end,
                               progress=False, auto_adjust=True)
            if len(raw) > 200:
                r = raw[["Close"]].pct_change().copy()
                r.columns = [col]
                r.index = _normalize_date(pd.Series(raw.index)).values
                r.index.name = "Date"
                frames[col] = r.reset_index()
                print(f"   ✅ {ticker}: {len(raw)} ngày")
            else:
                print(f"   ⚠️  {ticker}: ít dữ liệu ({len(raw)})")
        except Exception as e:
            print(f"   ⚠️  {ticker}: {e}")

    if not frames:
        return pd.DataFrame(columns=["Date"])

    sec_df = reduce(lambda a, b: pd.merge(a, b, on="Date", how="outer"),
                     frames.values())
    sec_df["Date"] = pd.to_datetime(sec_df["Date"]).dt.normalize()
    return sec_df


def collect_raw_dataset(start: str = START, end: str = END) -> pd.DataFrame:
    """Chạy toàn bộ pipeline thu thập dữ liệu thô.

    Thứ tự: VNIndex → global markets → foreign net buy → sector
    proxies, tất cả merge theo Date. Kết quả sẵn sàng để đưa vào
    `feature_engineering.py::build_features`.
    """
    print(f"Giai đoạn thu thập: {start} → {end}")
    df = fetch_vnindex(start, end)
    df = fetch_global_markets(df, start, end)
    df = fetch_foreign_net_buy(df, start, end)

    sector_df = fetch_sector_proxies(start, end)
    if not sector_df.empty:
        df = df.merge(sector_df, on="Date", how="left")
        ffill_cols = [c for c in SECTOR_PROXIES if c in df.columns]
        df[ffill_cols] = df[ffill_cols].ffill()

    print(f"\n✅ Raw dataset: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


if __name__ == "__main__":
    dataset = collect_raw_dataset()
    dataset.to_csv("VNIndex_Raw.csv", index=False)
    print(f"💾 Saved VNIndex_Raw.csv ({dataset.shape[0]} rows × {dataset.shape[1]} cols)")
