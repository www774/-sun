#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""凸字形 + 均线多头 + 10日内一年新高。

程序化定义：
1. 凸字形：20日均线近20个交易日持续向上，但后半段斜率明显低于前半段，形成
   “上升 -> 放缓/平台”的凸形结构；同时最新收盘价在MA20上方。
2. 多头排列：MA5 > MA10 > MA20，且MA20较5个交易日前上行。
3. 一年新高：最近10个交易日内，至少有一天最高价突破当日之前252个交易日的最高价。

这是技术筛选器，不构成投资建议。
"""

import numpy as np
import pandas as pd
import talib as tl


def _upward_convex_shape(data: pd.DataFrame) -> bool:
    if len(data) < 40:
        return False

    ma20 = tl.SMA(data["close"].astype(float).values, timeperiod=20)
    ma20 = pd.Series(ma20, index=data.index).replace([np.inf, -np.inf], np.nan).dropna()
    if len(ma20) < 20:
        return False

    recent = ma20.iloc[-20:]
    first_slope = (recent.iloc[9] - recent.iloc[0]) / 9.0
    second_slope = (recent.iloc[-1] - recent.iloc[10]) / 9.0

    # 上升但后段明显放缓，避免把连续暴涨当成“凸字形”。
    if first_slope <= 0 or second_slope <= 0:
        return False
    if second_slope >= first_slope * 0.80:
        return False

    return float(data.iloc[-1]["close"]) > float(ma20.iloc[-1])


def _bullish_ma(data: pd.DataFrame) -> bool:
    if len(data) < 30:
        return False
    close = data["close"].astype(float).values
    ma5 = tl.SMA(close, timeperiod=5)
    ma10 = tl.SMA(close, timeperiod=10)
    ma20 = tl.SMA(close, timeperiod=20)
    if any(np.isnan(x[-1]) for x in (ma5, ma10, ma20)):
        return False
    if not (ma5[-1] > ma10[-1] > ma20[-1]):
        return False
    if ma20[-1] <= ma20[-6]:
        return False
    return True


def _new_high_within_10d(data: pd.DataFrame, lookback: int = 252, window: int = 10) -> bool:
    if len(data) < lookback + 1:
        return False
    highs = data["high"].astype(float).reset_index(drop=True)
    start = max(lookback, len(highs) - window)
    for i in range(start, len(highs)):
        previous_high = highs.iloc[i - lookback:i].max()
        if pd.notna(previous_high) and highs.iloc[i] >= previous_high:
            return True
    return False


def check(code_name, data, date=None, lookback: int = 252, window: int = 10) -> bool:
    """返回是否同时满足三项条件。"""
    if data is None or len(data.index) < lookback + 30:
        return False

    data = data.copy()
    if "date" in data.columns:
        end_date = code_name[0] if date is None else date.strftime("%Y-%m-%d")
        data["date"] = data["date"].astype(str)
        data = data.loc[data["date"] <= str(end_date)].copy()
    if len(data.index) < lookback + 30:
        return False
    data = data.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    return (
        _upward_convex_shape(data)
        and _bullish_ma(data)
        and _new_high_within_10d(data, lookback=lookback, window=window)
    )


def score(code_name, data, date=None, lookback: int = 252, window: int = 10):
    """返回可解释的指标，用于日报排序。"""
    if data is None:
        return None
    data = data.copy()
    if "date" in data.columns:
        end_date = code_name[0] if date is None else date.strftime("%Y-%m-%d")
        data["date"] = data["date"].astype(str)
        data = data.loc[data["date"] <= str(end_date)].copy()
    if len(data) < lookback + 30:
        return None
    data = data.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    close = data["close"].astype(float).values
    ma5 = tl.SMA(close, 5)
    ma10 = tl.SMA(close, 10)
    ma20 = tl.SMA(close, 20)
    ma20s = pd.Series(ma20).dropna()
    recent = ma20s.iloc[-20:]
    first_slope = (recent.iloc[9] - recent.iloc[0]) / 9.0
    second_slope = (recent.iloc[-1] - recent.iloc[10]) / 9.0

    highs = data["high"].astype(float).reset_index(drop=True)
    breakout_days = []
    for i in range(max(lookback, len(highs) - window), len(highs)):
        if highs.iloc[i] >= highs.iloc[i-lookback:i].max():
            breakout_days.append(str(data.iloc[i]["date"]))

    return {
        "convex": bool(first_slope > 0 and second_slope > 0 and second_slope < first_slope * 0.80 and close[-1] > ma20[-1]),
        "bullish_ma": bool(ma5[-1] > ma10[-1] > ma20[-1] and ma20[-1] > ma20[-6]),
        "year_high_10d": bool(breakout_days),
        "close": float(close[-1]),
        "ma5": float(ma5[-1]),
        "ma10": float(ma10[-1]),
        "ma20": float(ma20[-1]),
        "ma20_slope_first": float(first_slope),
        "ma20_slope_second": float(second_slope),
        "breakout_date": breakout_days[-1] if breakout_days else "",
    }
