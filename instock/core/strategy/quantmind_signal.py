#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QuantMind-style explainable technical scoring layer.

This module is deliberately independent from any AI provider. It converts the
existing convex-bullish universe into a 0-100 technical score and exposes
features that can later be consumed by Qlib/ML models.
"""

import numpy as np
import pandas as pd
import talib as tl


def _num(data, col):
    return pd.to_numeric(data[col], errors="coerce").astype(float)


def calculate_signal(data: pd.DataFrame) -> dict:
    if data is None or len(data) < 260:
        return {"score": 0, "risk": "高", "signal": "数据不足"}

    d = data.copy().sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    close = _num(d, "close").values
    high = _num(d, "high").values
    low = _num(d, "low").values
    volume = _num(d, "volume").values if "volume" in d.columns else None

    ma5 = tl.SMA(close, 5)
    ma10 = tl.SMA(close, 10)
    ma20 = tl.SMA(close, 20)
    upper, middle, lower = tl.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    atr = tl.ATR(high, low, close, timeperiod=14)

    i = len(d) - 1
    score = 0
    features = {}

    # 15: MA5/10/20 bullish alignment.
    bullish = bool(ma5[i] > ma10[i] > ma20[i] and ma20[i] > ma20[i-5])
    score += 15 if bullish else 0
    features["bullish_ma"] = bullish

    # 15: one-year high within the latest 10 sessions.
    breakout_date = ""
    for j in range(max(252, len(d) - 10), len(d)):
        prior = np.nanmax(high[j-252:j])
        if np.isfinite(prior) and high[j] >= prior:
            breakout_date = str(d.iloc[j]["date"])
    year_high = bool(breakout_date)
    score += 15 if year_high else 0
    features["year_high_10d"] = year_high
    features["breakout_date"] = breakout_date

    # 15: convex shape, consistent with the existing strategy definition.
    ma20s = pd.Series(ma20).dropna()
    convex = False
    if len(ma20s) >= 20:
        r = ma20s.iloc[-20:]
        s1 = (r.iloc[9] - r.iloc[0]) / 9.0
        s2 = (r.iloc[-1] - r.iloc[10]) / 9.0
        convex = bool(s1 > 0 and s2 > 0 and s2 < s1 * 0.80 and close[i] > ma20[i])
    score += 15 if convex else 0
    features["convex"] = convex

    # 15: volume breakout.
    volume_ratio = np.nan
    volume_breakout = False
    if volume is not None and i >= 20 and np.isfinite(volume[i]):
        avg20 = np.nanmean(volume[i-20:i])
        if avg20 > 0:
            volume_ratio = volume[i] / avg20
            volume_breakout = bool(volume_ratio >= 1.5 and close[i] >= close[i-1])
    score += 15 if volume_breakout else 0
    features["volume_ratio"] = float(volume_ratio) if np.isfinite(volume_ratio) else None
    features["volume_breakout"] = volume_breakout

    # 10: Bollinger squeeze + expansion.
    bandwidth = np.nan
    squeeze = False
    if np.isfinite(upper[i]) and np.isfinite(lower[i]) and middle[i] != 0:
        bandwidth = (upper[i] - lower[i]) / abs(middle[i])
        recent_bw = (upper[max(0, i-20):i+1] - lower[max(0, i-20):i+1]) / np.abs(middle[max(0, i-20):i+1])
        recent_bw = recent_bw[np.isfinite(recent_bw)]
        squeeze = bool(len(recent_bw) >= 10 and bandwidth <= np.nanpercentile(recent_bw, 35) and close[i] >= middle[i])
    score += 10 if squeeze else 0
    features["boll_width"] = float(bandwidth) if np.isfinite(bandwidth) else None
    features["boll_squeeze"] = squeeze

    # 10: bottom-volume burst (>=3x 20-day average), treated as a supporting signal.
    bottom_3x = False
    if volume is not None and i >= 20 and np.isfinite(volume[i]):
        avg20 = np.nanmean(volume[i-20:i])
        low_60 = np.nanmin(close[max(0, i-60):i])
        bottom_3x = bool(avg20 > 0 and volume[i] >= 3 * avg20 and close[i] <= low_60 * 1.20)
    score += 10 if bottom_3x else 0
    features["bottom_3x_volume"] = bottom_3x

    # 10: pullback holds MA20 (or is currently above it with modest extension).
    distance_ma20 = (close[i] / ma20[i] - 1) if ma20[i] else np.nan
    pullback_hold = bool(np.isfinite(distance_ma20) and -0.01 <= distance_ma20 <= 0.08)
    score += 10 if pullback_hold else 0
    features["distance_ma20"] = float(distance_ma20) if np.isfinite(distance_ma20) else None
    features["pullback_hold"] = pullback_hold

    # 10: volatility/risk adjustment. This is not an AI score; it is a guardrail.
    risk = "低"
    if np.isfinite(atr[i]) and close[i] > 0:
        atr_pct = atr[i] / close[i]
        if atr_pct > 0.08 or (np.isfinite(distance_ma20) and distance_ma20 > 0.15):
            risk = "高"
            score = max(0, score - 5)
        elif atr_pct > 0.05 or (np.isfinite(distance_ma20) and distance_ma20 > 0.10):
            risk = "中"
    features["atr_pct"] = float(atr[i] / close[i]) if np.isfinite(atr[i]) and close[i] else None

    if score >= 90:
        signal = "★★★★★ 强势候选"
    elif score >= 80:
        signal = "★★★★ 重点关注"
    elif score >= 70:
        signal = "★★★ 等待确认"
    else:
        signal = "暂不考虑"

    return {
        "score": int(score),
        "risk": risk,
        "signal": signal,
        "close": float(close[i]),
        "ma5": float(ma5[i]),
        "ma10": float(ma10[i]),
        "ma20": float(ma20[i]),
        **features,
    }
