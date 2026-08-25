#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每日运行“凸字形 + 多头排列 + 10日内一年新高”选股。"""

import datetime as dt
import json
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.append(ROOT)

from instock.core.singleton_stock import stock_hist_data, stock_data
from instock.core.strategy.convex_bullish_52w import check, score
from instock.lib.trade_time import get_trade_date_last


REPORT_DIR = os.path.join(ROOT, "reports")


def main():
    run_date, _ = get_trade_date_last()
    stocks = stock_hist_data(date=run_date).get_data()
    if not stocks:
        raise RuntimeError("无法取得历史行情数据")

    spot = stock_data(run_date).get_data()
    name_map = {}
    if spot is not None and not spot.empty:
        name_map = dict(zip(spot["code"].astype(str), spot["name"].astype(str)))

    rows = []
    for code_name, data in stocks.items():
        try:
            if not check(code_name, data, date=run_date):
                continue
            metrics = score(code_name, data, date=run_date)
            if not metrics:
                continue
            code = str(code_name[1])
            rows.append({
                "date": str(run_date),
                "code": code,
                "name": name_map.get(code, ""),
                "close": metrics["close"],
                "ma5": metrics["ma5"],
                "ma10": metrics["ma10"],
                "ma20": metrics["ma20"],
                "convex": metrics["convex"],
                "bullish_ma": metrics["bullish_ma"],
                "year_high_10d": metrics["year_high_10d"],
                "breakout_date": metrics["breakout_date"],
                "ma20_slope_first": metrics["ma20_slope_first"],
                "ma20_slope_second": metrics["ma20_slope_second"],
            })
        except Exception as exc:
            print(f"skip {code_name}: {exc}")

    os.makedirs(REPORT_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["breakout_date", "code"], ascending=[False, True])

    csv_path = os.path.join(REPORT_DIR, f"convex_bullish_52w_{run_date}.csv")
    json_path = os.path.join(REPORT_DIR, f"convex_bullish_52w_{run_date}.json")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"运行日期: {run_date}")
    print(f"候选数量: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    if rows:
        print(df[["code", "name", "close", "ma5", "ma10", "ma20", "breakout_date"]].to_string(index=False))


if __name__ == "__main__":
    main()
