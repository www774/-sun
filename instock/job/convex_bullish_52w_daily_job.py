#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每日运行“凸字形 + 多头排列 + 10日内一年新高”选股，并生成日报。"""

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


def _fmt(value):
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _write_daily_markdown(df: pd.DataFrame, run_date: str, path: str):
    lines = [
        f"# {run_date} 当日选股名单",
        "",
        "> 筛选策略：凸字形 + MA5/MA10/MA20 多头排列 + 10日内一年新高。",
        "> 仅用于技术筛选，不构成投资建议。",
        "",
        f"**候选数量：{len(df)}**",
        "",
    ]

    if df.empty:
        lines.append("今日没有股票同时满足全部筛选条件。")
    else:
        lines.extend([
            "| 排名 | 代码 | 名称 | 收盘价 | MA5 | MA10 | MA20 | 一年新高日期 |",
            "|---:|---|---|---:|---:|---:|---:|---|",
        ])
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            lines.append(
                f"| {rank} | {row['code']} | {row['name']} | "
                f"{_fmt(row['close'])} | {_fmt(row['ma5'])} | {_fmt(row['ma10'])} | "
                f"{_fmt(row['ma20'])} | {row['breakout_date']} |"
            )

        lines.extend([
            "",
            "## 技术条件",
            "- 凸字形：MA20 前段持续上升、后段斜率明显放缓，且最新收盘价位于 MA20 上方。",
            "- 多头排列：MA5 > MA10 > MA20，且 MA20 高于 5 个交易日前。",
            "- 一年新高：最近 10 个交易日内至少一次突破此前 252 个交易日最高价。",
        ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


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
        df = df.sort_values(["breakout_date", "code"], ascending=[False, True]).reset_index(drop=True)

    csv_path = os.path.join(REPORT_DIR, f"convex_bullish_52w_{run_date}.csv")
    json_path = os.path.join(REPORT_DIR, f"convex_bullish_52w_{run_date}.json")
    md_path = os.path.join(REPORT_DIR, f"daily_stock_list_{run_date}.md")
    latest_path = os.path.join(REPORT_DIR, "daily_stock_list_latest.md")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    _write_daily_markdown(df, str(run_date), md_path)
    _write_daily_markdown(df, str(run_date), latest_path)

    print(f"运行日期: {run_date}")
    print(f"候选数量: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"当日选股名单: {md_path}")
    print(f"最新选股名单: {latest_path}")
    if rows:
        print(df[["code", "name", "close", "ma5", "ma10", "ma20", "breakout_date"]].to_string(index=False))


if __name__ == "__main__":
    main()
