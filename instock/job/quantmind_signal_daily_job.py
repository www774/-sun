#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a ranked daily technical signal report from the existing stock data."""

import json
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.append(ROOT)

from instock.core.singleton_stock import stock_hist_data, stock_data
from instock.core.strategy.convex_bullish_52w import check
from instock.core.strategy.quantmind_signal import calculate_signal
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
            metrics = calculate_signal(data)
            if metrics.get("score", 0) < 70:
                continue
            code = str(code_name[1])
            rows.append({
                "date": str(run_date),
                "code": code,
                "name": name_map.get(code, ""),
                **metrics,
            })
        except Exception as exc:
            print(f"skip {code_name}: {exc}")

    os.makedirs(REPORT_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["score", "code"], ascending=[False, True]).reset_index(drop=True)
        top10 = df.head(10).copy()
    else:
        top10 = df

    csv_path = os.path.join(REPORT_DIR, f"quantmind_signal_{run_date}.csv")
    json_path = os.path.join(REPORT_DIR, f"quantmind_signal_{run_date}.json")
    md_path = os.path.join(REPORT_DIR, f"quantmind_signal_{run_date}.md")
    latest_path = os.path.join(REPORT_DIR, "quantmind_signal_latest.md")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, default=str)

    lines = [
        f"# {run_date} QuantMind 风格技术信号",
        "",
        "> 筛选入口：凸字形 + MA5/MA10/MA20多头 + 10日内一年新高。",
        "> 评分：均线15 + 新高15 + 凸字15 + 放量15 + 布林10 + 底部放量10 + 回踩MA20 10，并进行波动率风险调整。",
        "> 当前版本为可解释技术评分层，尚未接入机器学习预测分。仅用于技术筛选，不构成投资建议。",
        "",
        f"**候选数量：{len(df)}；Top 10：{len(top10)}**",
        "",
        "| 排名 | 代码 | 名称 | 综合分 | 收盘 | MA5 | MA10 | MA20 | 放量倍数 | 布林宽度 | 风险 | 信号 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for rank, (_, row) in enumerate(top10.iterrows(), start=1):
        vol = "" if pd.isna(row.get("volume_ratio")) else f"{row['volume_ratio']:.2f}x"
        bw = "" if pd.isna(row.get("boll_width")) else f"{row['boll_width']:.3f}"
        lines.append(
            f"| {rank} | {row['code']} | {row['name']} | {int(row['score'])} | "
            f"{row['close']:.2f} | {row['ma5']:.2f} | {row['ma10']:.2f} | {row['ma20']:.2f} | "
            f"{vol} | {bw} | {row['risk']} | {row['signal']} |"
        )

    lines.extend(["", "## 信号说明", "- 放量突破：当日成交量达到近20日均量1.5倍且收盘不弱。", "- 布林收窄：当前布林带宽度处于近期偏低区间且价格站在中轨上方。", "- 底部三倍放量：成交量达到20日均量3倍，并且价格仍处于近60日低位附近。", "- 回踩MA20：收盘相对MA20处于-1%至+8%的可控区域。", "- 风险：结合ATR波动率和距离MA20的乖离度进行提示。", ""])
    text = "\n".join(lines)
    for path in (md_path, latest_path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    print(f"运行日期: {run_date}")
    print(f"候选数量: {len(df)}")
    print(f"Top10: {len(top10)}")
    print(f"报告: {md_path}")


if __name__ == "__main__":
    main()
