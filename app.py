
import math
import json
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="加密猎手·市场扫描器 V1.2",
    page_icon="🎯",
    layout="wide",
)

COINGECKO_API = "https://api.coingecko.com/api/v3"
KRAKEN_API = "https://api.kraken.com/0/public"

st.title("🎯 加密猎手 · 市场扫描器 V1.2")
st.caption("免费容错版 · CoinGecko 主数据源 + Kraken 公共K线辅助 · 单个数据源失败不再拖垮整个扫描器")

with st.sidebar:
    st.header("入池规则")
    min_market_cap = st.number_input("最低市值（亿美元）", min_value=0.1, value=3.0, step=0.5) * 100_000_000
    min_volume = st.number_input("最低24H市场成交额（万美元）", min_value=100.0, value=3000.0, step=500.0) * 10_000
    min_amplitude = st.number_input("最低24H振幅（%）", min_value=0.1, value=2.0, step=0.5)
    max_amplitude = st.number_input("最高24H振幅（%）", min_value=2.0, value=20.0, step=1.0)
    min_turnover = st.number_input("最低换手率（%）", min_value=0.0, value=0.5, step=0.1) / 100
    pool_size = st.slider("市场质量池显示数量", 10, 100, 40, 5)
    tech_scan_size = st.slider("技术扫描前N名", 5, 25, 10, 1)

    st.divider()
    st.subheader("CoinGecko 免费接口")
    st.caption("可留空。若出现限流，可填免费的 Demo Key。")
    cg_demo_key = st.text_input("CoinGecko Demo Key（可留空）", type="password")

def get_json(url, params=None, headers=None, timeout=20):
    r = requests.get(
        url,
        params=params,
        headers=headers or {},
        timeout=timeout,
        # 明确UA，降低部分公共API对默认客户端的拦截概率
    )
    r.raise_for_status()
    return r.json()

def cg_headers():
    h = {"User-Agent": "CryptoHunterScanner/1.2"}
    if cg_demo_key.strip():
        h["x-cg-demo-api-key"] = cg_demo_key.strip()
    return h

@st.cache_data(ttl=300, show_spinner=False)
def get_coingecko_markets(headers_json: str):
    headers = json.loads(headers_json)
    rows = []
    # 免费版只拉前500名，已经足够覆盖我们设定的市值与流动性候选池
    for page in (1, 2):
        data = get_json(
            f"{COINGECKO_API}/coins/markets",
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 250,
                "page": page,
                "sparkline": "false",
                "price_change_percentage": "24h",
            },
            headers=headers,
        )
        rows.extend(data)
    return rows

@st.cache_data(ttl=600, show_spinner=False)
def get_kraken_pairs():
    data = get_json(f"{KRAKEN_API}/AssetPairs")
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    return data["result"]

@st.cache_data(ttl=120, show_spinner=False)
def get_kraken_ohlc(pair: str, interval: int):
    data = get_json(
        f"{KRAKEN_API}/OHLC",
        params={"pair": pair, "interval": interval},
    )
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    keys = [k for k in data["result"].keys() if k != "last"]
    if not keys:
        raise RuntimeError("Kraken OHLC无数据")
    return data["result"][keys[0]]

def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))

def score_log(value, low, high, points):
    if value is None or not np.isfinite(value) or value <= low:
        return 0.0
    if value >= high:
        return points
    return points * math.log(value / low) / math.log(high / low)

def sweet_spot(value, left, ideal_left, ideal_right, right, points):
    if value is None or not np.isfinite(value):
        return 0.0
    if value < left or value > right:
        return 0.0
    if ideal_left <= value <= ideal_right:
        return points
    if value < ideal_left:
        return points * (value - left) / max(ideal_left - left, 1e-9)
    return points * (right - value) / max(right - ideal_right, 1e-9)

def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def atr(df, n=14):
    prev = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def kraken_to_df(raw):
    df = pd.DataFrame(
        raw,
        columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"],
    )
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def build_kraken_symbol_map(pairs):
    """
    将 Kraken 的 altname/wsname 映射到常见币种symbol。
    只优先 USD/USDT 报价，避免跨币种技术形态差异。
    """
    result = {}
    for key, item in pairs.items():
        alt = str(item.get("altname", ""))
        ws = str(item.get("wsname", ""))
        quote = str(item.get("quote", ""))

        base_symbol = None
        if "/" in ws:
            base, q = ws.split("/", 1)
            if q in ("USD", "USDT"):
                base_symbol = base
        elif alt.endswith("USD"):
            base_symbol = alt[:-3]
        elif alt.endswith("USDT"):
            base_symbol = alt[:-4]

        if not base_symbol:
            continue

        # Kraken传统XBT对应BTC
        base_symbol = base_symbol.replace("XBT", "BTC")
        # 优先USDT，其次USD
        priority = 2 if ("USDT" in ws or alt.endswith("USDT")) else 1
        old = result.get(base_symbol)
        if old is None or priority > old[1]:
            result[base_symbol] = (alt or key, priority)
    return {k: v[0] for k, v in result.items()}

def technical_state(base_symbol: str, pair_map: Dict[str, str]) -> Dict:
    pair = pair_map.get(base_symbol.upper())
    if not pair:
        return {
            "4H方向": "暂无K线源",
            "1H状态": "-",
            "15M位置": "-",
            "技术机会分": 0,
            "技术数据源": "无",
        }

    # Kraken OHLC支持分钟制interval；240=4小时、60=1小时、15=15分钟
    d4 = kraken_to_df(get_kraken_ohlc(pair, 240))
    d1 = kraken_to_df(get_kraken_ohlc(pair, 60))
    d15 = kraken_to_df(get_kraken_ohlc(pair, 15))

    # Kraken文档说明最后一条OHLC是当前尚未完成周期，因此舍弃最后一行
    d4 = d4.iloc[:-1].copy()
    d1 = d1.iloc[:-1].copy()
    d15 = d15.iloc[:-1].copy()

    if len(d4) < 205 or len(d1) < 60 or len(d15) < 60:
        return {
            "4H方向": "历史不足",
            "1H状态": "-",
            "15M位置": "-",
            "技术机会分": 0,
            "技术数据源": "Kraken",
        }

    d4["ema50"] = ema(d4["close"], 50)
    d4["ema200"] = ema(d4["close"], 200)
    d1["ema20"] = ema(d1["close"], 20)
    d1["ema50"] = ema(d1["close"], 50)
    d1["atr"] = atr(d1)
    d15["ema20"] = ema(d15["close"], 20)
    d15["atr"] = atr(d15)

    x4 = d4.iloc[-1]
    x1 = d1.iloc[-1]
    x15 = d15.iloc[-1]

    if x4["close"] > x4["ema50"] > x4["ema200"]:
        direction = "多头"
    elif x4["close"] < x4["ema50"] < x4["ema200"]:
        direction = "空头"
    else:
        direction = "等待"

    context = "等待"
    if direction == "多头":
        if x1["close"] < x1["ema50"]:
            context = "风险"
        elif x1["ema20"] > x1["ema50"]:
            context = "回调" if abs(x1["close"] - x1["ema20"]) <= x1["atr"] else "趋势"
    elif direction == "空头":
        if x1["close"] > x1["ema50"]:
            context = "风险"
        elif x1["ema20"] < x1["ema50"]:
            context = "回调" if abs(x1["close"] - x1["ema20"]) <= x1["atr"] else "趋势"

    zone_low = x15["ema20"] - 0.50 * x15["atr"]
    zone_high = x15["ema20"] + 0.50 * x15["atr"]
    distance_atr = abs(x15["close"] - x15["ema20"]) / max(x15["atr"], 1e-9)

    if zone_low <= x15["close"] <= zone_high and direction == "多头":
        zone = "多头区域"
    elif zone_low <= x15["close"] <= zone_high and direction == "空头":
        zone = "空头区域"
    elif distance_atr <= 1.0 and direction != "等待":
        zone = "接近区域"
    else:
        zone = "等待"

    score = 0
    if direction != "等待":
        score += 35
    if context == "趋势":
        score += 20
    elif context == "回调":
        score += 35
    if zone in ("多头区域", "空头区域"):
        score += 30
    elif zone == "接近区域":
        score += 15
    if context == "风险":
        score = 0

    return {
        "4H方向": direction,
        "1H状态": context,
        "15M位置": zone,
        "技术机会分": int(clamp(score)),
        "技术数据源": "Kraken",
    }

# -----------------------------
# 主程序：基础市场池必须能独立运行
# -----------------------------
try:
    with st.spinner("正在读取 CoinGecko 免费市场数据……"):
        market_rows = get_coingecko_markets(json.dumps(cg_headers()))

    df = pd.DataFrame(market_rows)
    if df.empty:
        raise RuntimeError("CoinGecko没有返回市场数据。")

    required = [
        "id", "symbol", "name", "market_cap", "total_volume",
        "current_price", "high_24h", "low_24h", "market_cap_rank"
    ]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    df["symbol"] = df["symbol"].astype(str).str.upper()
    for col in ["market_cap", "total_volume", "current_price", "high_24h", "low_24h"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    excluded = {
        "USDT","USDC","USDE","FDUSD","TUSD","DAI","USDP","BUSD",
        "FRAX","PYUSD","EURC"
    }
    df = df[~df["symbol"].isin(excluded)].copy()

    df["市场换手率"] = df["total_volume"] / df["market_cap"].replace(0, np.nan)
    df["24H振幅%"] = (
        (df["high_24h"] - df["low_24h"])
        / df["current_price"].replace(0, np.nan)
        * 100
    )

    hard = (
        (df["market_cap"] >= min_market_cap)
        & (df["total_volume"] >= min_volume)
        & (df["市场换手率"] >= min_turnover)
        & (df["24H振幅%"] >= min_amplitude)
        & (df["24H振幅%"] <= max_amplitude)
    )
    pool = df[hard].copy()

    def quality_score(r):
        # V1.2只使用当前稳定可获得的市场质量因子。
        score = 0.0
        score += score_log(r["market_cap"], 3e8, 5e10, 20)
        score += score_log(r["total_volume"], 3e7, 3e9, 30)
        score += sweet_spot(r["市场换手率"], 0.005, 0.03, 0.35, 1.20, 25)
        score += sweet_spot(r["24H振幅%"], 1.5, 3.0, 10.0, 22.0, 25)
        return int(round(clamp(score)))

    if not pool.empty:
        pool["市场质量分"] = pool.apply(quality_score, axis=1)
        pool = pool.sort_values(
            ["市场质量分", "total_volume", "market_cap"],
            ascending=False,
        ).reset_index(drop=True)

    st.success("✅ CoinGecko 主数据源连接正常。即使技术辅助源不可用，市场质量池仍可正常使用。")

    st.subheader("一、市场质量池")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("读取市场币种", len(df))
    c2.metric("硬条件合格", len(pool))
    c3.metric("重点显示", min(pool_size, len(pool)))
    c4.metric("数据模式", "免费容错")

    if pool.empty:
        st.warning("当前没有币种满足入池条件。可以适当降低左侧筛选门槛。")
        st.stop()

    show = pool.head(pool_size).copy()
    show["排名"] = np.arange(1, len(show) + 1)
    show["市值(亿$)"] = show["market_cap"] / 1e8
    show["24H成交(亿$)"] = show["total_volume"] / 1e8
    show["换手率%"] = show["市场换手率"] * 100

    table = show[
        [
            "排名", "symbol", "name", "市场质量分", "market_cap_rank",
            "市值(亿$)", "24H成交(亿$)", "换手率%", "24H振幅%"
        ]
    ].rename(
        columns={
            "symbol": "币种",
            "name": "名称",
            "market_cap_rank": "市值排名",
        }
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "市场质量分": st.column_config.ProgressColumn(
                "市场质量分", min_value=0, max_value=100, format="%d"
            ),
            "市值(亿$)": st.column_config.NumberColumn(format="%.2f"),
            "24H成交(亿$)": st.column_config.NumberColumn(format="%.2f"),
            "换手率%": st.column_config.NumberColumn(format="%.2f%%"),
            "24H振幅%": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )

    st.info(
        "V1.2 的市场质量分由市值、24H成交额、换手率和24H振幅构成。"
        "它只表示该币是否更适合日内交易，不是胜率，也不是买卖信号。"
    )

    # -----------------------------
    # 技术辅助层：失败也不影响主扫描器
    # -----------------------------
    st.subheader("二、加密猎手·技术候选池")
    st.caption("技术扫描是辅助层。当前优先尝试 Kraken 公共K线；没有对应交易对的币会显示“暂无K线源”，不会导致整个网页报错。")

    if st.button(
        f"🎯 扫描前 {min(tech_scan_size, len(pool))} 名技术状态",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("正在连接技术辅助数据源……"):
                pairs = get_kraken_pairs()
                pair_map = build_kraken_symbol_map(pairs)

            rows = []
            candidates = pool.head(tech_scan_size)
            progress = st.progress(0)

            for idx, (_, r) in enumerate(candidates.iterrows(), start=1):
                try:
                    tech = technical_state(r["symbol"], pair_map)
                except Exception:
                    tech = {
                        "4H方向": "暂不可用",
                        "1H状态": "-",
                        "15M位置": "-",
                        "技术机会分": 0,
                        "技术数据源": "失败",
                    }

                rows.append(
                    {
                        "币种": r["symbol"],
                        "市场质量分": int(r["市场质量分"]),
                        **tech,
                    }
                )
                progress.progress(idx / len(candidates))

            result = pd.DataFrame(rows)
            result["综合关注分"] = (
                result["市场质量分"] * 0.55
                + result["技术机会分"] * 0.45
            ).round().astype(int)

            def level(row):
                if row["技术机会分"] >= 80 and row["市场质量分"] >= 75:
                    return "★★★ 重点"
                if row["技术机会分"] >= 65:
                    return "★★ 观察"
                if row["技术机会分"] >= 50:
                    return "★ 接近"
                return "等待"

            result["状态"] = result.apply(level, axis=1)
            result = result.sort_values(
                ["综合关注分", "技术机会分", "市场质量分"],
                ascending=False,
            ).reset_index(drop=True)

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "市场质量分": st.column_config.ProgressColumn(
                        min_value=0, max_value=100, format="%d"
                    ),
                    "技术机会分": st.column_config.ProgressColumn(
                        min_value=0, max_value=100, format="%d"
                    ),
                    "综合关注分": st.column_config.ProgressColumn(
                        min_value=0, max_value=100, format="%d"
                    ),
                },
            )

            best = result[
                (result["技术机会分"] >= 65)
                & (result["4H方向"].isin(["多头", "空头"]))
            ]

            if len(best):
                st.success(
                    "当前建议打开 TradingView 继续等待 5M / 1M 信号："
                    + "、".join(best["币种"].head(8).tolist())
                )
            else:
                st.warning("当前没有达到技术候选标准的币，或者部分币暂时没有免费的技术K线源。")

        except Exception as e:
            st.warning(
                "技术辅助源暂时不可用，但市场质量池仍然正常。"
                "你仍可优先查看市场质量排名靠前的币。"
            )
            st.caption(f"辅助源信息：{e}")

except requests.HTTPError as e:
    st.error(f"CoinGecko 免费主数据源请求失败：{e}")
    st.caption("可能是短时免费接口限流。稍后刷新，或在左侧填写免费的 CoinGecko Demo Key。")
except Exception as e:
    st.error(f"市场质量池运行失败：{e}")

st.divider()
st.caption(
    "加密猎手·市场扫描器 V1.2｜免费容错版。"
    "主扫描与技术辅助分离：辅助接口故障不会再让整个扫描器停止工作。"
)
