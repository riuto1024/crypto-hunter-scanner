
import math
import json
from typing import Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="加密猎手·市场扫描器 V1.1", page_icon="🎯", layout="wide")

BYBIT_API = "https://api.bybit.com"
COINGECKO_API = "https://api.coingecko.com/api/v3"

st.title("🎯 加密猎手 · 市场扫描器 V1.1")
st.caption("免费版 · CoinGecko + Bybit 公共行情 · 市场质量池 → 技术候选池 → TradingView 等 5M / 1M 精确进场")

with st.sidebar:
    st.header("入池规则")
    min_market_cap = st.number_input("最低市值（亿美元）", min_value=0.1, value=3.0, step=0.5) * 100_000_000
    min_spot_volume = st.number_input("最低24H市场成交额（万美元）", min_value=100.0, value=3000.0, step=500.0) * 10_000
    min_futures_volume = st.number_input("最低24H永续成交额（万美元）", min_value=100.0, value=10000.0, step=1000.0) * 10_000
    max_spread_pct = st.number_input("最大盘口价差（%）", min_value=0.01, value=0.10, step=0.01)
    min_amplitude_pct = st.number_input("最低24H振幅（%）", min_value=0.1, value=2.0, step=0.5)
    max_amplitude_pct = st.number_input("最高24H振幅（%）", min_value=2.0, value=20.0, step=1.0)
    pool_size = st.slider("市场质量池显示数量", 10, 80, 30, 5)
    tech_scan_size = st.slider("技术扫描前N名", 5, 25, 12, 1)
    st.divider()
    st.subheader("免费 CoinGecko")
    st.caption("可留空。若遇 CoinGecko 限流，可填免费 Demo Key。")
    cg_demo_key = st.text_input("CoinGecko Demo Key（可留空）", type="password")

def get_json(url, params=None, headers=None, timeout=20):
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data

def cg_headers():
    return {"x-cg-demo-api-key": cg_demo_key.strip()} if cg_demo_key.strip() else {}

@st.cache_data(ttl=60, show_spinner=False)
def get_bybit_tickers():
    data = get_json(f"{BYBIT_API}/v5/market/tickers", params={"category": "linear"})
    if data.get("retCode") != 0:
        raise RuntimeError(data.get("retMsg", "Bybit ticker接口失败"))
    return data["result"]["list"]

@st.cache_data(ttl=300, show_spinner=False)
def get_bybit_instruments():
    rows = []
    cursor = ""
    while True:
        params = {"category": "linear", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        data = get_json(f"{BYBIT_API}/v5/market/instruments-info", params=params)
        if data.get("retCode") != 0:
            raise RuntimeError(data.get("retMsg", "Bybit instruments接口失败"))
        rows.extend(data["result"]["list"])
        cursor = data["result"].get("nextPageCursor", "")
        if not cursor:
            break
    return rows

@st.cache_data(ttl=300, show_spinner=False)
def get_coingecko_markets(headers_json):
    headers = json.loads(headers_json)
    rows = []
    for page in (1, 2):
        data = get_json(
            f"{COINGECKO_API}/coins/markets",
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 250,
                "page": page,
                "sparkline": "false",
            },
            headers=headers,
        )
        rows.extend(data)
    return rows

@st.cache_data(ttl=120, show_spinner=False)
def get_bybit_klines(symbol, interval, limit=220):
    data = get_json(
        f"{BYBIT_API}/v5/market/kline",
        params={"category": "linear", "symbol": symbol, "interval": interval, "limit": limit},
    )
    if data.get("retCode") != 0:
        raise RuntimeError(data.get("retMsg", "Bybit K线接口失败"))
    return data["result"]["list"]

def klines_to_df(raw):
    # Bybit返回倒序，先转正序。
    raw = list(reversed(raw))
    df = pd.DataFrame(raw, columns=["start","open","high","low","close","volume","turnover"])
    for c in ["open","high","low","close","volume","turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def atr(df, n=14):
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, x))

def score_log(value, low, high, pts):
    if value is None or not np.isfinite(value) or value <= low:
        return 0.0
    if value >= high:
        return pts
    return pts * math.log(value/low) / math.log(high/low)

def sweet_spot(value, left, ideal_left, ideal_right, right, pts):
    if value is None or not np.isfinite(value) or value < left or value > right:
        return 0.0
    if ideal_left <= value <= ideal_right:
        return pts
    if value < ideal_left:
        return pts * (value-left) / max(ideal_left-left, 1e-9)
    return pts * (right-value) / max(right-ideal_right, 1e-9)

def technical_state(symbol):
    d4 = klines_to_df(get_bybit_klines(symbol, "240", 230))
    d1 = klines_to_df(get_bybit_klines(symbol, "60", 230))
    d15 = klines_to_df(get_bybit_klines(symbol, "15", 230))

    # Bybit最后一根可能仍在形成，统一舍弃。
    d4 = d4.iloc[:-1].copy()
    d1 = d1.iloc[:-1].copy()
    d15 = d15.iloc[:-1].copy()

    d4["ema50"] = ema(d4["close"], 50)
    d4["ema200"] = ema(d4["close"], 200)
    d1["ema20"] = ema(d1["close"], 20)
    d1["ema50"] = ema(d1["close"], 50)
    d1["atr"] = atr(d1)
    d15["ema20"] = ema(d15["close"], 20)
    d15["atr"] = atr(d15)

    x4, x1, x15 = d4.iloc[-1], d1.iloc[-1], d15.iloc[-1]

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
            context = "回调" if abs(x1["close"]-x1["ema20"]) <= x1["atr"] else "趋势"
    elif direction == "空头":
        if x1["close"] > x1["ema50"]:
            context = "风险"
        elif x1["ema20"] < x1["ema50"]:
            context = "回调" if abs(x1["close"]-x1["ema20"]) <= x1["atr"] else "趋势"

    zone_low = x15["ema20"] - 0.50 * x15["atr"]
    zone_high = x15["ema20"] + 0.50 * x15["atr"]
    distance_atr = abs(x15["close"]-x15["ema20"]) / max(x15["atr"], 1e-9)

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
    if zone in ("多头区域","空头区域"):
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
    }

try:
    with st.spinner("正在读取免费市场数据……"):
        instruments = get_bybit_instruments()
        tickers = get_bybit_tickers()
        cg = get_coingecko_markets(json.dumps(cg_headers()))

    # Bybit USDT永续
    inst = pd.DataFrame(instruments)
    if inst.empty:
        raise RuntimeError("Bybit交易对数据为空。")

    inst = inst[
        (inst["quoteCoin"] == "USDT")
        & (inst["status"] == "Trading")
        & (inst["contractType"].astype(str).str.contains("Perpetual", na=False))
    ][["symbol","baseCoin"]].drop_duplicates()

    t = pd.DataFrame(tickers)
    if t.empty:
        raise RuntimeError("Bybit行情数据为空。")

    for c in ["lastPrice","turnover24h","highPrice24h","lowPrice24h","bid1Price","ask1Price",
              "fundingRate","openInterestValue"]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")

    t = t.merge(inst, on="symbol", how="inner")

    # CoinGecko按symbol匹配；同symbol保留市值最大者。
    cgdf = pd.DataFrame(cg)
    if cgdf.empty:
        raise RuntimeError("CoinGecko市场数据为空。")
    cgdf["symbol_key"] = cgdf["symbol"].astype(str).str.upper()
    cgdf = cgdf.sort_values("market_cap", ascending=False).drop_duplicates("symbol_key")
    cgkeep = cgdf[["symbol_key","name","market_cap","total_volume","market_cap_rank"]]

    df = t.merge(cgkeep, left_on="baseCoin", right_on="symbol_key", how="inner")

    excluded = {"USDC","USDE","FDUSD","TUSD","DAI","USDP","BUSD","EUR","TRY","BRL"}
    df = df[~df["baseCoin"].isin(excluded)].copy()

    df["市场换手率"] = df["total_volume"] / df["market_cap"].replace(0, np.nan)
    df["永续成交额"] = df["turnover24h"]
    df["24H振幅%"] = (df["highPrice24h"] - df["lowPrice24h"]) / df["lastPrice"].replace(0,np.nan) * 100

    mid = (df["bid1Price"] + df["ask1Price"]) / 2
    df["盘口价差%"] = (df["ask1Price"] - df["bid1Price"]) / mid.replace(0,np.nan) * 100
    df["资金费率%"] = df["fundingRate"].fillna(0) * 100
    df["OI(百万$)"] = df["openInterestValue"] / 1e6

    hard = (
        (df["market_cap"] >= min_market_cap)
        & (df["total_volume"] >= min_spot_volume)
        & (df["永续成交额"] >= min_futures_volume)
        & (df["24H振幅%"] >= min_amplitude_pct)
        & (df["24H振幅%"] <= max_amplitude_pct)
        & (df["盘口价差%"].fillna(999) <= max_spread_pct)
    )
    pool = df[hard].copy()

    def quality_score(r):
        score = 0.0
        score += score_log(r["market_cap"], 3e8, 5e10, 10)
        score += score_log(r["永续成交额"], 1e8, 5e9, 20)
        score += score_log(r["total_volume"], 3e7, 3e9, 15)
        score += sweet_spot(r["市场换手率"], 0.005, 0.03, 0.35, 1.20, 15)
        score += sweet_spot(r["24H振幅%"], 1.5, 3.0, 10.0, 22.0, 15)

        spread = r["盘口价差%"]
        if pd.notna(spread):
            score += 15 if spread <= 0.02 else 12 if spread <= 0.05 else 7 if spread <= 0.10 else 0

        funding = abs(r["资金费率%"])
        score += 10 if funding <= 0.02 else 7 if funding <= 0.05 else 3 if funding <= 0.10 else 0
        return int(round(clamp(score)))

    if not pool.empty:
        pool["市场质量分"] = pool.apply(quality_score, axis=1)
        pool = pool.sort_values(["市场质量分","永续成交额","market_cap"], ascending=False).reset_index(drop=True)

    st.success("免费行情源连接正常：CoinGecko + Bybit")

    st.subheader("一、市场质量池")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Bybit USDT永续", len(inst))
    c2.metric("完成市值匹配", len(df))
    c3.metric("硬条件合格", len(pool))
    c4.metric("重点显示", min(pool_size, len(pool)))

    if pool.empty:
        st.warning("当前没有币种满足入池条件，可以适当降低左侧门槛。")
        st.stop()

    show = pool.head(pool_size).copy()
    show["排名"] = np.arange(1, len(show)+1)
    show["市值(亿$)"] = show["market_cap"]/1e8
    show["24H市场成交(亿$)"] = show["total_volume"]/1e8
    show["永续成交(亿$)"] = show["永续成交额"]/1e8
    show["换手率%"] = show["市场换手率"]*100

    table = show[["排名","baseCoin","name","市场质量分","市值(亿$)","24H市场成交(亿$)",
                  "永续成交(亿$)","换手率%","24H振幅%","盘口价差%","资金费率%","OI(百万$)"]].rename(
        columns={"baseCoin":"币种","name":"名称"}
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.info("市场质量分只代表“是否适合日内交易”，不是胜率，也不是买卖信号。")

    st.subheader("二、加密猎手·技术候选池")
    st.caption("只扫描市场质量排名靠前的币，检查4H / 1H / 15M，降低免费接口压力。")

    if st.button(f"🎯 扫描前 {min(tech_scan_size, len(pool))} 名技术状态", type="primary", use_container_width=True):
        rows = []
        candidates = pool.head(tech_scan_size)
        progress = st.progress(0)

        for idx, (_, r) in enumerate(candidates.iterrows(), start=1):
            try:
                tech = technical_state(r["symbol"])
            except Exception:
                tech = {"4H方向":"数据失败","1H状态":"-","15M位置":"-","技术机会分":0}
            rows.append({
                "币种": r["baseCoin"],
                "市场质量分": int(r["市场质量分"]),
                **tech,
                "资金费率%": r["资金费率%"],
                "OI(百万$)": r["OI(百万$)"],
            })
            progress.progress(idx/len(candidates))

        result = pd.DataFrame(rows)
        result["综合关注分"] = (result["市场质量分"]*0.55 + result["技术机会分"]*0.45).round().astype(int)

        def level(r):
            if r["技术机会分"] >= 80 and r["市场质量分"] >= 75:
                return "★★★ 重点"
            if r["技术机会分"] >= 65:
                return "★★ 观察"
            if r["技术机会分"] >= 50:
                return "★ 接近"
            return "等待"

        result["状态"] = result.apply(level, axis=1)
        result = result.sort_values(["综合关注分","技术机会分","市场质量分"], ascending=False).reset_index(drop=True)
        st.dataframe(result, use_container_width=True, hide_index=True)

        best = result[(result["技术机会分"] >= 65) & result["4H方向"].isin(["多头","空头"])]
        if len(best):
            st.success("当前建议打开 TradingView 继续等待5M/1M信号： " + "、".join(best["币种"].head(8).tolist()))
        else:
            st.warning("当前没有达到技术候选标准的币，不需要勉强交易。")

except requests.HTTPError as e:
    st.error(f"免费行情接口请求失败：{e}")
    st.caption("如果只是 CoinGecko 限流，可稍后刷新或填入免费的 Demo Key。")
except Exception as e:
    st.error(f"程序运行失败：{e}")

st.divider()
st.caption("V1.1：免费行情源采用 CoinGecko + Bybit。市场质量评分与技术机会评分分离；最终执行仍由 TradingView 的“加密猎手·精准交易”完成。")
