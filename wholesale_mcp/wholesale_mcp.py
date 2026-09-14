"""
wholesale_mcp — 批发电价(LMP)的 MCP server
==============================================
补上 EIA 缺的那一块:批发市场的日内价格波动,也就是储能套利的真正来源。

为什么单独一个 server:数据源不同(抓 ISO 官网 vs EIA 官方 API)、依赖不同
(要 gridstatus + pandas)、稳定性不同(ISO 改版就可能坏)。让它自己崩,
别拖累已经跑通的 eia_mcp。

概念速记:
  ISO/RTO  区域电力批发市场的运营方。CAISO=加州, ERCOT=德州。
  LMP      节点边际电价,每个电网节点各有各的价(输电拥堵导致)。
           ERCOT 管自己的叫 SPP(结算点价格),是一回事。
  日前市场  提前一天竞价约定次日每小时价格,较平稳。
  实时市场  当天每 5-15 分钟结算,波动极大。
  储能赚的是同一天内低价充电、高价放电的差 —— 两头都是批发价,
  和零售电价无关。

不需要 API key(开源版直接抓 ISO 公开数据)。

安装:
    uv init && rm main.py
    uv add "mcp[cli]" gridstatus

调试:
    uv run mcp dev wholesale_mcp.py

挂 Claude Desktop:
    mcp install wholesale_mcp.py

注意:一次拉全部节点的数据会有几十万行,所以本文件只查交易枢纽
(trading hub),不查全节点。
"""

from datetime import date, timedelta
from typing import Annotated

import gridstatus
import pandas as pd
from gridstatus import Markets
from pydantic import Field

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("wholesale_mcp")

# 支持的市场。每个 ISO 的取价方式不一样,所以配置里带上取数函数用的参数。
#   CAISO 用 get_lmp + 三个交易枢纽(NP15 北加州, SP15 南加州, ZP26 中部)
#   ERCOT 用 get_spp + location_type="Trading Hub"
ISOS = {
    "caiso": {
        "name": "California ISO (CAISO)",
        "hubs": ["TH_NP15_GEN-APND", "TH_SP15_GEN-APND", "TH_ZP26_GEN-APND"],
        "price_col": "LMP",
        "tz": "US/Pacific",
    },
    "ercot": {
        "name": "Electric Reliability Council of Texas (ERCOT)",
        "hubs": None,
        "price_col": "SPP",
        "tz": "US/Central",
    },
}

# 进程内缓存:同一天同一个市场的数据不会变,重复查直接命中。
_cache: dict[str, pd.DataFrame] = {}


def _fetch_prices(iso: str, day: str, real_time: bool) -> pd.DataFrame:
    """拉某个 ISO 某一天的枢纽价格。返回含 Interval Start / Location / <price_col> 的 DataFrame。"""
    cache_key = f"{iso}|{day}|{'rt' if real_time else 'da'}"
    if cache_key in _cache:
        return _cache[cache_key]

    cfg = ISOS[iso]
    market = Markets.REAL_TIME_15_MIN if real_time else Markets.DAY_AHEAD_HOURLY

    if iso == "caiso":
        df = gridstatus.CAISO().get_lmp(date=day, market=market, locations=cfg["hubs"])
    else:
        df = gridstatus.Ercot().get_spp(date=day, market=market, location_type="Trading Hub")

    _cache[cache_key] = df
    return df


def _resolve_day(day: str | None) -> str:
    """没指定日期就用前天 —— 日前市场数据当天才公布,往前留两天最稳。"""
    if day:
        return day
    return (date.today() - timedelta(days=2)).strftime("%Y-%m-%d")


# --- TOOL:单个市场的日内价差(套利空间) ---
@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
def get_lmp_spread(
    iso: Annotated[
        str,
        Field(description="Wholesale market: 'caiso' (California) or 'ercot' (Texas)."),
    ],
    day: Annotated[
        str | None,
        Field(
            default=None,
            description="Date as YYYY-MM-DD. Defaults to two days ago, since "
            "day-ahead data for very recent dates may not be published yet.",
        ),
    ] = None,
    real_time: Annotated[
        bool,
        Field(
            default=False,
            description="False (default) for day-ahead hourly prices — sufficient "
            "for most arbitrage analysis and much faster to fetch. True for "
            "real-time 15-minute prices, which capture short spikes but take "
            "significantly longer to download.",
        ),
    ] = False,
) -> str:
    """Get intraday wholesale electricity price spread at a market's trading hubs:
    daily min, max, average, and peak-to-trough spread in $/MWh. This spread is
    what energy storage arbitrages — charging at the low, discharging at the high.
    Retail prices do not measure this."""
    iso_key = iso.lower()
    if iso_key not in ISOS:
        return f"Unknown iso '{iso}'. Options: {', '.join(ISOS)}"

    target_day = _resolve_day(day)
    try:
        df = _fetch_prices(iso_key, target_day, real_time)
    except Exception as e:
        hint = (
            "Network could not reach the ISO. Check connectivity and retry."
            if "NameResolution" in str(e) or "Max retries" in str(e)
            else "Try an earlier date — recent days may not be published yet."
        )
        return f"Could not fetch {iso.upper()} prices for {target_day}: {type(e).__name__}. {hint}"

    price_col = ISOS[iso_key]["price_col"]
    if df is None or df.empty or price_col not in df.columns:
        return f"No price data returned for {iso.upper()} on {target_day}."

    market_label = "real-time 15-min" if real_time else "day-ahead hourly"
    lines = [f"{ISOS[iso_key]['name']} — {market_label} prices, {target_day} ($/MWh):"]

    for location, group in df.groupby("Location"):
        prices = pd.to_numeric(group[price_col], errors="coerce").dropna()
        if prices.empty:
            continue
        low, high, avg = prices.min(), prices.max(), prices.mean()
        spread = high - low
        # 找出最高/最低出现在什么时候 —— 储能关心的是"几点充、几点放"
        # 统一转成 ISO 本地时区再显示。gridstatus 返回的时间戳可能是 UTC,
        # 直接 strftime 会把傍晚 18:45 显示成次日凌晨 01:45。
        tz = ISOS[iso_key]["tz"]
        low_time = pd.Timestamp(group.loc[prices.idxmin(), "Interval Start"])
        high_time = pd.Timestamp(group.loc[prices.idxmax(), "Interval Start"])
        if low_time.tz is not None:
            low_time = low_time.tz_convert(tz)
            high_time = high_time.tz_convert(tz)
        lines.append(
            f"  {location}\n"
            f"    low  ${low:>8.2f} at {low_time.strftime('%H:%M')}\n"
            f"    high ${high:>8.2f} at {high_time.strftime('%H:%M')}\n"
            f"    avg  ${avg:>8.2f} | spread ${spread:,.2f}"
        )

    if len(lines) == 1:
        return f"No usable price rows for {iso.upper()} on {target_day}."

    lines.append(
        "  (Spread = same-day high minus low. Storage charges near the low and "
        "discharges near the high; this is the arbitrage opportunity.)"
    )
    return "\n".join(lines)


# --- TOOL:跨市场对比波动性 ---
@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
def compare_lmp_volatility(
    day: Annotated[
        str | None,
        Field(default=None, description="Date as YYYY-MM-DD. Defaults to two days ago."),
    ] = None,
    real_time: Annotated[
        bool,
        Field(default=False, description="True for real-time prices instead of day-ahead."),
    ] = False,
) -> str:
    """Compare intraday price spreads across CAISO and ERCOT on the same day.
    Wider and more volatile spreads mean better storage economics, so this is a
    direct way to compare how attractive each market is for a storage project."""
    target_day = _resolve_day(day)
    market_label = "real-time 15-min" if real_time else "day-ahead hourly"
    lines = [f"Intraday price spread by market — {market_label}, {target_day} ($/MWh):"]
    any_data = False

    for iso_key, cfg in ISOS.items():
        try:
            df = _fetch_prices(iso_key, target_day, real_time)
        except Exception as e:
            lines.append(f"  {cfg['name']}: fetch failed ({type(e).__name__})")
            continue

        price_col = cfg["price_col"]
        if df is None or df.empty or price_col not in df.columns:
            lines.append(f"  {cfg['name']}: no data")
            continue

        prices = pd.to_numeric(df[price_col], errors="coerce").dropna()
        if prices.empty:
            lines.append(f"  {cfg['name']}: no usable prices")
            continue

        any_data = True
        spread = prices.max() - prices.min()
        lines.append(
            f"  {cfg['name']}\n"
            f"    range ${prices.min():.2f} – ${prices.max():.2f} | "
            f"spread ${spread:,.2f} | avg ${prices.mean():.2f}\n"
            f"    hubs: {df['Location'].nunique()} | intervals: {len(df)}"
        )

    if not any_data:
        return (
            f"No price data available for {target_day} in either market. "
            "Try an earlier date."
        )

    lines.append(
        "  (Note: spread here is across all hubs in the market, so it reflects both "
        "time-of-day and location differences. Use get_lmp_spread for one hub at a time.)"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
