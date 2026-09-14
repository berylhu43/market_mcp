"""
eia_mcp — 美国能源信息署(EIA)数据的 MCP server
================================================
和 market_mcp 是两个独立 server:那个查股票(Alpha Vantage),这个查行业数据(EIA)。
一个 agent 可以同时连两个,各查各的。

针对储能赛道选的几个 tool:
  get_capacity        装机容量(电池储能 / 抽水蓄能 / 风 / 光),可按州 —— 赛道规模和竞品势头
  get_generation      发电量构成,可按州 —— 风光渗透率(储能需求的根本驱动)
  get_electricity_price  零售电价,可按州和部门 —— 项目经济性
  eia_explore         探索工具:列出某个 route 有哪些 facet 和可选值(见下方说明)

EIA key(免费): https://www.eia.gov/opendata/register.php

约定同 market_mcp:模型读的文本(docstring / Field description)用英文,# 注释用中文。

安装:
    uv init && rm main.py
    uv add "mcp[cli]"

调试:
    EIA_API_KEY=你的key uv run mcp dev eia_mcp.py

挂 Claude Desktop:
    mcp install eia_mcp.py -v EIA_API_KEY=你的key

!! 关于字段代码的说明
EIA 的 energy_source_code / technology 这些取值很多,而且各数据集不完全一致。
下面 SOURCE_CODES 里的映射是按 EIA 文档常见代码写的,但没在真机验证过。
如果某个查询返回空,用 eia_explore 工具看看那个 route 实际支持哪些 facet 取值,
再回来改 SOURCE_CODES —— 这也是为什么要留那个探索工具。
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from dateutil.relativedelta import relativedelta
from typing import Annotated

from pydantic import Field

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("eia_mcp")

EIA_BASE = "https://api.eia.gov/v2"

_cache: dict[str, dict] = {}
_RETRY_BACKOFF = (2, 5)


# 每个 source → (facet 名, [代码列表])
# 说明:
#  - 煤没有统称代码,按煤种分开传多个
#  - 抽水蓄能的 energy_source_code 也是 WAT,所以靠 prime_mover_code 区分
#  - hydro 用 prime_mover HY(常规水轮机),避免把抽蓄算进来
SOURCE_CODES = {
    "battery":        {"energy_source_code": ["MWH"]},
    "pumped_hydro":   {"energy_source_code": ["WAT"], "prime_mover_code": ["PS"]},
    "hydro":          {"energy_source_code": ["WAT"], "prime_mover_code": ["HY"]},
    "flywheel":       {"prime_mover_code": ["FW"]},
    "compressed_air": {"prime_mover_code": ["CE"]},
    "solar":          {"energy_source_code": ["SUN"]},
    "wind":           {"energy_source_code": ["WND"]},
    "natural_gas":    {"energy_source_code": ["NG"]},
    "nuclear":        {"energy_source_code": ["NUC"]},
    "coal":           {"energy_source_code": ["BIT", "SUB", "LIG", "WC", "RC"]},
    "geothermal":     {"energy_source_code": ["GEO"]},
}

# electric-power-operational-data 用的是 fueltypeid,和 operating-generator-capacity
# 的 energy_source_code 是两套代码,不能混用。
# 以下取值已通过 /facet/fueltypeid 端点验证。
GENERATION_CODES = {
    "solar":        ["SUN"],   # 含公用事业级和分布式
    "solar_utility":["SPV"],   # 仅公用事业级光伏
    "solar_small":  ["DPV"],   # 分布式/屋顶光伏(估算值)
    "wind":         ["WND"],
    "wind_offshore":["WNS"],
    "hydro":        ["HYC"],   # 常规水电
    "pumped_hydro": ["HPS"],   # 抽水蓄能
    "natural_gas":  ["NG"],
    "nuclear":      ["NUC"],
    "coal":         ["COW"],   # 所有煤类
    "geothermal":   ["GEO"],
    "renewables":   ["AOR"],   # 所有可再生,算渗透率很方便
    "fossil":       ["FOS"],
    "all":          ["ALL"],
}

def _api_key() -> str:
    key = os.environ.get("EIA_API_KEY")
    if not key:
        raise RuntimeError(
            "EIA_API_KEY is not set. Get a free key at "
            "https://www.eia.gov/opendata/register.php and pass it with "
            "`mcp install eia_mcp.py -v EIA_API_KEY=<your_key>`."
        )
    return key


def _call_eia(route: str, params: list[tuple[str, str]]) -> dict:
    """Single entry point for every EIA call: attaches the key, caches, retries on
    transient failures, and surfaces EIA's error payloads as clear errors.

    route 例: "electricity/operating-generator-capacity/data"
    params 用 list of tuple 而不是 dict,因为 EIA 的 facets[stateid][] 这类参数
    同一个名字可以出现多次(多选),dict 装不下。
    """
    cache_key = route + "?" + urllib.parse.urlencode(sorted(params))
    if cache_key in _cache:
        return _cache[cache_key]

    for wait in (0, *_RETRY_BACKOFF):
        if wait:
            time.sleep(wait)

        query = urllib.parse.urlencode([*params, ("api_key", _api_key())])
        url = f"{EIA_BASE}/{route}/?{query}"
        try:
            with urllib.request.urlopen(url, timeout=25) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):  # 临时性问题,值得重试
                continue
            body = e.read().decode()[:200] if hasattr(e, "read") else ""
            raise RuntimeError(f"EIA returned HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to reach EIA: {e}") from e

        if "error" in data:
            raise RuntimeError(f"EIA error: {data['error']}")

        _cache[cache_key] = data
        return data

    raise RuntimeError("EIA request kept failing after retries. Try again shortly.")


def _rows(data: dict) -> list[dict]:
    return data.get("response", {}).get("data", [])


def _state_note(state: str | None) -> str:
    return f"{state}" if state else "all US states"


# --- TOOL:装机容量(储能赛道最核心的数字) ---
@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
def get_capacity(
    source: Annotated[
        str,
        Field(
            description="Energy source. One of: battery (grid battery storage), "
            "pumped_hydro, solar, wind, hydro, natural_gas, nuclear, coal, flywheel, compressed_air, geothermal."
        ),
    ],
    state: Annotated[
        str | None,
        Field(
            default=None,
            description="Two-letter state code, e.g. 'CA', 'TX'. Omit for all US states combined.",
        ),
    ] = None,
    months: Annotated[
        int,
        Field(default=12, description="How many recent months of data to return (1-60)", ge=1, le=60),
    ] = 12,
) -> str:
    """Get installed generating/storage capacity in megawatts for an energy source,
    optionally for one state. Use this to size a market or track how fast a
    technology is being built out (e.g. battery storage growth)."""
    facets = SOURCE_CODES.get(source.lower())
    if not facets:
        return f"Unknown source '{source}'. Options: {', '.join(SOURCE_CODES)}"


    # 算出查询窗口:从今天往前推 months 个月。
    start_date = date.today() - relativedelta(months=months + 6)
    start = start_date.strftime("%Y-%m")

    params = [
        ("frequency", "monthly"),
        ("start", start),
        ("data[0]", "nameplate-capacity-mw"),
        ("facets[status][]", "OP"),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "desc"),
        ("length", "5000"),  # 这是机组级明细,一个月可能上百行;5000 是 EIA 单次上限
    ]
    # 不同 facet 之间是"与",同一 facet 的多个值是"或"
    params += [
        (f"facets[{name}][]", code)
        for name, codes in facets.items()
        for code in codes
    ]
    if state:
        params.append(("facets[stateid][]", state.upper()))

    resp = _call_eia("electricity/operating-generator-capacity/data", params).get("response", {})
    rows = resp.get("data", [])
    total = int(resp.get("total") or 0)

    if not rows:
        return (
            f"can't find {source}({_state_note(state)}) capacity data."
            "code error or no data available —— you can use eia_explore to check the available facet values for this route."
        )

    # 同一个 period 可能有多行,按期间汇总
    by_period: dict[str, float] = {}
    for r in rows:
        period = r.get("period", "?")
        try:
            mw = float(r.get("nameplate-capacity-mw") or 0)
        except (TypeError, ValueError):
            continue
        by_period[period] = by_period.get(period, 0.0) + mw

    periods = sorted(by_period, reverse=True)[:months]
    lines = [f"{source} capacity in {_state_note(state)} (MW):"]
    if total > len(rows):
        lines.append(f"  WARNING: got {len(rows)} of {total} rows; data may be incomplete")
    lines += [f"  {p}: {by_period[p]:,.0f} MW" for p in periods]

    if len(periods) >= 2:
        newest, oldest = by_period[periods[0]], by_period[periods[-1]]
        if oldest > 0:
            pct = (newest - oldest) / oldest * 100
            lines.append(f"  period of change({periods[-1]} → {periods[0]}): {pct:+.1f}%")
    return "\n".join(lines)


# --- TOOL:发电量构成(看风光渗透率) ---
@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
def get_generation(
    source: Annotated[
        str,
        Field(description="Energy source: solar, wind, hydro, pumped_hydro, natural_gas, nuclear, coal, all."),
    ],
    state: Annotated[
        str | None,
        Field(default=None, description="Two-letter state code, e.g. 'CA'. Omit for US total."),
    ] = None,
    months: Annotated[
        int,
        Field(default=12, description="How many recent months to return (1-36)", ge=1, le=36),
    ] = 12,
) -> str:
    """Get monthly electricity generation (thousand megawatthours) for an energy
    source, optionally by state. Use this to track renewable penetration — the
    main driver of storage demand."""
    codes = GENERATION_CODES.get(source.lower())
    if not codes:
        return f"Unknown source '{source}'. Options: {', '.join(GENERATION_CODES)}"
    
    params = [
        ("frequency", "monthly"),
        ("data[0]", "generation"),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "desc"),
        ("length", str(months * 2)),
    ]
    params += [("facets[fueltypeid][]", c) for c in codes]

    if state:
        params.append(("facets[location][]", state.upper()))

    rows = _rows(_call_eia("electricity/electric-power-operational-data/data", params))
    if not rows:
        return (
            f"can't find {source}({_state_note(state)}) generation data."
            "You can use eia_explore to check the available facet values for this route."
        )

    lines = [f"{source}({_state_note(state)}) monthly generation (thousand MWh):"]
    seen = set()
    for r in rows:
        period = r.get("period")
        if period in seen:
            continue
        seen.add(period)
        lines.append(f"  {period}: {r.get('generation', '?')}")
        if len(seen) >= months:
            break
    return "\n".join(lines)


# --- TOOL:零售电价(项目经济性) ---
@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
def get_electricity_price(
    state: Annotated[
        str | None,
        Field(default=None, description="Two-letter state code, e.g. 'CA'. Omit for US average."),
    ] = None,
    sector: Annotated[
        str,
        Field(
            default="all",
            description="Customer sector: residential, commercial, industrial, or all.",
        ),
    ] = "all",
    months: Annotated[
        int,
        Field(default=12, description="How many recent months to return (1-36)", ge=1, le=36),
    ] = 12,
) -> str:
    """Get average retail electricity price in cents per kilowatthour, by state and
    customer sector. Higher and more volatile prices generally improve the
    economics of energy storage projects."""
    sector_codes = {
        "residential": "RES",
        "commercial": "COM",
        "industrial": "IND",
        "transportation": "TRA",
        "other": "OTH",
        "all": "ALL",
    }
    sec = sector_codes.get(sector.lower())
    if not sec:
        return f"Unknown sector '{sector}'。Option: {', '.join(sector_codes)}"

    params = [
        ("frequency", "monthly"),
        ("data[0]", "price"),
        ("facets[sectorid][]", sec),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "desc"),
        ("length", str(months * 2)),
    ]
    if state:
        params.append(("facets[stateid][]", state.upper()))

    rows = _rows(_call_eia("electricity/retail-sales/data", params))
    if not rows:
        return f"can't find {_state_note(state)} {sector} electricity price data."

    lines = [f"{_state_note(state)} {sector} average retail electricity price (cents/kWh):"]
    seen = set()
    for r in rows:
        period = r.get("period")
        if period in seen:
            continue
        seen.add(period)
        lines.append(f"  {period}: {r.get('price', '?')}")
        if len(seen) >= months:
            break
    return "\n".join(lines)


# --- TOOL:即将退役的机组(储能选址线索) ---
@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
def get_planned_retirements(
    state: Annotated[
        str | None,
        Field(default=None, description="Two-letter state code, e.g. 'TX'. Omit for all US."),
    ] = None,
    years_ahead: Annotated[
        int,
        Field(default=5, description="Include retirements planned within this many years (1-15)", ge=1, le=15),
    ] = 5,
    source: Annotated[
        str | None,
        Field(
            default=None,
            description="Optional energy source filter: coal, natural_gas, nuclear, hydro, "
            "solar, wind, battery, pumped_hydro, geothermal.",
        ),
    ] = None,
) -> str:
    """List power generators with a planned retirement date, including capacity,
    fuel type, retirement date, and county. Retiring plants are strong candidates
    for storage siting: they already have grid interconnection and leave a
    capacity gap that needs filling."""
    # 只查最近几个月的快照就够了(退役计划是当前状态,不是时间序列),
    # 缩短窗口能避免撞 5000 行上限。
    start_date = date.today() - relativedelta(months=8)
    params = [
        ("frequency", "monthly"),
        ("start", start_date.strftime("%Y-%m")),
        ("data[0]", "nameplate-capacity-mw"),
        ("data[1]", "planned-retirement-year-month"),
        ("data[2]", "county"),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "desc"),
        ("length", "5000"),
    ]
    if state:
        params.append(("facets[stateid][]", state.upper()))
    if source:
        facets = SOURCE_CODES.get(source.lower())
        if not facets:
            return f"Unknown source '{source}'. Options: {', '.join(SOURCE_CODES)}"
        # 不同 facet 之间是"与",同一 facet 的多个值是"或"
        params += [
            (f"facets[{name}][]", code)
            for name, codes in facets.items()
            for code in codes
        ]

    resp = _call_eia("electricity/operating-generator-capacity/data", params).get("response", {})
    rows = resp.get("data", [])
    total_rows = int(resp.get("total") or 0)
    if not rows:
        return f"No generator data found for {_state_note(state)}."

    # 这个数据集是"每月快照":同一台机组每个月都出现一行。
    # 只保留最新那个月的快照,否则同一台机组会被重复统计。
    latest = max((r.get("period") or "") for r in rows)

    cutoff = ""  # 退役日期形如 "2029-06";用字符串比较即可,不必解析日期
    try:
        cutoff = f"{int(latest[:4]) + years_ahead}-12"
    except (ValueError, IndexError):
        pass

    seen: set[tuple] = set()
    found = []
    for r in rows:
        if r.get("period") != latest:
            continue
        retire = r.get("planned-retirement-year-month")
        if not retire or (cutoff and retire > cutoff):
            continue
        key = (r.get("plantid"), r.get("generatorid"))
        if key in seen:
            continue
        seen.add(key)
        try:
            mw = float(r.get("nameplate-capacity-mw") or 0)
        except (TypeError, ValueError):
            mw = 0.0
        found.append({
            "retire": retire,
            "mw": mw,
            "plant": r.get("plantName") or r.get("plantid") or "?",
            "county": r.get("county") or "?",
            "fuel": r.get("energy_source_code") or "?",
        })

    if not found:
        return f"No generators scheduled to retire within {years_ahead} years in {_state_note(state)}."

    found.sort(key=lambda x: x["retire"])
    total = sum(f["mw"] for f in found)

    lines = [
        f"Planned retirements in {_state_note(state)} within {years_ahead} years "
        f"(as of {latest}; {len(found)} units, {total:,.0f} MW total):"
    ]
    for f in found[:30]:
        lines.append(
            f"  {f['retire']} | {f['mw']:>7,.0f} MW | {f['fuel']:<4} | "
            f"{f['plant']} ({f['county']} County)"
        )
    if len(found) > 30:
        lines.append(f"  ... and {len(found) - 30} more units not shown")
    if total_rows > len(rows):
        lines.append(f"  WARNING: got {len(rows)} of {total_rows} rows; may be incomplete")
    return "\n".join(lines)


# --- TOOL:按维度拆解装机(看竞争格局 / 技术路线 / 电网分布) ---
# by 参数 → (EIA 里用于分组的字段, 显示名字段)
BREAKDOWN_FIELDS = {
    "technology": ("technology", "technology"),
    "entity": ("entityid", "entityName"),
    "balancing_authority": ("balancing_authority_code", "balancing-authority-name"),
    "sector": ("sector", "sectorName"),
    "plant": ("plantid", "plantName"),
}


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
def get_capacity_breakdown(
    source: Annotated[
        str,
            Field(description="Energy source: battery, pumped_hydro, flywheel, compressed_air, "
              "solar, wind, hydro, natural_gas, nuclear, coal, geothermal."),
    ],
    by: Annotated[
        str,
        Field(
            description="How to group the capacity. One of: "
            "technology (battery vs pumped hydro etc), "
            "entity (owner/developer — use this to see who is building the market), "
            "balancing_authority (grid region such as ERCOT or CAISO), "
            "sector (utility vs independent power producer), "
            "plant (individual facilities)."
        ),
    ],
    state: Annotated[
        str | None,
        Field(default=None, description="Two-letter state code, e.g. 'TX'. Omit for all US."),
    ] = None,
    top: Annotated[
        int,
        Field(default=15, description="How many groups to show, ranked by capacity (1-40)", ge=1, le=40),
    ] = 15,
) -> str:
    """Break down installed capacity by owner, technology, grid region, sector, or
    plant, ranked largest first. Use this to see competitive landscape (who owns
    the capacity), technology mix, or geographic concentration — questions that a
    single total MW number cannot answer."""
    facets = SOURCE_CODES.get(source.lower())
    if not facets:
        return f"Unknown source '{source}'. Options: {', '.join(SOURCE_CODES)}"

    fields = BREAKDOWN_FIELDS.get(by.lower())
    if not fields:
        return f"Unknown 'by' value '{by}'. Options: {', '.join(BREAKDOWN_FIELDS)}"
    key_field, name_field = fields

    # 只要最近一个月的快照就够了(这是存量数据,不是时间序列),
    # 少取几个月能避免撞 5000 行上限。
    start_date = date.today() - relativedelta(months=8)
    params = [
        ("frequency", "monthly"),
        ("start", start_date.strftime("%Y-%m")),
        ("data[0]", "nameplate-capacity-mw"),
        ("facets[status][]", "OP"),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "desc"),
        ("length", "5000"),
    ]
    # 不同 facet 之间是"与",同一 facet 的多个值是"或"
    params += [
        (f"facets[{name}][]", code)
        for name, codes in facets.items()
        for code in codes
    ]
    if state:
        params.append(("facets[stateid][]", state.upper()))

    resp = _call_eia("electricity/operating-generator-capacity/data", params).get("response", {})
    rows = resp.get("data", [])
    total_rows = int(resp.get("total") or 0)
    if not rows:
        return f"No {source} capacity found for {_state_note(state)}."

    # 每月快照:只保留最新月份,否则同一台机组会被重复统计
    latest = max((r.get("period") or "") for r in rows)

    groups: dict[str, float] = {}
    units: dict[str, int] = {}
    for r in rows:
        if r.get("period") != latest:
            continue
        label = r.get(name_field) or r.get(key_field) or "(unknown)"
        try:
            mw = float(r.get("nameplate-capacity-mw") or 0)
        except (TypeError, ValueError):
            continue
        groups[label] = groups.get(label, 0.0) + mw
        units[label] = units.get(label, 0) + 1

    if not groups:
        return f"No {source} capacity found for {_state_note(state)} in {latest}."

    ranked = sorted(groups.items(), key=lambda kv: kv[1], reverse=True)
    grand_total = sum(groups.values())

    lines = [
        f"{source} capacity in {_state_note(state)} by {by} "
        f"(as of {latest}, {grand_total:,.0f} MW across {len(groups)} groups):"
    ]
    for label, mw in ranked[:top]:
        share = mw / grand_total * 100 if grand_total else 0
        lines.append(f"  {mw:>8,.0f} MW ({share:>4.1f}%) | {units[label]:>3} units | {label}")
    if len(ranked) > top:
        rest = sum(mw for _, mw in ranked[top:])
        lines.append(f"  {rest:>8,.0f} MW ({rest / grand_total * 100:>4.1f}%) | others ({len(ranked) - top} groups)")
    if total_rows > len(rows):
        lines.append(f"  WARNING: got {len(rows)} of {total_rows} rows; may be incomplete")
    return "\n".join(lines)



# --- TOOL:探索工具(排查用) ---
@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
def eia_explore(
    route: Annotated[
        str,
        Field(
            description="An EIA v2 route path, e.g. 'electricity' or "
            "'electricity/operating-generator-capacity'. Omit the /data suffix."
        ),
    ],
) -> str:
    """Explore what a given EIA route offers: its child routes, available
    frequencies, facets, and data columns. Use this when another tool returns no
    data, to check which facet values actually exist for that dataset."""
    data = _call_eia(route.strip("/"), [])
    resp = data.get("response", {})

    parts = []
    if resp.get("routes"):
        parts.append("子路由:\n" + "\n".join(
            f"  {r.get('id')} — {r.get('name', '')}" for r in resp["routes"][:20]
        ))
    if resp.get("facets"):
        parts.append("可用 facet:\n" + "\n".join(
            f"  {f.get('id')} — {f.get('description', '')}" for f in resp["facets"]
        ))
    if resp.get("frequency"):
        freqs = [f.get("id") for f in resp["frequency"]]
        parts.append(f"频率: {', '.join(str(f) for f in freqs)}")
    if resp.get("data"):
        parts.append(f"数据列: {', '.join(resp['data'].keys())}")

    return "\n\n".join(parts) if parts else f"{route} 没返回可用的元数据。"


if __name__ == "__main__":
    mcp.run()
