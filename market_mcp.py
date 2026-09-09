"""
market_mcp — 用 Alpha Vantage 做市场数据的 MCP server
=====================================================
主要结构：几个 tool + 一个 resource + 一个 prompt。

它提供的是"事实数据"(报价、基本面),不给买卖建议。

需要一个 Alpha Vantage 免费 key:
    https://www.alphavantage.co/support/#api-key

!! 免费额度很紧:每天约 25 次、每分钟 5 次。所以本文件做了两件事:
   - compare_symbols 每支票只调 1 次 API(只查基本面 OVERVIEW),别一次比太多支
   - 加了进程内缓存,同一次会话里重复查同一支不再重复请求

安装依赖(HTTP 只用标准库 urllib,不依赖任何第三方 HTTP 包):
    pip install "mcp[cli]"

本地调试(key 通过环境变量传,别写死在代码里):
    ALPHAVANTAGE_API_KEY=你的key uv run mcp dev market_mcp.py

挂到 Claude Desktop(-v 把 key 作为环境变量传进去):
    mcp install market_mcp.py -v ALPHAVANTAGE_API_KEY=你的key
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Annotated

from pydantic import Field

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("market_mcp")

AV_BASE = "https://www.alphavantage.co/query"

# 进程内缓存:key 是请求参数拼成的字符串,value 是返回的 JSON。
# 免费额度很小,同一会话里重复查同一支就命中缓存,不再花配额。
# 注意:进程重启会清空(Claude Desktop 每次会话都会重新起这个进程)。
_cache: dict[str, dict] = {}


def _api_key() -> str:
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        raise RuntimeError(
            "No ALPHAVANTAGE_API_KEY set. Please pass it via "
            "`mcp install market_mcp.py -v ALPHAVANTAGE_API_KEY=yourkey`."
        )
    return key


def _call_av(params: dict) -> dict:
    """统一请求入口:带上 key、走缓存、并把 Alpha Vantage 的各种'非正常返回'翻译成清楚的报错。"""
    cache_key = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    if cache_key in _cache:
        return _cache[cache_key]

    query = urllib.parse.urlencode({**params, "apikey": _api_key()})
    try:
        with urllib.request.urlopen(f"{AV_BASE}?{query}", timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        raise RuntimeError(f"failed to call Alpha Vantage: {e}") from e

    # Alpha Vantage 不用 HTTP 错误码报错,而是在 JSON 里塞这几个字段:
    if "Error Message" in data:
        raise RuntimeError(f"Alpha Vantage error (bug in code): {data['Error Message']}")
    if "Note" in data or "Information" in data:
        raise RuntimeError(
            "triggered Alpha Vantage rate limit (free tier allows ~25 calls/day, ~5 calls/minute)."
            "wait a bit and try again, or reduce the number of stocks compared at once."
        )

    _cache[cache_key] = data
    return data


def _fetch_overview(symbol: str) -> dict:
    """Fetch company fundamentals. Shared by compare_symbols and
    get_company_overview (and shares the cache)."""
    data = _call_av({"function": "OVERVIEW", "symbol": symbol})
    return data if data.get("Symbol") else {}


# --- TOOL:公司名 → 股票代码 ---
@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
def search_symbol(
    query: Annotated[str, Field(description="Company name or keyword, e.g. 'Tesla', 'Exxon', 'solar'")],
) -> str:
    """Resolve a company name or keyword into a stock ticker. Use this first
    when you are unsure of a company's symbol."""
    data = _call_av({"function": "SYMBOL_SEARCH", "keywords": query})
    matches = data.get("bestMatches", [])
    if not matches:
        return f"no stocks found matching '{query}'."
    lines = [
        f"{m.get('1. symbol')} — {m.get('2. name')} ({m.get('4. region')})"
        for m in matches[:5]
    ]
    return "\n".join(lines)


# --- TOOL:单支实时报价 ---
@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
def get_quote(
    symbol: Annotated[str, Field(description="Stock ticker, e.g. 'AAPL'. If unsure, use search_symbol first.")],
) -> str:
    """Get the current quote for a single stock: latest price, absolute change, percent change, and volume."""
    q = _call_av({"function": "GLOBAL_QUOTE", "symbol": symbol}).get("Global Quote", {})
    if not q:
        return f"no quote found for {symbol}, please verify the symbol (you can use search_symbol)."
    return (
        f"{symbol}: current price ${q.get('05. price')}, "
        f"change {q.get('09. change')} ({q.get('10. change percent')}), "
        f"volume {q.get('06. volume')}, data date {q.get('07. latest trading day')}"
    )


# --- TOOL:单支公司基本面 ---
@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
def get_company_overview(
    symbol: Annotated[str, Field(description="Stock ticker, e.g. 'XOM'")],
) -> str:
    """Get fundamentals for a single company: sector, market cap, P/E ratio,
    profit margin, and 52-week range."""

    o = _fetch_overview(symbol)
    if not o:
        return f"no company overview found for {symbol} (possibly not a US stock or incorrect symbol)."
    return (
        f"{o.get('Name')} ({symbol})\n"
        f"sector: {o.get('Sector')} / {o.get('Industry')}\n"
        f"market cap: {o.get('MarketCapitalization')}\n"
        f"PE ratio: {o.get('PERatio')}\n"
        f"profit margin: {o.get('ProfitMargin')}\n"
        f"52-week range: {o.get('52WeekLow')} - {o.get('52WeekHigh')}"
    )


# --- TOOL:横向对比(核心功能) ---
@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
def compare_symbols(
    symbols: Annotated[
        list[str],
        Field(
            description="A list of stock tickers to compare side by side, e.g. ['NEE','FSLR','ENPH']",
            min_length=2,
            max_length=5,
        ),
    ],
) -> str:
    """Compare fundamentals of several same-sector stocks side by side: sector,
    market cap, P/E ratio, profit margin. Calls the API once per symbol
    (OVERVIEW); the free tier is small, so keep it to a few symbols at a time."""
    rows = []
    for sym in symbols:
        try:
            o = _fetch_overview(sym)
        except RuntimeError as e:
            rows.append(f"{sym}: failed to fetch data ({e})")
            continue
        if not o:
            rows.append(f"{sym}: no company overview found (symbol might be incorrect)")
            continue
        rows.append(
            f"{o.get('Name', '?')} ({sym})\n"
            f"  sector: {o.get('Sector', '?')}\n"
            f"  market cap: {o.get('MarketCapitalization', '?')}\n"
            f"  PE: {o.get('PERatio', '?')} | profit margin: {o.get('ProfitMargin', '?')}"
        )
    return "\n\n".join(rows)


# --- RESOURCE:关注列表(可编辑的背景上下文) ---
@mcp.resource("market://watchlist")
def watchlist() -> str:
    """An editable watchlist provided as background context. Change it to the
    tickers you actually care about."""
    return "watchlist(energy sector as example, modify as needed): NEE, XOM, FSLR, ENPH, CEG"


# --- PROMPT:板块横向对比模板 ---
@mcp.prompt(name="compare_sector")
def compare_sector(names: str = "") -> str:
    """Build a prompt that asks the model to compare a set of same-sector
    companies side by side."""
    target = names or "companies in the same sector (e.g. energy, solar, EV, etc.)"
    return (
        f"please compare {target} side by side. steps:\n"
        "1. if a company name is given instead of a symbol, use search_symbol to find the code;\n"
        "2. use compare_symbols to fetch each company's sector, market cap, P/E ratio, and profit margin;\n"
        "3. summarize the differences in scale and valuation objectively.\n"
        "only state the facts, do not provide buy/sell recommendations."
    )


if __name__ == "__main__":
    mcp.run()