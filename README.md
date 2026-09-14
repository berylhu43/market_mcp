# Energy Market Research Agent

An MCP-based research agent for the US energy storage market. It combines
equities data (Alpha Vantage) with industry data (US Energy Information
Administration) so an LLM can answer questions that need both — market sizing,
competitive landscape, siting opportunities, and company fundamentals.

Built as three independent pieces: two MCP servers that expose data as tools,
and a LangGraph agent that orchestrates them.

---

## Why it's structured this way

**Servers are capabilities; the agent is one application.** Each MCP server
wraps one data source and can be reused by any MCP client — Claude Desktop, the
agent in this repo, or something else later. Swapping the orchestration layer
(LangGraph → something else) doesn't touch the servers.

**One server per data source.** `alphavantage_mcp` and `eia_mcp` are separate
processes with separate credentials, separate rate limits, and separate failure
modes. If EIA changes a field name, equities tools keep working.

---

## Layout

```
market_mcp/
├── alphavantage_mcp/       # equities: quotes, fundamentals, news
│   └── alphavantage_mcp.py
├── eia_mcp/                # industry: capacity, generation, prices
│   └── eia_mcp.py
└── market_agent/           # LangGraph agent + eval harness
    ├── market_agent.py
    └── eval_agent.py
```

Each folder is its own `uv` project with its own `.venv` and `.env`.

---

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

API keys (both free):
- Alpha Vantage — https://www.alphavantage.co/support/#api-key
- EIA — https://www.eia.gov/opendata/register.php
- Anthropic (for the agent) — https://console.anthropic.com

```bash
# servers
cd alphavantage_mcp && uv init && uv add "mcp[cli]"
cd ../eia_mcp && uv init && uv add "mcp[cli]" python-dateutil

# agent
cd ../market_agent && uv init
uv add langgraph langchain langchain-anthropic langchain-mcp-adapters python-dotenv
```

Copy `.env.example` to `.env` in each folder and fill in the keys.

---

## Running

**As an agent (CLI):**

```bash
cd market_agent
uv run python market_agent.py "compare battery storage capacity in Texas and California"
```

**In Claude Desktop:**

```bash
cd alphavantage_mcp && mcp install alphavantage_mcp.py -v ALPHAVANTAGE_API_KEY=<key>
cd ../eia_mcp && mcp install eia_mcp.py -v EIA_API_KEY=<key>
```

Fully quit and reopen Claude Desktop afterwards. Editing server code afterwards
needs only a restart — `mcp install` records a launch command, not a copy of the
code. Reinstall only if the filename, path, keys, or dependencies change.

**Debugging a server in isolation:**

```bash
cd eia_mcp && EIA_API_KEY=<key> uv run mcp dev eia_mcp.py
```

This opens the MCP Inspector, where each tool can be called by hand. Much faster
than round-tripping through the agent.

---

## Tools

### `alphavantage_mcp`

| Tool | Purpose |
|---|---|
| `search_symbol` | Company name → ticker |
| `get_quote` | Current price, change, volume |
| `get_company_overview` | Sector, market cap, P/E, margin, 52-week range |
| `compare_symbols` | Fundamentals for 2–5 tickers side by side |
| `get_news` | Headlines with sentiment, by ticker and/or topic |

### `eia_mcp`

| Tool | Purpose |
|---|---|
| `get_capacity` | Installed MW by source, optionally by state, over time |
| `get_capacity_breakdown` | Same capacity grouped by owner / plant / technology / grid region / sector |
| `get_generation` | Monthly generation (thousand MWh) by source and state |
| `get_electricity_price` | Average retail price (cents/kWh) by state and sector |
| `get_planned_retirements` | Generators with a scheduled retirement date, with capacity and county |
| `eia_explore` | Lists a route's facets, data columns, and valid facet values |

`eia_explore` is a debugging tool, and it earns its place: EIA's facet names and
codes differ between datasets, so when a query returns nothing, this is how you
find out what the dataset actually accepts.

---

## Things learned the hard way

These are written down because each one produced wrong-looking-right output at
some point.

**Facet codes are per-dataset, not global.** `operating-generator-capacity` uses
`energy_source_code` / `prime_mover_code` / `stateid`;
`electric-power-operational-data` uses `fueltypeid` / `location`. The same code
string can exist in both with different meanings — `COL` means "coal excluding
waste coal" in one dataset and doesn't exist at all in the other. Two separate
code tables (`SOURCE_CODES`, `GENERATION_CODES`) exist for this reason.

**Pumped hydro is not an energy source.** Its `energy_source_code` is `WAT`
(water), same as conventional hydro. What distinguishes it is
`prime_mover_code = PS`. Conventional hydro is `WAT` + `HY`. Filtering on the
wrong facet silently returns zero rows, which reads as "no data" rather than
"wrong query."

**Coal has no single code.** In the capacity dataset it's `BIT`, `SUB`, `LIG`,
`WC`, `RC` — multiple values on the same facet, which EIA treats as OR.
Different facets are ANDed together.

**`operating-generator-capacity` is a monthly snapshot, not a time series.**
Every generator reappears in every month. Any aggregation must dedupe on
`(plantid, generatorid)` within a single period, or totals multiply by the number
of months fetched.

**Row limits produce plausible wrong answers.** EIA caps responses at 5000 rows.
Because results are sorted newest-first, truncation silently drops the oldest
months — which makes growth rates look larger than they are. Always compare
`response.total` against the number of rows returned; EIA's own `warnings` field
fires even when nothing was truncated, so it isn't a reliable signal. Constrain
with `start` rather than fetching everything and slicing locally.

**Retail price is not the storage arbitrage spread.** Retail prices include
transmission, distribution, and fixed costs, and are averaged monthly. Storage
arbitrage happens in the wholesale market, intraday. Retail data answers "is this
an expensive market," not "how much can a battery earn." EIA's API does not
expose wholesale LMPs — those live in EIA's Wholesale Electricity Market Portal
or via ISO feeds.

---

## Evaluation

```bash
cd market_agent
uv run python eval_agent.py                    # all cases
uv run python eval_agent.py --case quote_basic # one case
```

Agent output is non-deterministic, so the eval asserts on three things rather
than exact text:

1. **Trajectory** — which tools were called, with what arguments, how many times.
   This layer is deterministic and catches the most failures.
2. **Content** — whether the final answer contains values that came from the
   tool, which catches the model answering from memory instead of calling a tool.
3. **Behavior** — how it handles bad tickers, out-of-scope questions, and
   requests for investment advice.

Half the cases are boundary cases on purpose. The failure mode that matters most
is not a crash — it's the model filling a data gap with a confident guess that
looks identical to a real answer.

---

## Limitations

- Alpha Vantage's free tier allows roughly 25 calls/day and 5/min. A single
  multi-part question can exhaust it. Both servers cache in-process, but the
  cache dies with the process — every CLI run starts cold.
- `get_news` topic filtering is coarse; `energy_transportation` returns a lot of
  unrelated transport news. Querying by ticker works better.
- Private companies don't appear in equities data at all, which matters in
  storage, where many players are pre-IPO. Ownership breakdowns from EIA
  partially fill this gap, since they name operators regardless of listing status.
- EIA groups by legal entity, not corporate parent — subsidiaries appear
  separately from their parent companies.

---

## Not investment advice

These tools return public data. Any interpretation in an agent's response is
model-generated commentary, not analysis by a licensed advisor.