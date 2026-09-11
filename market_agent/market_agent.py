"""
market_agent.py — 用 LangGraph 搭一个连 market_mcp 的最小 agent
================================================================
目的:跟你手写过的 while-loop 做对照,看清 LangGraph 到底接管了哪几块。

手写 → LangGraph 的逐条对应:
  你手写的 while 循环本身        →  一张图(StateGraph)+ 它的 runtime
  循环里"把对话发给模型"那步     →  agent 节点(call_model)
  循环里"执行模型点名的 tool"    →  tools 节点(ToolNode,自动执行)
  "模型还想调 tool 吗?"判断      →  条件边(有 tool_calls 去 tools,否则 END)
  tool 执行完再回去问模型        →  一条 tools -> agent 的回边(这条就是"循环")
  你手动往 messages 里 append    →  add_messages reducer 自动累加(MessagesState 内建)
  你手动把 MCP tool 包成函数     →  MultiServerMCPClient 自动转成 LangChain tool

目录结构假设(agent 和 server 是兄弟文件夹):
    Desktop/
    ├── market_mcp/     market_mcp.py 在这里
    └── market_agent/   本文件在这里

安装(在 market_agent/ 文件夹里):
    uv init && rm main.py
    uv add langgraph langchain langchain-anthropic langchain-mcp-adapters

环境变量(见 .env.example):
    ANTHROPIC_API_KEY      给模型
    ALPHAVANTAGE_API_KEY   给 market_mcp(会传进它的子进程)
    MARKET_MCP_DIR         可选,server 所在目录;不设就按上面的兄弟结构自动找

运行:
    uv run python market_agent.py "帮我比一下 NEE、FSLR、ENPH"
"""

import asyncio
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()  # 让 .env 里的环境变量生效,不然 uv run 时不会自动读


# 用哪个模型,按你有权限的改。sonnet 便宜、够用。
MODEL = os.environ.get("AGENT_MODEL", "anthropic:claude-sonnet-4-6")

# server 位置:优先读环境变量,没设就按"兄弟文件夹"结构找 ../market_mcp。
# 不写死绝对路径,别人 clone 下来照结构放就能跑,换了结构设个环境变量即可。
MARKET_MCP_DIR = os.environ.get(
    "MARKET_MCP_DIR",
    str(Path(__file__).resolve().parent.parent / "market_mcp"),
)


def _require_env(name: str) -> str:
    """把"忘了设环境变量"变成一句人能读懂的提示,而不是一坨 KeyError。"""
    value = os.environ.get(name)
    if not value:
        sys.exit(f"缺少环境变量 {name}。可以写进 .env,或在命令前加 {name}=... 传入。")
    return value


async def build_agent():
    anthropic_key = _require_env("ANTHROPIC_API_KEY")
    av_key = _require_env("ALPHAVANTAGE_API_KEY")

    server_path = Path(MARKET_MCP_DIR)
    if not (server_path / "market_mcp.py").exists():
        sys.exit(
            f"在 {server_path} 里找不到 market_mcp.py。\n"
            "把 MARKET_MCP_DIR 设成 server 所在目录,或按兄弟文件夹结构摆放。"
        )

    # 1) 连 market_mcp(stdio 子进程),把它的 tool 自动转成 LangChain tool。
    #    这一步替你做了手写时最烦的"给每个 MCP tool 包一层可调函数"。
    client = MultiServerMCPClient(
        {
            "market": {
                "transport": "stdio",
                "command": "uv",
                # --directory 让 uv 去 server 自己的目录跑,不受 agent 环境影响。
                # --with 锁定子进程的 mcp 版本:agent 这边 adapter 依赖 mcp 1.x,
                # 而 market_mcp 需要 2.x,分开起就不打架。
                "args": [
                    "run",
                    "--directory", str(server_path),
                    "--with", "mcp[cli]==2.2.0",
                    "python", "market_mcp.py",
                ],
                # 子进程不一定继承你 shell 的环境变量,显式传 key 进去
                "env": {"ALPHAVANTAGE_API_KEY": av_key},
            }
        }
    )
    tools = await client.get_tools()

    # 2) 绑定模型 + tools。bind_tools 就是把 tool 的 schema 交给模型,让它会 tool-calling。
    model = init_chat_model(MODEL, api_key=anthropic_key).bind_tools(tools)

    # 3) agent 节点 = 手写循环里"发对话给模型、拿回一条回复"那一步。
    #    返回的 dict 会被 add_messages 自动追加进 state["messages"],不用你手动 append。
    async def call_model(state: MessagesState):
        response = await model.ainvoke(state["messages"])
        return {"messages": [response]}

    # 4) 条件边 = 手写的 "if 模型要调 tool: 去执行; else: 结束"。
    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        return "tools" if last.tool_calls else END

    # 5) 把节点和边拼成图。
    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))               # 自动执行模型点名的 tool
    builder.add_edge(START, "agent")                          # 入口先进 agent
    builder.add_conditional_edges("agent", should_continue)   # agent 之后:去 tools 还是 END
    builder.add_edge("tools", "agent")                        # 跑完回 agent —— 这条回边就是"循环"
    return builder.compile()


async def main(question: str):
    agent = await build_agent()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]}
    )

    # 想看清每一步调了什么,把下面这段注释打开(这是 LangGraph 相对手写好调试的地方之一)
    for msg in result["messages"]:
        msg.pretty_print()

    print(result["messages"][-1].content)


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "帮我比一下 NEE、FSLR、ENPH"
    asyncio.run(main(q))