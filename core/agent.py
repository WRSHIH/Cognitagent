import operator
import logging
from typing import TypedDict, Annotated, List

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 導入我們集中管理的 LLM 服務和工具列表
from core.services import get_langchain_gemini_pro
from core.tool_registry import ALL_TOOLS
from core.utils import load_prompt


# --- 1. 定義 Agent 的狀態 (State) ---
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    plan: str
    past_steps: Annotated[list, operator.add]
    response: str


# --- 2. 定義圖中的節點 (Nodes) ---

llm_with_tools = get_langchain_gemini_pro().bind_tools(ALL_TOOLS)
# agent_runnable = prompt | llm_with_tools

def agent_node(state: AgentState) -> dict:

    print("--- Agent 思考下一步 ---")
    current_messages = state['messages']
    response_message = llm_with_tools.invoke(current_messages)
    print(f"--- Agent 決策: {response_message.content} ---")

    if response_message.tool_calls:
        print(f"--- 準備執行工具: {response_message.tool_calls}")

    return {"messages": [response_message]}

# 建立一個 ToolNode，它會自動根據 LLM 的 tool_calls 指令去執行對應的工具
tool_node = ToolNode(ALL_TOOLS)

# 建立條件判斷邊的邏輯
def should_continue(state: AgentState) -> str:
    """
    條件判斷邊 (Conditional Edge)：決定流程應該結束還是繼續呼叫工具。
    """
    last_message = state['messages'][-1]
    if not last_message.tool_calls:
        print("---🔚 判斷：結束流程 ---")
        return "end"
    else:
        print("---➡️ 判斷：繼續執行工具 ---")
        return "continue"


# --- 3. 建立工廠函式 (Factory Function) ---
def create_agent_graph():
    logging.info("Initializing Agent Graph...")

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
                                    source="agent",
                                    path=should_continue,
                                    path_map={
                                        "continue": "tools",
                                        "end": END,
                                    },
                                )
    workflow.add_edge("tools", "agent")

    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)

    logging.info("✅ Agent Graph compiled successfully.")

    return app