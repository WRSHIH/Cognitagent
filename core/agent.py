import operator
import logging
from typing import TypedDict, Annotated, List

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage

# 導入我們集中管理的 LLM 服務和工具列表
from core.services import get_langchain_gemini_pro
from core.tool_registry import ALL_TOOLS

# --- 1. 定義 Agent 的狀態 (State) ---
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    plan: str
    past_steps: Annotated[list, operator.add]
    response: str


# --- 2. 定義圖中的節點 (Nodes) ---

llm_with_tools = get_langchain_gemini_pro().bind_tools(ALL_TOOLS)

def agent_node(state: AgentState) -> dict:
    """
    Agent 節點：接收當前狀態，調用 LLM 進行思考，決定下一步行動。
    """

    print("--- Agent 思考下一步 ---")
    logging.info("---[Agent Node]: Thinking...")
    response_message = llm_with_tools.invoke(state['messages'])
    print(f"--- Agent 決策: {response_message.content} ---")
    if isinstance(response_message, AIMessage) and response_message.tool_calls:
        print(f"--- 準備執行工具: {response_message.tool_calls} ---")
        logging.info(f"--- [Agent Node]: Decision made. Tool calls: {bool(response_message.tool_calls)} ---")
    else:
        logging.info(f"--- [Agent Node]: Decision made. Response without tools ---")

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
        logging.info("---[Conditional Edge]: No tool call, ending graph.")
        return "end"
    else:
        logging.info("---[Conditional Edge]: Tool call detected, continuing to tools node.")
        return "continue"


# --- 3. 建立工廠函式 (Factory Function) ---
def create_agent_graph():
    """
    這個工廠函式封裝了所有組裝 Agent 的步驟，
    並返回一個已編譯好、可直接執行的 LangGraph 物件。
    """
    logging.info("Initializing Agent Graph...")

    # 建立一個新的 StateGraph，並指定其狀態結構
    workflow = StateGraph(AgentState)

    # 新增節點到圖中
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # 設定圖的進入點
    workflow.set_entry_point("agent")

    # 新增條件判斷邊
    workflow.add_conditional_edges(
        source="agent",
        path=should_continue,
        path_map={
            "continue": "tools",
            "end": END,
        },
    )

    # 新增一般邊
    workflow.add_edge("tools", "agent")

    # 設定對話記憶體
    # InMemorySaver 會將每個 thread 的對話狀態保存在記憶體中
    memory = InMemorySaver()

    # 編譯圖，使其成為可執行的物件
    app = workflow.compile(checkpointer=memory)

    logging.info("✅ Agent Graph compiled successfully.")

    return app