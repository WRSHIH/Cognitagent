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


systemprompt = load_prompt('agent_sys_prompt.txt')
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", systemprompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# --- 1. 定義 Agent 的狀態 (State) ---
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    # plan: str
    # past_steps: Annotated[list, operator.add]
    # response: str


# --- 2. 定義圖中的節點 (Nodes) ---

llm_with_tools = get_langchain_gemini_pro().bind_tools(ALL_TOOLS)
agent_runnable = prompt | llm_with_tools

def agent_node(state: AgentState) -> dict:
    """
    Agent 節點：接收當前狀態，調用 LLM 進行思考，決定下一步行動。
    """

    print("--- Agent 思考下一步 ---")
    logging.info("---[Agent Node]: Thinking...")
    response_message = agent_runnable.invoke(state)
    print(f"--- Agent 決策: {response_message.content} ---")
    if isinstance(response_message, AIMessage) and response_message.tool_calls:
        print(f"--- 準備執行工具: {response_message.tool_calls} ---")
        logging.info(f"--- [Agent Node]: Decision made. Tool calls: {bool(response_message.tool_calls)} ---")
    else:
        logging.info(f"--- [Agent Node]: Decision made. Response without tools ---")

    return {"messages": [response_message]}

# 建立一個 ToolNode，它會自動根據 LLM 的 tool_calls 指令去執行對應的工具
tool_node = ToolNode(ALL_TOOLS)

def generate_response_node(state: AgentState) -> dict:
    """
    專門的回應生成節點。在工具執行完畢後被呼叫。
    它的唯一任務是根據包含工具結果的完整對話歷史，生成最終答案。
    """
    print("--- 根據工具結果生成最終回應 ---")
    logging.info("--- [Responder Node]: Generating final response...")

    # 我們不再綁定工具，因為這一步的目標只是生成文字
    llm_responder = get_langchain_gemini_pro() 
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個善於總結的助理，請根據使用者問題和工具提供的資訊，生成一個最終的、完整的回答。"),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    responder_runnable = final_prompt | llm_responder
    response_message = responder_runnable.invoke(state)
    
    return {"messages": [response_message]}

# 建立條件判斷邊的邏輯
def should_continue(state: AgentState) -> str:
    """
    條件判斷邊 (Conditional Edge)：決定流程應該結束還是繼續呼叫工具。
    """
    if not state['messages']:
        return "end"

    last_message = state['messages'][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logging.info("---[Conditional Edge]: Tool call detected, continuing to tools node.")
        return "continue"
    else:
        # 如果不是 AIMessage，或者 AIMessage 裡沒有 tool_calls，就結束
        logging.info("---[Conditional Edge]: No tool call, ending graph.")
        return "end"


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
    workflow.add_node("responder", generate_response_node)

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
    workflow.add_edge("tools", "responder")
    workflow.add_edge("responder", END)

    # 設定對話記憶體
    # InMemorySaver 會將每個 thread 的對話狀態保存在記憶體中
    memory = InMemorySaver()

    # 編譯圖，使其成為可執行的物件
    app = workflow.compile(checkpointer=memory)

    logging.info("✅ Agent Graph compiled successfully.")

    return app