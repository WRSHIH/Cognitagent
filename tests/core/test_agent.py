# tests/core/test_agent.py

import pytest
from unittest.mock import MagicMock, patch

# 導入我們要測試的目標
from core.agent import AgentState, agent_node, should_continue

# 導入 LangChain 的訊息類別，以便我們偽造 LLM 的回應
from langchain_core.messages import AIMessage, HumanMessage, ToolCall


def test_should_continue_with_tool_calls():
    """
    測試場景：當最後一則訊息包含 tool_calls 時，應回傳 'continue'。
    """
    # 準備一個包含 tool_calls 的 AIMessage
    ai_message_with_tools = AIMessage(
        content="好的，我來幫您查詢。",
        tool_calls=[ToolCall(name="some_tool", args={"query": "test"}, id="1")]
    )
    state = AgentState(messages=[HumanMessage(content="你好"), ai_message_with_tools]) # pyright: ignore[reportCallIssue]
    
    assert should_continue(state) == "continue"

def test_should_continue_without_tool_calls():
    """
    測試場景：當最後一則訊息是 AIMessage 但不含 tool_calls 時，應回傳 'end'。
    """
    ai_message_without_tools = AIMessage(content="我無法幫您查詢。")
    state = AgentState(messages=[HumanMessage(content="你好"), ai_message_without_tools]) # pyright: ignore[reportCallIssue]
    
    assert should_continue(state) == "end"

def test_should_continue_with_human_message_last():
    """
    測試場景：當最後一則訊息是 HumanMessage 時，應回傳 'end'。
    """
    state = AgentState(messages=[
        AIMessage(content="你好嗎？"),
        HumanMessage(content="我很好")
    ]) # pyright: ignore[reportCallIssue]
    
    assert should_continue(state) == "end"

def test_should_continue_with_empty_messages():
    """
    測試邊界案例：當訊息列表為空時，應回傳 'end'。
    """
    # 雖然 LangGraph 的 StateGraph 不允許 messages 為空，但測試其健壯性是好的實踐
    state = AgentState(messages=[]) # pyright: ignore[reportCallIssue]
    assert should_continue(state) == "end"


# tests/core/test_agent.py

# ... (前述 import 和測試) ...

# 使用 patch 來模擬 llm_with_tools 這個在 agent.py 模組全域範圍的物件
@patch('core.agent.llm_with_tools')
def test_agent_node_responds_without_tools(mock_llm_with_tools):
    """
    測試場景：模擬 LLM 直接回傳答案，不呼叫工具。
    """
    # 1. 準備 (Arrange)
    # 準備一個假的 Agent 狀態
    input_state = AgentState(messages=[HumanMessage(content="嗨！")]) # pyright: ignore[reportCallIssue]
    
    # 準備一個假的 LLM 回應 (不含 tool_calls)
    fake_response = AIMessage(content="你好，有什麼可以幫您的嗎？")
    
    # 設定當 mock_llm_with_tools.invoke 被呼叫時，就回傳我們準備好的假回應
    mock_llm_with_tools.invoke.return_value = fake_response
    
    # 2. 執行 (Act)
    result = agent_node(input_state)
    
    # 3. 斷言 (Assert)
    # 驗證 llm_with_tools.invoke 是否被以正確的參數呼叫了一次
    mock_llm_with_tools.invoke.assert_called_once_with(input_state['messages'])
    
    # 驗證回傳的結果是否符合預期
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert result["messages"][0] == fake_response
    assert not result["messages"][0].tool_calls # 確保沒有 tool_calls

@patch('core.agent.llm_with_tools')
def test_agent_node_decides_to_use_tools(mock_llm_with_tools):
    """
    測試場景：模擬 LLM 決定呼叫工具。
    """
    # 1. 準備 (Arrange)
    input_state = AgentState(messages=[HumanMessage(content="請幫我查天氣")]) # pyright: ignore[reportCallIssue]
    
    # 準備一個假的 LLM 回應 (包含 tool_calls)
    fake_tool_call = ToolCall(name="search_weather", args={"city": "台北"}, id="tool_123")
    fake_response_with_tools = AIMessage(
        content="", 
        tool_calls=[fake_tool_call]
    )
    mock_llm_with_tools.invoke.return_value = fake_response_with_tools
    
    # 2. 執行 (Act)
    result = agent_node(input_state)
    
    # 3. 斷言 (Assert)
    mock_llm_with_tools.invoke.assert_called_once_with(input_state['messages'])
    
    # 驗證回傳的訊息是否包含我們偽造的 tool_calls
    
    returned_message = result["messages"][0]
    assert len(returned_message.tool_calls) == 1
    actual_tool_call = returned_message.tool_calls[0]
    
    assert actual_tool_call['name'] == fake_tool_call['name']
    assert actual_tool_call['args'] == fake_tool_call['args']
    assert actual_tool_call['id'] == fake_tool_call['id']