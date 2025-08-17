# tests/app/test_ui.py

# 導入我們要測試的目標

import pytest
import json
from unittest.mock import MagicMock, AsyncMock
from app.ui import chat_client, format_log_message

def test_format_log_message_for_chat_model_end():
    """
    測試場景：格式化一個 LLM 的最終回應事件。
    """
    event_data = {
        "type": "on_chat_model_end",
        "data": {
            "output": {
                "content": "這是最終答案。"
            }
        }
    }
    result = format_log_message(event_data)
    assert "**EVENT:** `on_chat_model_end`" in result
    assert "** Agent回應:**" in result
    assert "這是最終答案。" in result

def test_format_log_message_for_tool_end():
    """
    測試場景：格式化一個工具執行的結束事件。
    """
    event_data = {
        "type": "on_tool_end",
        "data": {
            "name": "MyTool",
            "output": "工具的輸出結果"
        }
    }
    result = format_log_message(event_data)
    assert "**EVENT:** `on_tool_end`" in result
    assert "** 工具 `MyTool` 執行完畢, 輸出:**" in result
    assert "工具的輸出結果" in result

def test_format_log_message_for_unknown_event():
    """
    測試場景：格式化一個未知的事件類型。
    """
    event_data = {"type": "some_other_event", "data": {}}
    result = format_log_message(event_data)
    assert "**EVENT:** `some_other_event`" in result
    # 驗證它不會意外地包含其他事件的特定格式
    assert "Agent回應" not in result
    assert "工具" not in result


# 建立一個假的異步迭代器來模擬 response.aiter_lines()
class MockAsyncIterator:
    def __init__(self, lines):
        self._lines = iter(lines)
    def __aiter__(self):
        return self
    async def __anext__(self):
        try:
            return next(self._lines)
        except StopIteration:
            raise StopAsyncIteration

# --- 非同步測試 (套用最終修正) ---

@pytest.mark.asyncio
async def test_chat_client_success_stream(mocker):
    # 1. 準備最終的 response 物件
    sse_events = [
        f"data: {json.dumps({'type': 'thread_id', 'id': '123'})}",
        f"data: {json.dumps({'type': 'on_chat_model_end', 'data': {'output': {'content': '最終答案'}}})}",
    ]
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.aiter_lines = MagicMock(return_value=MockAsyncIterator(sse_events))

    # 2. 【關鍵修正】建立一個兩層的模擬結構
    #    層級 2: 模擬 client.stream() 回傳的物件 (stream_cm)
    mock_stream_cm = AsyncMock()
    mock_stream_cm.__aenter__.return_value = mock_response

    #    層級 1: 模擬 httpx.AsyncClient() 回傳的物件 (client_cm)
    #    它的 .__aenter__() 方法應該回傳一個帶有 .stream 方法的假 client
    mock_client_instance = AsyncMock()
    mock_client_instance.stream = MagicMock(return_value=mock_stream_cm)
    
    mock_client_cm = AsyncMock()
    mock_client_cm.__aenter__.return_value = mock_client_instance

    # 3. Patch httpx.AsyncClient，讓它在被呼叫時回傳我們最外層的管理器
    mocker.patch('app.ui.httpx.AsyncClient', return_value=mock_client_cm)

    # 執行與斷言
    final_state = None
    async for state in chat_client("你好", [], None):
        final_state = state

    history, _, thread_id = final_state # pyright: ignore[reportGeneralTypeIssues]
    assert history[0][1] == "最終答案"
    assert thread_id == "123"

@pytest.mark.asyncio
async def test_chat_client_api_error(mocker):
    # 準備
    mock_response = AsyncMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    
    mock_stream_cm = AsyncMock()
    mock_stream_cm.__aenter__.return_value = mock_response
    
    mock_client_instance = AsyncMock()
    mock_client_instance.stream = MagicMock(return_value=mock_stream_cm)
    
    mock_client_cm = AsyncMock()
    mock_client_cm.__aenter__.return_value = mock_client_instance

    mocker.patch('app.ui.httpx.AsyncClient', return_value=mock_client_cm)

    # 執行與斷言
    final_state = None
    async for state in chat_client("你好", [], None):
        final_state = state

    history, _, _ = final_state # pyright: ignore[reportGeneralTypeIssues]
    assert "錯誤: Internal Server Error" in history[0][1]

@pytest.mark.asyncio
async def test_chat_client_connection_error(mocker):
    # 準備
    mocker.patch('app.ui.httpx.ConnectError', ConnectionError) # 將 ConnectError 映射到內建的 ConnectionError

    mock_client_instance = AsyncMock()
    mock_client_instance.stream = MagicMock(side_effect=ConnectionError("Connection failed"))

    mock_client_cm = AsyncMock()
    mock_client_cm.__aenter__.return_value = mock_client_instance

    mocker.patch('app.ui.httpx.AsyncClient', return_value=mock_client_cm)
    
    # 執行與斷言
    final_state = None
    async for state in chat_client("你好", [], None):
        final_state = state

    history, _, _ = final_state # pyright: ignore[reportGeneralTypeIssues]
    assert "無法連接到後端 API" in history[0][1]