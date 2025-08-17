# tests/test_main.py

import pytest
import json
import uuid
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from main import app 

@pytest.fixture
def client():
    return TestClient(app)

# 【修正 1】
# 這個輔助函式現在是一個 'async def'，它本身就是一個非同步產生器
async def mock_agent_stream_generator(events_to_yield):
    for event in events_to_yield:
        yield event

def test_chat_stream_with_new_thread(client, mocker):
    # 準備
    fake_agent_events = [
        {'event': 'on_tool_start', 'data': {'name': 'some_tool'}},
        {'event': 'on_tool_end', 'data': {'output': 'tool_output'}},
    ]
    
    mock_agent_executable = MagicMock()
    # 【修正 2】直接將 astream_events 的 return_value 設為我們的非同步產生器
    mock_agent_executable.astream_events.return_value = mock_agent_stream_generator(fake_agent_events)
    mocker.patch('main.agent_executable', new=mock_agent_executable)
    
    # 執行
    response = client.post("/api/v1/chat/stream", json={"message": "你好"})
    
    # 斷言
    assert response.status_code == 200
    sse_data_lines = [line[len('data: '):] for line in response.text.split('\n') if line.startswith('data:')]
    
    first_event = json.loads(sse_data_lines[0])
    assert first_event['type'] == 'thread_id'
    
    second_event = json.loads(sse_data_lines[1])
    assert second_event['type'] == 'on_tool_end'
    # 【修正 3】修正 KeyError，正確的結構是 second_event['data']['output']
    assert second_event['data']['output'] == 'tool_output'

def test_chat_stream_with_existing_thread(client, mocker):
    # 準備
    existing_thread_id = str(uuid.uuid4())
    mock_agent_executable = MagicMock()
    mock_agent_executable.astream_events.return_value = mock_agent_stream_generator([])
    mocker.patch('main.agent_executable', new=mock_agent_executable)
    
    # 執行
    response = client.post(
        "/api/v1/chat/stream",
        json={"message": "繼續", "thread_id": existing_thread_id}
    )
    
    # 斷言
    assert response.status_code == 200
    first_event = json.loads(response.text.split('\n')[0][len('data: '):])
    assert first_event['id'] == existing_thread_id
    
    config_arg = mock_agent_executable.astream_events.call_args.kwargs['config']
    assert config_arg['configurable']['thread_id'] == existing_thread_id

# 一個會拋出異常的假非同步產生器
async def mock_agent_stream_that_fails(error_message):
    # 為了讓它成為一個產生器，我們需要一個 yield 語句，即使它不會被執行
    if False:
        yield
    raise Exception(error_message)

def test_chat_stream_agent_error(client, mocker):
    # 準備
    error_msg = "Something went wrong"
    mock_agent_executable = MagicMock()
    mock_agent_executable.astream_events.return_value = mock_agent_stream_that_fails(error_msg)
    mocker.patch('main.agent_executable', new=mock_agent_executable)
    
    # 執行
    response = client.post("/api/v1/chat/stream", json={"message": "一個會壞掉的問題"})
    
    # 斷言
    assert response.status_code == 200
    sse_data_lines = [line[len('data: '):] for line in response.text.split('\n') if line.startswith('data:')]
    last_event = json.loads(sse_data_lines[-1])
    
    assert last_event['type'] == 'error'
    assert last_event['message'] == error_msg

def test_chat_stream_validation_missing_message(client):
    """
    整合測試：驗證當請求缺少必要的 'message' 欄位時，
    API (透過 Pydantic) 是否會回傳 422 錯誤。
    """
    # 準備一個不合法的請求 payload (缺少 message)
    invalid_payload = {
        "thread_id": "some-thread-id"
    }

    # 執行並斷言
    response = client.post("/api/v1/chat/stream", json=invalid_payload)
    
    # FastAPI 在 Pydantic 驗證失敗時，會自動回傳 422 Unprocessable Entity
    assert response.status_code == 422

def test_chat_stream_validation_wrong_message_type(client):
    """
    整合測試：驗證當 'message' 欄位的型別錯誤時，
    API (透過 Pydantic) 是否會回傳 422 錯誤。
    """
    # 準備一個不合法的請求 payload (message 是數字而非字串)
    invalid_payload = {
        "message": 12345,
        "thread_id": "some-thread-id"
    }

    # 執行並斷言
    response = client.post("/api/v1/chat/stream", json=invalid_payload)
    
    assert response.status_code == 422