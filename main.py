import uuid
import json
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import ChatPromptValue 

# 導入我們之前建立的 Agent 工廠函式和資料模型
from core.agent import create_agent_graph
from app.models.schemas import ChatRequest

# --- 日誌與應用程式初始化 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Advanced Agent API",
    version="1.0.0",
)

# 允許跨來源請求 (CORS)，這對於前後端分離的開發至關重要
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生產環境中應指定前端的具體來源
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 在應用程式啟動時，呼叫工廠函式一次，建立 Agent 執行緒
agent_executable = create_agent_graph()


# --- 新增一個輔助函式來轉換資料 ---
def convert_docs_to_dict(docs):
    """
    一個更通用的輔助函式，將任何繼承自 BaseMessage 的 LangChain 物件轉換為可序列化的字典。
    """
    # 只要是 LangChain 的 Message 物件
    if isinstance(docs, BaseMessage): 
        return {"type": docs.type, "content": docs.content}
    if isinstance(docs, ChatPromptValue):
        return convert_docs_to_dict(docs.to_messages())
    # 如果是列表，遞迴處理裡面的每個元素
    if isinstance(docs, list):
        return [convert_docs_to_dict(doc) for doc in docs]
    # 如果是字典，遞迴處理裡面的每個值
    if isinstance(docs, dict):
        return {key: convert_docs_to_dict(value) for key, value in docs.items()}
    # 如果是其他基本類型，直接返回
    return docs


# --- API 端點 (Endpoint) 定義 ---
@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    當前端發送一個聊天請求時，這個端點會啟動 Agent 的執行過程，
    並將 Agent 的回應以串流的方式發送回前端。
    如果請求中沒有提供 thread_id，則會自動生成一個新的 UUID 作為 thread_id。
    串流的過程中，會將 Agent 的事件格式化為 SSE 事件，
    並在串流結束時發送結束事件。
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [("user", request.message)]}

    async def event_generator():
        """
        一個異步產生器，用於從 Agent 的 stream 中讀取事件並格式化為 SSE 事件。
        這個產生器會在串流開始時發送 thread_id，然後持續接收 Agent 的事件，
        並將這些事件格式化為 JSON 格式的 SSE 事件發送給前端。
        當 Agent 的執行結束時，會發送一個結束事件。
        """
        try:
            # 首次發送 thread_id 給前端
            yield json.dumps({"type": "thread_id", "id": thread_id})

            # 非同步地迭代 Agent 的執行事件
            async for event in agent_executable.astream_events(inputs, config=config, version="v1"): # pyright: ignore[reportArgumentType]
                event_type = event['event']
                serializable_data = convert_docs_to_dict(event['data'])
                payload = {"type": event_type, "data": serializable_data}

                # 我們只串流結束事件，以簡化前端處理
                if event_type.endswith('_end'):
                    yield json.dumps(payload)

        except Exception as e:
            logging.error(f"串流時發生錯誤: {e}", exc_info=True)
            yield json.dumps({"type": "error", "message": str(e)})

    return EventSourceResponse(event_generator(), media_type="text/event-stream")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Advanced Agent API. Please use the /docs endpoint for documentation."}