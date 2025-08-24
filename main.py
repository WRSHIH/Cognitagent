import uuid
import json
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import ChatPromptValue 

# 導入我們之前建立的 Agent 工廠函式和資料模型
from core.agent import create_master_graph
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
agent_executable = create_master_graph()

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
    inputs = {"main_goal": request.message}

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

            async for event in agent_executable.astream_events(inputs, config=config, version="v1"): # pyright: ignore[reportArgumentType]
                event_type = event['event']
                serializable_data = jsonable_encoder(event['data'])
                

                if event_type.endswith('_end'):
                    payload = {"type": event_type, "data": serializable_data}
                    yield json.dumps(payload)

                if event_type == 'on_chain_end':
                    output_data = serializable_data.get('output', {})
                    if output_data and ('simple_executor' in output_data or 'synthesizer' in output_data):
                        final_node_output = output_data.get('simple_executor') or output_data.get('synthesizer', {})
                        if final_node_output and 'response' in final_node_output:
                            final_answer_payload = {"type": "final_answer",
                                                    "data": {"content": final_node_output['response']}}
                            yield json.dumps(final_answer_payload)
        except Exception as e:
            logging.error(f"串流時發生錯誤: {e}", exc_info=True)
            yield json.dumps({"type": "error", "message": str(e)})

    return EventSourceResponse(event_generator(), media_type="text/event-stream")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Advanced Agent API. Please use the /docs endpoint for documentation."}

