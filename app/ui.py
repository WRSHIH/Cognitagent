import gradio as gr
import httpx
import json
import uuid

# 後端 API 的位址
API_URL = "http://127.0.0.1:8080/api/v1/chat/stream"

def format_log_message(event_data: dict) -> str:
    """將從 API 收到的事件資料格式化為人類可讀的日誌訊息。"""
    event_type = event_data.get("type", "unknown_event")
    data = event_data.get("data", {})
    log_msg = f"**EVENT:** `{event_type}`\n"

    if event_type == "on_chat_model_end":
        content = data.get('output', {}).get('content', '')
        if content:
            log_msg += f"** Agent回應:**\n---\n{content}\n---\n"

    elif event_type == "on_tool_end":
        tool_name = data.get('name', 'unknown_tool')
        output = data.get('output', 'No output')
        log_msg += f"** 工具 `{tool_name}` 執行完畢, 輸出:**\n```json\n{output}\n```\n"

    return log_msg + "\n"


async def chat_client(message: str, history: list, thread_id: str | None):
    """
    Gradio 的核心函式，現在它扮演一個 API 客戶端的角色。
    """
    payload = {"message": message, "thread_id": thread_id}
    history = history or []

    agent_log = "** Agent 工作日誌**\n\n"
    full_response = ""
    history.append([message, ""])

    try:
        # 使用 httpx 建立一個異步的串流請求
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", API_URL, json=payload) as response:
                if response.status_code != 200:
                    history[-1][1] = f"錯誤: {response.text}"
                    yield history, agent_log, thread_id
                    return

                # 處理從後端 API 發送過來的 SSE 事件
                async for line in response.aiter_lines():
                    if line.startswith('data:'):
                        try:
                            data_str = line[len('data:'):].strip()
                            event_data = json.loads(data_str)

                            if event_data.get("type") == "thread_id" and not thread_id:
                                thread_id = event_data.get("id")

                            # 更新 Agent 工作日誌
                            agent_log += format_log_message(event_data)

                            # 如果是 LLM 的回應，更新聊天視窗的內容
                            if event_data.get("type") == "on_chat_model_end":
                                llm_output = event_data.get("data", {}).get("output", {})
                                if isinstance(llm_output, dict):
                                     full_response = llm_output.get("content", "")
                                     history[-1][1] = full_response

                            yield history, agent_log, thread_id

                        except json.JSONDecodeError:
                            continue # 忽略無法解析的行
    except httpx.ConnectError as e:
        history[-1][1] = f"無法連接到後端 API: {API_URL}。請確認後端服務是否已啟動。\n{e}"
        yield history, agent_log, thread_id


# --- 建立並啟動 Gradio 介面 ---
with gr.Blocks(theme=gr.themes.Soft(), title="Agent UI") as demo:
    
    # 使用 gr.State 來跨回合儲存 thread_id
    session_thread_id = gr.State(None)

    gr.Markdown("#  Agentic General Assistant (Decoupled UI)")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(elem_id="chatbot",
                                 bubble_full_width=False,
                                 label="主對話視窗", 
                                 height=600)
        with gr.Column(scale=1):
            agent_log = gr.Markdown(value="** Agent 工作日誌**", label="Agent's Working Log")

    with gr.Row():
        txt_input = gr.Textbox(show_label=False, 
                               placeholder="請輸入您的問題...",
                               container=False,
                               scale=4)
        btn_submit = gr.Button(" 送出", variant="primary", scale=1, min_width=0)

    

    txt_input.submit(
        fn=chat_client,
        inputs=[txt_input, chatbot, session_thread_id],
        outputs=[chatbot, agent_log, session_thread_id]
    ).then(lambda: "", [], [txt_input])

    btn_submit.click(
        fn=chat_client,
        inputs=[txt_input, chatbot, session_thread_id],
        outputs=[chatbot, agent_log, session_thread_id]
    ).then(lambda: "", [], [txt_input])


if __name__ == "__main__":
    demo.launch(share=True)