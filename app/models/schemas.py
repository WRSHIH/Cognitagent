from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    """
    聊天請求的資料模型
    """
    message: str = Field(..., description="使用者的輸入訊息")
    thread_id: Optional[str] = Field(None, description="對話的線程 ID，若無則會自動產生")