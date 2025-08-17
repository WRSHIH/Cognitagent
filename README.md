# GraphRAG-Based-Agent：一個具備自主知識進化能力的智慧型代理

本專案旨在建構一個基於 GraphRAG 架構的進階智慧型代理，此代理具備自主學習之能力。其核心功能不僅體現在透過多樣化工具與外部世界進行互動，更在於能將擷取之新型、有價值資訊整合回核心知識庫，從而實現知識體系的持續性成長與進化。

## 📋 主要特色 (Key Features)
### 🤖 代理核心 (Agentic Core)：基於 LangGraph 框架進行建構，賦予代理多步驟推理、自主決策及工具運用之能力，使其得以處理高度複雜的使用者請求。

### 🧠 混合式 RAG (Hybrid RAG)：整合 LlamaIndex 與 Qdrant 向量資料庫，同步利用向量語意檢索與傳統關鍵字稀疏檢索 (BM25)，以提供更為精準且全面的知識庫查詢效能。

## 🛠️ 多功能工具集 (Versatile Tools)：

### 深度研究工具：賦予代理對內部知識庫進行多角度、探索式查詢的能力。

### 即時網路搜尋：整合 Tavily Search API，確保代理能夠獲取最新的網路即時資訊。

### 知識進化工具：本專案獨特的「知識寫入」機制，能將對話過程中所生成或自網路搜尋獲得的新資訊，經過「原子化」拆解與「智能化」合併後，回寫至 Qdrant 核心知識庫。

## 🌐 前後端分離架構 (Decoupled Architecture)：

### 後端 API: 採用 FastAPI 框架，提供一個高效能、非同步的 API 服務層，並支援 Server-Sent Events (SSE) 以實現串流式回應。

### 前端 UI: 運用 Gradio 函式庫，打造一個獨立的互動式使用者介面，以利於快速的功能驗證與效果展示。

## 📦 容器化部署 (Containerized)：提供完整的 Dockerfile 設定，允許將應用程式及其所有依賴項一鍵打包，便於在任何支援 Docker 的標準化環境中進行快速部署。

## ✅ 高品質測試 (Robustly Tested)：專案之核心模組均具備完整的單元測試與整合測試 (Pytest)，旨在確保程式碼的穩定性、可靠性與長期可維護性。

## 🛠️ 技術棧 (Technology Stack)
### Web 框架: FastAPI, Uvicorn, SSE-Starlette

### Agent / RAG 框架: LangChain, LangGraph, LlamaIndex

### 模型與嵌入: Google Gemini Pro, Gemini Flash, Google GenAI Embeddings

### 向量資料庫: Qdrant

### 網路搜尋: Tavily Search

### 資料處理: Pydantic (資料驗證), Unstructured (文件解析)

### UI 介面: Gradio

### 測試框架: Pytest, Pytest-Mock, Pytest-Asyncio, Pytest-Cov

## 🚀 快速入門 (Quick Start)
請遵循以下程序，於您的本地端環境中設定並啟動本專案。

### 1. 環境設定 (Environment Setup)
#### 1. 克隆專案原始碼儲存庫
git clone https://your-repository-url/GraphRAG-Based-Agent.git
cd GraphRAG-Based-Agent

#### 2. 建立並啟動 Python 虛擬環境
python -m venv .venv
source .venv/bin/activate

#### 3. 安裝所有必要的依賴套件
pip install -r requirements.txt


### 2. 設定環境變數 (Configure Environment Variables)
本專案之 API 金鑰與相關設定係透過 .env 檔案進行集中管理。

請將 .env.example 檔案複製一份並重新命名為 .env：

cp .env.example .env


請編輯 .env 檔案，並填入您個人的 API 金鑰與相關設定：

#### .env

#### --- 必要金鑰 ---
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
QDRANT_API_KEY="YOUR_QDRANT_API_KEY_OR_LEAVE_EMPTY"
TAVILY_API_KEY="YOUR_TAVILY_API_KEY"

#### --- 服務設定 ---
QDRANT_URL="http://localhost:6333"

#### --- 模型設定 ---
GEMINI_FLASH="models/gemini-1.5-flash-latest"
GEMINI_PRO="models/gemini-1.5-pro-latest"
GEMINI_EMBED="models/text-embedding-004"


### 3. 啟動向量資料庫 (Start Vector Database)
本專案採用 Qdrant 作為其向量資料庫。建議透過 Docker 進行快速啟動。

docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant


### 4. 導入初始資料 (Ingest Initial Data)
在啟動 Agent 服務之前，必須先將初始知識文件導入至 Qdrant 資料庫中。

請將您的 .pdf, .md, .txt 等格式之文件放置於 data/source_docs/ 目錄下（若該目錄不存在，請自行建立）。

執行以下導入腳本：

python -m script.ingest --source-dir ./data/source_docs --collection my_KnowledgeBase --recreate


--collection: 用以指定 Qdrant 中的集合 (collection) 名稱。

--recreate: 此參數將會刪除並重建一個同名的集合，以確保資料的初始狀態為乾淨。

🏃 如何使用 (How to Use)
1. 啟動後端 API
請於專案根目錄下執行以下指令以啟動後端服務：

uvicorn main:app --host 0.0.0.0 --port 8000 --reload


--reload 參數將啟用熱重載功能，當您修改程式碼後，伺服器將會自動重啟。

服務啟動後，即可瀏覽 http://localhost:8000/docs 以查閱並進行 API 的互動式測試。

2. 啟動前端介面
請開啟一個新的終端機視窗，並執行以下指令以啟動 Gradio 使用者介面：

gradio app/ui.py


隨後，請使用瀏覽器訪問 http://127.0.0.1:7860 (或終端機所提示之位址)，即可開始與本智慧型代理進行互動。

✅ 執行測試 (Running Tests)
本專案配備了完整的自動化測試套件，用以確保程式碼的品質與穩定性。

# 執行所有已定義的測試案例
python -m pytest

# 執行測試並同步計算程式碼覆蓋率
python -m pytest --cov


🐳 使用 Docker 運行 (Running with Docker)
專案已內建 Dockerfile 以利於容器化部署。

# 1. 建置 Docker 映像檔
docker build -t graphrag-agent .

# 2. 運行 Docker 容器
#    請確保您的 .env 檔案已根據需求填寫完畢
docker run -p 8080:8080 --env-file .env graphrag-agent


容器成功運行後，API 服務將可於 http://localhost:8080 進行存取。

## 📂 專案結構 (Project Structure)
GraphRAG-Based-Agent/
├── app/                  # FastAPI 應用與 Gradio UI 相關程式碼
│   ├── models/           # Pydantic 資料模型 (schemas)
│   ├── ui.py             # Gradio 前端介面
│   └── ...
├── core/                 # 專案核心邏輯
│   ├── tools/            # Agent 使用的工具 (RAG, Web Search, Knowledge Writer)
│   ├── agent.py          # LangGraph Agent 的定義與流程
│   ├── config.py         # Pydantic 設定管理
│   ├── services.py       # 外部服務初始化 (LLM, Embeddings, DB Client)
│   └── tool_registry.py  # 工具註冊中心
├── data/                 # 存放資料
│   └── source_docs/      # 初始知識文件存放處
├── promps/               # Prompt 模板
├── script/               # 輔助腳本
│   └── ingest.py         # 資料導入腳本
├── tests/                # Pytest 測試檔案
│   ├── app/
│   ├── core/
│   └── script/
├── .env.example          # 環境變數範本
├── Dockerfile            # Docker 容器設定檔
├── main.py               # FastAPI 應用程式入口
└── requirements.txt      # Python 依賴套件列表




