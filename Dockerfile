
# Step 1: 使用官方的 Python 映像檔作為基礎

# 我們選擇 slim 版本以獲得較小的映像檔體積

FROM python:3.12-slim



# Step 2: 設定工作目錄

# 容器內的所有後續操作都會在這個目錄下進行

WORKDIR /app



# Step 3: 安裝系統級別的相依套件 (如果需要)

# 例如，某些函式庫可能需要 build-essential

# RUN apt-get update && apt-get install -y --no-install-recommends build-essential



# Step 4: 複製相依套件列表並安裝

# 這是最關鍵的一步。我們先只複製這個檔案，

# Docker 會快取這一步，如果 requirements.txt 沒有變更，後續建置會加速。

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt



# Step 5: 複製整個應用程式的程式碼到容器中

# 將您本地的 app, core, prompts 等所有資料夾複製進去

COPY . .



# Step 6: 開放應用程式運行的端口

# 讓容器外部可以訪問這個端口

EXPOSE 8080



# Step 7: 定義容器啟動時要執行的命令

# 使用 uvicorn 啟動 main.py 中的 FastAPI 應用

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]