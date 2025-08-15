import pytest
from unittest.mock import MagicMock, patch

# 引入我們要測試的函式
from core.tools.knowledge_writer import (
    atomize_knowledge,
    merge_knowledge_with_llm,
    save_new_knowledge,
)

# 為了模擬 LlamaIndex 的物件，我們可以建立一些假的類別
class MockNode:
    def __init__(self, text, node_id, score, metadata=None):
        self._text = text
        self.node_id = node_id
        self._score = score
        self.metadata = metadata or {}

    def get_text(self):
        return self._text

    def get_score(self):
        return self._score


# 測試 atomize_knowledge
@patch('core.tools.knowledge_writer.get_llama_gemini_flash')
def test_atomize_knowledge_success(mock_llm_flash):
    """
    場景：LLM 成功回傳一個格式正確的 JSON 字串。
    驗證：函式能否正確移除程式碼區塊標記，並解析 JSON。
    """
    # 1. 準備一個假的 LLM 回應
    mock_response = MagicMock()
    mock_response.text = """
    ```json
    [
        {"text": "原子知識點一", "type": "Fact"},
        {"text": "原子知識點二", "type": "Definition"}
    ]
    ```
    """
    # 2. 準備一個假的 LLM 物件，並設定它的 .complete() 方法
    mock_llm_instance = MagicMock()
    mock_llm_instance.complete.return_value = mock_response
    
    # 3. 【關鍵】設定當 get_llm_flash() 函式被呼叫時，回傳我們準備好的假 LLM 物件
    mock_llm_flash.return_value = mock_llm_instance  

    # 4. 斷言 (Assert)
    result = atomize_knowledge("這是一段示例文本。")
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]['text'] == "原子知識點一"

    # 驗證 mock_llm_flash 函式被呼叫了一次
    mock_llm_flash.assert_called_once()
    # 驗證 llm_flash.complete 是否真的被以包含輸入文本的 prompt 呼叫了一次
    mock_llm_instance.complete.assert_called_once()
    prompt_arg = mock_llm_instance.complete.call_args[0][0]
    assert "這是一段示例文本。" in prompt_arg

@patch('core.tools.knowledge_writer.get_llama_gemini_flash')
def test_atomize_knowledge_json_error(mock_llm_flash):
    """
    場景：LLM 回傳了格式錯誤的文本，無法被解析為 JSON。
    驗證：函式應能捕捉異常，並回傳一個包含原始文本的預設結構。
    """
    # 1. 準備 (Arrange)
    mock_response = MagicMock()
    mock_response.text = "這不是一個有效的JSON。"
    mock_llm_flash.complete.return_value = mock_response
    
    input_text = "這是一段輸入文本。"

    # 2. 執行 (Act)
    result = atomize_knowledge(input_text)

    # 3. 斷言 (Assert)
    assert result == [{"text": input_text, "type": "Unprocessed", "keywords": [], "question": ""}]


# 測試 merge_knowledge_with_llm
@patch('core.tools.knowledge_writer.get_llama_gemini_flash')
def test_merge_knowledge_with_llm(mock_llm_flash):
    """
    場景：成功呼叫 LLM 進行知識合併。
    驗證：函式是否回傳了 LLM 處理後的文本。
    """
    # 1. 準備 (Arrange)
    mock_response = MagicMock()
    mock_response.text = "這是合併後的最終版本。"
    mock_llm_instance = MagicMock()
    mock_llm_instance.complete.return_value = mock_response
    mock_llm_flash.return_value = mock_llm_instance
    
    old_text = "舊版本"
    new_text = "新版本"

    # 2. 執行 (Act)
    result = merge_knowledge_with_llm("舊版本", new_text)

    # 3. 斷言 (Assert)
    assert result == "這是合併後的最終版本。"
    # 驗證 prompt 中是否正確地包含了新舊文本
    prompt_arg = mock_llm_instance.complete.call_args[0][0]
    assert f'{old_text}' in prompt_arg
    assert f'{new_text}' in prompt_arg



# 測試 save_new_knowledge

# 我們將使用 mocker.patch.object 來模擬這個複雜函式所依賴的其他函式
@patch('core.tools.knowledge_writer.atomize_knowledge')
@patch('core.tools.knowledge_writer.get_index')
def test_save_new_knowledge_add_new(mock_index, mock_atomize):
    """
    場景：傳入一段全新的知識，知識庫中不存在相似內容。
    驗證：系統應將其視為新知識並執行插入操作。
    """
    # 1. 準備 (Arrange)
    # 模擬 `atomize_knowledge` 的回傳值
    mock_atomize.return_value = [{"text": "這是一筆全新的知識", "type": "Fact", "keywords": ["全新"], "question": "這知識是？"}]
    
    # 模擬 retriever：當被 retrieve 呼叫時，回傳空列表，表示沒找到相似節點
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_index_instance = MagicMock()
    mock_index_instance.as_retriever.return_value = mock_retriever
    
    # 2. 【關鍵】設定 get_index() 函式的回傳值
    mock_index.return_value = mock_index_instance

    # 執行與斷言
    result_message = save_new_knowledge("一段輸入文本")
    mock_index_instance.insert_nodes.assert_called_once()
    assert "新增知識: 1 條" in result_message



@patch('core.tools.knowledge_writer.atomize_knowledge')
@patch('core.tools.knowledge_writer.merge_knowledge_with_llm')
@patch('core.tools.knowledge_writer.get_index')
def test_save_new_knowledge_update_existing(mock_index, mock_merge, mock_atomize):
    """
    場景：傳入的知識與現有知識高度相似，且 LLM 成功將它們合併成新版本。
    驗證：系統應執行刪除舊節點和插入新節點的操作。
    """
    mock_atomize.return_value = [{"text": "新知識"}]
    mock_merge.return_value = "合併後的知識"

    # 準備假的 node, retriever, index 物件
    old_node = MockNode(text="舊知識", node_id="abc-123", score=0.98)
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [old_node] # 找到相似節點
    mock_index_instance = MagicMock()
    mock_index_instance.as_retriever.return_value = mock_retriever
    mock_index.return_value = mock_index_instance

    result_message = save_new_knowledge("輸入")
    mock_index_instance.delete_nodes.assert_called_once_with(["abc-123"])
    assert "更新知識: 1 條" in result_message

@patch('core.tools.knowledge_writer.atomize_knowledge')
@patch('core.tools.knowledge_writer.merge_knowledge_with_llm')
@patch('core.tools.knowledge_writer.get_index')
def test_save_new_knowledge_skip_duplicate(mock_index, mock_merge, mock_atomize):
    """
    場景：傳入的知識與現有知識高度相似，但 LLM 認為無需合併（回傳了與舊版一樣的內容）。
    驗證：系統應跳過操作，不進行任何資料庫寫入。
    """
    mock_atomize.return_value = [{"text": "新知識"}]
    old_text = "舊知識"
    mock_merge.return_value = old_text # 合併後無變化

    old_node = MockNode(text=old_text, node_id="xyz-789", score=0.99)
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [old_node]
    mock_index_instance = MagicMock()
    mock_index_instance.as_retriever.return_value = mock_retriever
    mock_index.return_value = mock_index_instance

    result_message = save_new_knowledge("輸入")
    mock_index_instance.delete_nodes.assert_not_called()
    assert "忽略知識: 1 條" in result_message