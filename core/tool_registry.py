from core.tools.rag_tool import DeepResearchKnowledgeBase
from core.tools.web_search import search_tool
from core.tools.knowledge_writer import SaveNewKnowledgeTool


ALL_TOOLS = [
    DeepResearchKnowledgeBase,
    search_tool,
    SaveNewKnowledgeTool,
]

print(f"✅ Tool registry initialized. Found {len(ALL_TOOLS)} tools.")