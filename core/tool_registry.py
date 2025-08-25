from core.tools.rag_tool import DeepResearchKnowledgeBase
from core.tools.web_search import search_tool
from core.tools.knowledge_writer import SaveNewKnowledgeTool
from core.tools.cognitive_processor import cognitive_processor_tool


ALL_TOOLS = [
    DeepResearchKnowledgeBase,
    search_tool,
    SaveNewKnowledgeTool,
    cognitive_processor_tool
]

print(f"✅ Tool registry initialized. Found {len(ALL_TOOLS)} tools.")