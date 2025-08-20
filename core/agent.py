import operator
import logging
import json
from typing import TypedDict, Annotated, List, Union, Any

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from pydantic import BaseModel, Field

# from langgraph.prebuilt import ToolNode
# from langgraph.checkpoint.memory import InMemorySaver
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 導入我們集中管理的 LLM 服務和工具列表
from core.services import get_langchain_gemini_pro
from core.tool_registry import ALL_TOOLS
from core.tools.web_search import search_tool
from core.tools.rag_tool import DeepResearchKnowledgeBase

# 狀態和結構的定義
class SubGoal(BaseModel):
    goal_id: int
    description: str
    dependencies: List[int] = Field(default_factory=list)
    status: str = "pending"
    raw_result: Union[dict, str, None] = None
    result_summary: str = ""

class HierarchicalPlan(BaseModel):
    main_goal: str
    sub_goals: List[SubGoal]

class Verdict(BaseModel):
    summary: str = Field(description="A brief summary of the sub-task's result.")
    new_status: str = Field(description="The new status for the sub-goal, e.g., 'completed' or 'failed'.")
    next_action: str = Field(description="The next action to take: 'CONTINUE', 'REPLAN', or 'ABORT'.")

class AgentState(TypedDict):
    """
    定義圖的狀態，追蹤整個任務流程。
    所有鍵都應有預設值，以避免 KeyError。
    """
    main_goal: str
    plan: Union[HierarchicalPlan, None]
    working_memory: Annotated[dict, operator.add]
    current_sub_goal_id: Union[int, None]
    sub_task_raw_result: Union[dict, str, None]
    replan_count: int
    is_human_intervention_needed: bool
    response: str

# 輔助函式
def find_next_executable_goal(plan: HierarchicalPlan) -> Union[SubGoal, None]:
    completed_ids = {g.goal_id for g in plan.sub_goals if g.status == "completed"}
    for goal in sorted(plan.sub_goals, key=lambda g: g.goal_id):
        if goal.status == "pending" and all(dep in completed_ids for dep in goal.dependencies):
            return goal
    return None

def get_specialist_for_goal(goal: SubGoal) -> str:
    desc = goal.description.lower()
    if any(kw in desc for kw in ["knowledge", "save", "update", "sop"]):
        return "knowledge"
    return "research"

def update_plan_status(plan: HierarchicalPlan, goal_id: int, verdict: dict, raw_result: Any) -> HierarchicalPlan:
    plan_copy = plan.model_copy(deep=True)
    for goal in plan_copy.sub_goals:
        if goal.goal_id == goal_id:
            goal.status = verdict.get('new_status', 'failed')
            goal.raw_result = raw_result
            goal.result_summary = verdict.get('summary', 'No summary provided.')
            break
    return plan_copy


# 核心及子圖節點的
class AgentNodes:
    def __init__(self, max_replans=3):
        self.MAX_REPLANS = max_replans

    def meta_planner_node(self, state: AgentState) -> dict:
        logging.info("--- 元規劃器：生成/更新高階策略樹 ---")
        structured_planner_llm = get_langchain_gemini_pro().with_structured_output(HierarchicalPlan)
        previous_plan_summary = ""
        if state.get('plan'):
            previous_plan_summary = f"Previous plan execution summary: {json.dumps(state['working_memory'], indent=2)}. Please refine the plan based on this."
        prompt = f"""Create a hierarchical plan to achieve the user's goal. Break it down into sub-goals with dependencies. Available Tools: {[tool.name for tool in ALL_TOOLS]}. User's Goal: {state['main_goal']}. {previous_plan_summary}"""
        try:
            plan = structured_planner_llm.invoke([SystemMessage(content=prompt)])
            return {"plan": plan, "is_human_intervention_needed": False}
        except Exception as e:
            logging.error(f"元規劃器發生嚴重錯誤: {e}")
            return {"is_human_intervention_needed": True, "response": f"Fatal error in planning: {e}"}

    def executive_node(self, state: AgentState) -> dict:
        logging.info("--- 執行官：決策下一子任務 ---")
        if state.get('is_human_intervention_needed'):
            # 【語法修正】節點只返回狀態更新，不返回 END
            return {}
        plan = state.get('plan')
        if not plan:
            logging.warning("--- 執行官：未找到有效計畫，任務終止 ---")
            # 【語法修正】
            return {"is_human_intervention_needed": True, "response": "Execution stopped due to a missing plan."}
        next_goal = find_next_executable_goal(plan)
        if next_goal:
            logging.info(f"--- 執行官：分派子任務 '{next_goal.description}' ---")
            return {"current_sub_goal_id": next_goal.goal_id}
        else:
            logging.info("--- 執行官：所有任務完成，準備綜合報告 ---")
            return {}

    def execute_subgraph_node(self, state: AgentState) -> dict:
        goal_id = state['current_sub_goal_id']
        plan = state['plan']
        assert plan is not None, "Plan cannot be None in executor"
        assert goal_id is not None, "Goal ID cannot be None in executor"
        current_goal = next(g for g in plan.sub_goals if g.goal_id == goal_id)
        specialist_name = get_specialist_for_goal(current_goal)
        logging.info(f"--- 專家 [{specialist_name.capitalize()}]: 開始處理 '{current_goal.description}' ---")
        try:
            if specialist_name == "research":
                result = search_tool.invoke({"query": current_goal.description})
            elif specialist_name == "knowledge":
                result = DeepResearchKnowledgeBase.invoke({"query": current_goal.description})
            else:
                raise ValueError(f"未知的專家: {specialist_name}")
            return {"sub_task_raw_result": result}
        except Exception as e:
            logging.error(f"專家 [{specialist_name.capitalize()}] 工具執行失敗: {e}")
            return {"sub_task_raw_result": f"Error executing tool: {e}"}

    def reflection_node(self, state: AgentState) -> dict:
        logging.info("--- 高級反思器：評估子任務結果 ---")
        goal_id = state['current_sub_goal_id']
        plan = state['plan']
        assert plan is not None, "Plan cannot be None in reflector"
        assert goal_id is not None, "Goal ID cannot be None in reflector"

        raw_result = state.get('sub_task_raw_result')
        
        structured_reflection_llm = get_langchain_gemini_pro().with_structured_output(Verdict)
        prompt = f"""Critically evaluate the result of a sub-task. Original Goal: {state['main_goal']}. Sub-Goal: {plan.sub_goals[goal_id-1].description}. Result: {str(raw_result)[:2000]}. Decide the next action ('CONTINUE', 'REPLAN'). Return JSON with keys: 'summary', 'new_status', 'next_action'."""
        
        verdict = structured_reflection_llm.invoke([SystemMessage(content=prompt)])
        
        next_action = verdict.get('next_action', 'REPLAN') # pyright: ignore[reportAttributeAccessIssue]
        logging.info(f"--- 反思決策: {next_action} ---")
        
        updated_plan = update_plan_status(plan, goal_id, verdict, raw_result) # pyright: ignore[reportArgumentType]
        summary = verdict.get('summary', 'No summary provided.') # pyright: ignore[reportAttributeAccessIssue]
        
        replan_count = state.get('replan_count', 0)
        if next_action == 'REPLAN':
            replan_count += 1
        
        if replan_count >= self.MAX_REPLANS:
            logging.warning(f"--- 反思器：重新規劃次數已達上限 ({self.MAX_REPLANS}) ---")
            return {"is_human_intervention_needed": True, "response": "Agent stopped: Maximum replan limit reached."}

        return {
            "plan": updated_plan,
            "working_memory": {f"goal_{goal_id}_summary": summary},
            "next_action": next_action,
            "replan_count": replan_count,
            "sub_task_raw_result": None # 【語法修正】明確清理臨時狀態
        }

    def synthesizer_node(self, state: AgentState) -> dict:
        logging.info("--- 綜合節點：生成最終報告 ---")
        final_prompt = f"""Synthesize the results from the working memory into a final answer for the user's goal. Goal: {state['main_goal']}. Working Memory: {json.dumps(state['working_memory'], indent=2)}"""
        response = get_langchain_gemini_pro().invoke([SystemMessage(content=final_prompt)])
        return {"response": response.content}
        
    def human_intervention_node(self, state: AgentState) -> dict:
        logging.error("--- 任務已暫停，需要人工介入 ---")
        logging.error(f"最終狀態摘要: {state.get('response', 'N/A')}")
        return {}


def create_master_graph():
    logging.info("正在初始化 Production-Grade DEHP Agent Graph...")
    
    nodes = AgentNodes(max_replans=3)
    workflow = StateGraph(AgentState)

    workflow.add_node("meta_planner", nodes.meta_planner_node)
    workflow.add_node("executive", nodes.executive_node)
    workflow.add_node("executor", nodes.execute_subgraph_node)
    workflow.add_node("reflector", nodes.reflection_node)
    workflow.add_node("synthesizer", nodes.synthesizer_node)
    workflow.add_node("human_intervention", nodes.human_intervention_node)

    workflow.add_edge(START, "meta_planner")
    
    def route_from_planner(state: AgentState):
        if state.get("is_human_intervention_needed"):
            return "human_intervention"
        return "executive"
    
    workflow.add_conditional_edges("meta_planner", route_from_planner)
    
    def route_from_executive(state: AgentState):
        # 【語法修正】增加對 plan 的檢查
        if state.get("is_human_intervention_needed") or not state.get("plan"):
            return "human_intervention"
        if not find_next_executable_goal(state['plan']): # pyright: ignore[reportArgumentType]
            return "synthesizer"
        return "executor"

    workflow.add_conditional_edges("executive", route_from_executive)
    workflow.add_edge("executor", "reflector")

    def route_from_reflector(state: AgentState):
        if state.get("is_human_intervention_needed"):
            return "human_intervention"
        next_action = state.get("next_action")
        if next_action == "REPLAN":
            return "meta_planner"
        return "executive" # CONTINUE
    
    workflow.add_conditional_edges("reflector", route_from_reflector)
    workflow.add_edge("synthesizer", END)
    workflow.add_edge("human_intervention", END)
    
    from langgraph.checkpoint.memory import InMemorySaver
    memory = InMemorySaver()
    
    return workflow.compile(checkpointer=memory)