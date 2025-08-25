import operator
import logging
import json
from typing import TypedDict, Annotated, List, Union, Any, Literal, Optional
from functools import lru_cache

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 導入我們集中管理的 LLM 服務和工具列表
from core.services import get_langchain_gemini_pro, get_langchain_gemini_flash, get_langchain_gemini_flash_lite
from core.tool_registry import ALL_TOOLS
from core.tools.web_search import search_tool
from core.tools.rag_tool import DeepResearchKnowledgeBase
from core.utils import load_prompt

# 狀態和結構的定義
class RouteDecision(BaseModel):
    decision: Literal["simple_query", "complex_task"] = Field(description="根據使用者目標的複雜度，決定要走的路徑。")
    reasoning: str = Field(description="做出此決策的簡要理由，不超過 30 字。")


class ToolSelection(BaseModel):
    tool_name: str = Field(description="從可用工具列表中選擇的最合適的工具的確切名稱。")
    reasoning: str = Field(description="對為什麼選擇這個工具的簡要說明（約 25 字）。")

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
    main_goal: str
    plan: Union[HierarchicalPlan, None]
    working_memory: Annotated[dict, operator.add]
    current_sub_goal_id: Union[int, None]
    sub_task_raw_result: Union[dict, str, None]
    replan_count: int
    is_human_intervention_needed: bool
    response: str
    route_decision: str

# 輔助函式
def find_next_executable_goal(plan: HierarchicalPlan) -> Union[SubGoal, None]:
    completed_ids = {g.goal_id for g in plan.sub_goals if g.status == "completed"}
    for goal in sorted(plan.sub_goals, key=lambda g: g.goal_id):
        if goal.status.lower() == "pending" and all(dep in completed_ids for dep in goal.dependencies):
            return goal
    return None

async def get_specialist_for_goal_llm(currentgoal: SubGoal) -> str:
    logging.info(f"--- LLM Tool Router: 正在為目標 '{currentgoal.description[:50]}...' 選擇工具 ---")
    formatted_tools = "\n".join([f"---\n -工具名稱: {tool.name}\n -功能描述: {tool.description}" for tool in ALL_TOOLS])
    prompt_template = ChatPromptTemplate.from_messages([
                                                                    ("system", 
                                                                    """你是一個智能的工具路由專家。你的任務是根據使用者提供的子目標，從下面的可用工具列表中，選擇最適合完成該目標的單一工具。
                                                                    請仔細閱讀每個工具的功能描述來做出最佳判斷。

                                                                    可用工具列表:
                                                                    {tools_list}

                                                                    你必須以 JSON 格式回應，包含 `tool_name` 和 `reasoning` 兩個欄位。
                                                                    `tool_name` 必須與上面「工具名稱」中的一個完全匹配。"""),
                                                                    ("human", "子目標: {sub_goal}")
                                                                ])
    structured_llm = get_langchain_gemini_flash_lite().with_structured_output(ToolSelection)
    chain = prompt_template | structured_llm

    try:
        response = await chain.ainvoke({"tools_list": formatted_tools, "sub_goal": currentgoal.description})
        if response.tool_name in {tool.name for tool in ALL_TOOLS} : # pyright: ignore[reportAttributeAccessIssue]
            logging.info(f"--- LLM 路由決策: 工具 '{response.tool_name}'. 理由: {response.reasoning} ---") # pyright: ignore[reportAttributeAccessIssue]
            return response.tool_name # pyright: ignore[reportAttributeAccessIssue]
        else:
            logging.warning(f"--- LLM 路由警告: 模型回傳了不存在的工具 '{response.tool_name}'。將啟用備用方案。 ---") # pyright: ignore[reportAttributeAccessIssue]
            return ""
        
    except Exception as e:
        logging.error(f"--- LLM 路由嚴重錯誤: {e}. 將啟用備用方案。 ---")
        return ""

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
    def __init__(self, max_replans=3, tools: Optional[List[BaseTool]] = None):
        self.MAX_REPLANS = max_replans
        self.tool = {tool.name: tool for tool in ALL_TOOLS}
    
    async def router_node(self, state: AgentState) -> dict:
        logging.info("--- 路由節點：評估任務複雜度 ---")
        goal = state['main_goal']        
        try:
            router_llm = get_langchain_gemini_flash_lite().with_structured_output(RouteDecision)
            router_prompt = load_prompt("agent_router.txt")
            formatted_prompt = router_prompt.format(goal=goal)
            response = await router_llm.ainvoke(formatted_prompt)
            return {"route_decision": response.decision} # pyright: ignore[reportAttributeAccessIssue]
        
        except Exception as e:
            logging.warning(f"--- LLM 路由判斷失敗: {e}. 降級至規則判斷模式。 ---")

            if any(keyword in goal for keyword in ["分析", "比較", "總結", "規劃", "報告", "研究"]) or len(goal) > 80:
                logging.info(f"--- 規則決策：複雜任務 -> 啟動規劃流程 ---")
                return {"route_decision": "complex_task"}
            else:
                logging.info(f"--- 規則決策：簡單查詢 -> 啟動快速路徑 ---")
                return {"route_decision": "simple_query"}

    async def simple_query_executor_node(self, state: AgentState) -> dict:
        logging.info("--- 快速路徑：直接執行簡單查詢 ---")
        goal = state['main_goal']
        try:
            llm_with_tools = get_langchain_gemini_flash_lite().bind_tools(ALL_TOOLS)
            logging.info("--- 快速路徑：LLM 正在決策... ---")
            response_message = await llm_with_tools.ainvoke(goal)
            if response_message.tool_calls: # pyright: ignore[reportAttributeAccessIssue]
                logging.info(f"--- 快速路徑：偵測到工具呼叫: {[tc['name'] for tc in response_message.tool_calls]} ---") # pyright: ignore[reportAttributeAccessIssue]

                tool_messages = []
                for tool_call in response_message.tool_calls: # pyright: ignore[reportAttributeAccessIssue]
                    tool_name = tool_call.get('name')
                    tool_to_call = self.tool.get(tool_name)
                    
                    observation = ""
                    if not tool_to_call:
                        error_msg = f"錯誤：模型嘗試呼叫一個不存在的工具 '{tool_name}'。"
                        observation = error_msg
                        logging.error(error_msg)
                    else:
                        try:
                            logging.info(f"--- 快速路徑：正在執行工具 '{tool_name}'，參數: {tool_call['args']} ---")
                            # 執行工具並取得結果
                            observation = await tool_to_call.ainvoke(tool_call['args'])
                        except Exception as e:
                            observation = f"工具 '{tool_name}' 執行時發生錯誤: {e}"
                            logging.error(observation)

                    tool_messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call.get("id"))) # pyright: ignore[reportPossiblyUnboundVariable]
                logging.info("--- 快速路徑：將工具結果傳回 LLM 進行綜合整理 ---")
                final_response_message = await llm_with_tools.ainvoke([HumanMessage(content=goal), response_message] + tool_messages)
                logging.info(f"--- 快速路徑：最終生成的回覆內容: '{final_response_message.content}' ---")
                return {"response": final_response_message.content}
            
            else:
                logging.info("--- 快速路徑：LLM 直接生成回覆 ---")
                if not response_message.content:
                    return {"response": "模型選擇不使用工具，但未提供直接的回覆。"}
                return {"response": response_message.content}
        except Exception as e:
            logging.error(f"簡單查詢執行失敗: {e}")
            return {"response": f"處理您的請求時發生錯誤: {e}"}
        
    async def meta_planner_node(self, state: AgentState) -> dict:
        logging.info("--- 元規劃器：生成/更新高階策略樹 ---")
        structured_planner_llm = get_langchain_gemini_pro().with_structured_output(HierarchicalPlan)
        previous_plan_summary = ""
        if state.get('plan'):
            previous_plan_summary = f"Previous plan execution summary: {json.dumps(state['working_memory'], indent=2)}. Please refine the plan based on this."
        prompt = f"""Create a hierarchical plan to achieve the user's goal. Break it down into sub-goals with dependencies. Available Tools: {[tool.name for tool in ALL_TOOLS]}. User's Goal: {state['main_goal']}. {previous_plan_summary}"""
        try:
            plan = await structured_planner_llm.ainvoke(prompt)
            return {"plan": plan, "is_human_intervention_needed": False}
        except Exception as e:
            logging.error(f"元規劃器發生嚴重錯誤: {e}")
            return {"is_human_intervention_needed": True, "response": f"Fatal error in planning: {e}"}

    def executive_node(self, state: AgentState) -> dict:
        logging.info("--- 執行官：決策下一子任務 ---")
        if state.get('is_human_intervention_needed'):
            return {}
        plan = state.get('plan')
        if not plan:
            logging.warning("--- 執行官：未找到有效計畫，任務終止 ---")
            return {"is_human_intervention_needed": True, "response": "Execution stopped due to a missing plan."}
        next_goal = find_next_executable_goal(plan)
        if next_goal:
            logging.info(f"--- 執行官：分派子任務 '{next_goal.description}' ---")
            return {"current_sub_goal_id": next_goal.goal_id}
        else:
            logging.info("--- 執行官：所有任務完成，準備綜合報告 ---")
            return {}

    async def execute_subgraph_node(self, state: AgentState) -> dict:
        goal_id = state['current_sub_goal_id']
        plan = state['plan']
        assert plan is not None, "Plan cannot be None in executor"
        assert goal_id is not None, "Goal ID cannot be None in executor"
        current_goal = next(g for g in plan.sub_goals if g.goal_id == goal_id)
        chosen_tool_name = get_specialist_for_goal_llm(current_goal)
        logging.info(f"--- 專家 [{chosen_tool_name}]: 開始處理 '{current_goal.description}' ---")
        
        try:
            if chosen_tool_name:
                result = await self.tool.get(chosen_tool_name).ainvoke({"query": current_goal.description}) # pyright: ignore[reportOptionalMemberAccess]
                return {"sub_task_raw_result": result}
            else:
                raise ValueError(f"工具未在工具列表中找到適合的工具。")
        except Exception as e:
            logging.error(f"[{chosen_tool_name}] 工具執行失敗: {e}")
            return {"sub_task_raw_result": f"Error executing tool '{chosen_tool_name}': {e}"}

    async def reflection_node(self, state: AgentState) -> dict:
        logging.info("--- 高級反思器：評估子任務結果 ---")
        goal_id = state['current_sub_goal_id']
        plan = state['plan']
        assert plan is not None, "Plan cannot be None in reflector"
        assert goal_id is not None, "Goal ID cannot be None in reflector"
        raw_result = state.get('sub_task_raw_result')

        structured_reflection_llm = get_langchain_gemini_flash().with_structured_output(Verdict)
        prompt = f"""Critically evaluate the result of a sub-task. Original Goal: {state['main_goal']}. Sub-Goal: {plan.sub_goals[goal_id-1].description}. Result: {str(raw_result)[:2000]}. Decide the next action ('CONTINUE', 'REPLAN'). Return JSON with keys: 'summary', 'new_status', 'next_action'."""
        verdict = await structured_reflection_llm.ainvoke(prompt)
        verdict = verdict.model_dump() # pyright: ignore[reportAttributeAccessIssue]
        next_action = verdict.get('next_action', 'REPLAN') # pyright: ignore[reportAttributeAccessIssue]
        logging.info(f"--- 反思決策: {next_action} ---")
        updated_plan = update_plan_status(plan, goal_id, verdict, raw_result) # pyright: ignore[reportArgumentType]
        summary = verdict.get('summary', 'No summary provided.') # pyright: ignore[reportAttributeAccessIssue]
        replan_count = state.get('replan_count', 0)
        
        if next_action.upper() == 'REPLAN':
            replan_count += 1
        
        if replan_count >= self.MAX_REPLANS:
            logging.warning(f"--- 反思器：重新規劃次數已達上限 ({self.MAX_REPLANS}) ---")
            return {"is_human_intervention_needed": True, "response": "Agent stopped: Maximum replan limit reached."}

        return {
            "plan": updated_plan,
            "working_memory": {f"goal_{goal_id}_summary": summary},
            "next_action": next_action,
            "replan_count": replan_count,
            "sub_task_raw_result": None
        }

    async def synthesizer_node(self, state: AgentState) -> dict:
        logging.info("--- 綜合節點：生成最終報告 ---")
        final_prompt = f"""Synthesize the results from the working memory into a final answer for the user's goal. Goal: {state['main_goal']}. Working Memory: {json.dumps(state['working_memory'], indent=2)}"""
        response = await get_langchain_gemini_pro().ainvoke(final_prompt)
        return {"response": response.content}
        
    def human_intervention_node(self, state: AgentState) -> dict:
        logging.error("--- 任務已暫停，需要人工介入 ---")
        logging.error(f"最終狀態摘要: {state.get('response', 'N/A')}")
        return {}


def create_master_graph():
    logging.info("正在初始化 Production-Grade DEHP Agent Graph...")
    
    nodes = AgentNodes(max_replans=3)
    workflow = StateGraph(AgentState)

    workflow.add_node("router", nodes.router_node)
    workflow.add_node("simple_executor", nodes.simple_query_executor_node)
    workflow.add_node("meta_planner", nodes.meta_planner_node)
    workflow.add_node("executive", nodes.executive_node)
    workflow.add_node("executor", nodes.execute_subgraph_node)
    workflow.add_node("reflector", nodes.reflection_node)
    workflow.add_node("synthesizer", nodes.synthesizer_node)
    workflow.add_node("human_intervention", nodes.human_intervention_node)

    workflow.add_edge(START, "router")
    def route_logic(state: AgentState):
        return state["route_decision"]
    
    workflow.add_conditional_edges(source="router",
                                   path=route_logic,
                                   path_map={"simple_query": "simple_executor",
                                             "complex_task": "meta_planner"})
    
    workflow.add_edge(start_key="simple_executor", end_key=END)
    
    def route_from_planner(state: AgentState):
        if state.get("is_human_intervention_needed"):
            return "human_intervention"
        return "executive"
    
    workflow.add_conditional_edges(source="meta_planner", path=route_from_planner)
    
    def route_from_executive(state: AgentState):
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
        if next_action.upper() == "REPLAN": # pyright: ignore[reportOptionalMemberAccess]
            return "meta_planner"
        return "executive" # CONTINUE
    
    workflow.add_conditional_edges("reflector", route_from_reflector)
    workflow.add_edge("synthesizer", END)
    workflow.add_edge("human_intervention", END)
    
    from langgraph.checkpoint.memory import InMemorySaver
    memory = InMemorySaver()
    
    return workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    Actions = AgentNodes(max_replans=3)
    Query = {"main_goal": "鐵達尼號的導演是誰"}
    # Actions.router_node(Query)
    Query_from_route = {"main_goal": "鐵達尼號的導演是誰", "route_decision": 'simple_query'}
    # print(Actions.simple_query_executor_node(Query_from_route))
    # planner_Res = Actions.meta_planner_node(Query)
    # print(planner_Res)
    After_planner_state = {'plan': HierarchicalPlan(main_goal='找出鐵達尼號的導演是誰', 
                                                    sub_goals=[SubGoal(goal_id=1, description='使用 tavily_search 搜尋“鐵達尼號的導演”', dependencies=[], status='PENDING', raw_result=None, result_summary=''), 
                                                               SubGoal(goal_id=2, description='總結搜尋結果並回答導演是誰', dependencies=[1], status='PENDING', raw_result=None, result_summary='')]), 
                                                               'is_human_intervention_needed': False,
                                                               "main_goal": "鐵達尼號的導演是誰",}
    # print(Actions.executive_node(After_planner_state))
    After_executive_state = {'plan': HierarchicalPlan(main_goal='找出鐵達尼號的導演是誰', 
                                                    sub_goals=[SubGoal(goal_id=1, description='使用 tavily_search 搜尋“鐵達尼號的導演”', dependencies=[], status='PENDING', raw_result=None, result_summary=''), 
                                                               SubGoal(goal_id=2, description='總結搜尋結果並回答導演是誰', dependencies=[1], status='PENDING', raw_result=None, result_summary='')]), 
                                                               'is_human_intervention_needed': False,
                                                               "main_goal": "鐵達尼號的導演是誰",
                                                               'current_sub_goal_id': 1,}
    # print(Actions.execute_subgraph_node(After_executive_state))
    After_execute_state = {'plan': HierarchicalPlan(main_goal='找出鐵達尼號的導演是誰', 
                                                    sub_goals=[SubGoal(goal_id=1, description='使用 tavily_search 搜尋“鐵達尼號的導演”', dependencies=[], status='PENDING', raw_result=None, result_summary=''), 
                                                               SubGoal(goal_id=2, description='總結搜尋結果並回答導演是誰', dependencies=[1], status='PENDING', raw_result=None, result_summary='')]), 
                                                               'is_human_intervention_needed': False,
                                                               "main_goal": "鐵達尼號的導演是誰",
                                                               'current_sub_goal_id': 1,
                                                               'sub_task_raw_result': {'query': '使用 tavily_search 搜尋“鐵達尼號的導演”', 
                                                                                       'follow_up_questions': None, 
                                                                                       'answer': None, 
                                                                                       'images': [], 
                                                                                       'results': [{'url': 'https://zh.wikipedia.org/zh-tw/%E6%B3%B0%E5%9D%A6%E5%B0%BC%E5%85%8B%E5%8F%B7_(1997%E5%B9%B4%E7%94%B5%E5%BD%B1)', 
                                                                                                    'title': '鐵達尼號(1997年電影) - 維基百科', 
                                                                                                    'content': '《鐵達尼號》（英語：Titanic）是一部於1997年上映的美國史詩浪漫災難電影，由詹姆士·卡麥隆創作、導演、監製、共同製作及共同編輯。電影部分情節是根據1912年4月14日至15', 
                                                                                                    'score': 0.6227582, 
                                                                                                    'raw_content': None},
                                                                                                    {'url': 'https://www.threads.com/@daniel.moviegoer/post/DGsj2UbzHd-/6-titanic%E9%90%B5%E9%81%94%E5%B0%BC%E8%99%9F%E5%B0%8E%E6%BC%94james-cameron-19977-schindlers-list-%E8%BE%9B%E5%BE%B7%E5%8B%92%E7%9A%84%E5%90%8D%E5%96%AE-%E5%B0%8E%E6%BC%94steven-spielberg-199', 
                                                                                                     'title': '6. 《Titanic鐵達尼號》（導演：James Cameron, 1997) 7. ...', 
                                                                                                     'content': "6. 《Titanic鐵達尼號》（導演：James Cameron, 1997) 7. 《Schindler's List 辛德勒的名單》 (導演：Steven Spielberg, 1993) 8. 《The", 
                                                                                                     'score': 0.6150191, 
                                                                                                     'raw_content': None}, 
                                                                                                     {'url': 'https://tw.news.yahoo.com/%E3%80%8C%E8%AD%A6%E5%91%8A%E9%83%BD%E4%B8%8D%E8%81%BD%E3%80%8D%E5%90%8D%E5%B0%8E%E8%A9%B9%E5%A7%86%E6%96%AF%E5%8D%A1%E9%BA%A5%E9%9A%86%E6%AD%8E%EF%BC%9A%E6%B3%B0%E5%9D%A6%E8%99%9F%E6%82%B2%E5%8A%87%E5%A6%82%E9%90%B5%E9%81%94%E5%B0%BC%E8%99%9F%E4%BA%8B%E4%BB%B6%E9%87%8D%E6%BC%94-024027016.html', 
                                                                                                      'title': '「警告都不聽」名導詹姆斯‧卡麥隆歎：泰坦號悲劇如鐵達尼號 ...', 
                                                                                                      'content': '電影《鐵達尼號》（Titanic）導演兼深海探險家詹姆斯‧ 卡麥隆說，關於這具旅遊潛水器的許多安全警告均遭忽視，並感歎泰坦號悲劇如鐵達尼事件重演。泰坦號', 
                                                                                                      'score': 0.55072874, 
                                                                                                      'raw_content': None}], 
                                                                                        'response_time': 0.77, 
                                                                                        'request_id': 'dca737c5-588e-4db2-aa13-94c34abcf495'},
                                                                                        }
    # print(Actions.reflection_node(After_execute_state))
    After_reflection_state = {'plan': HierarchicalPlan(main_goal='找出鐵達尼號的導演是誰', 
                                                        sub_goals=[SubGoal(goal_id=1, 
                                                                            description='使用tavily_search 搜尋“鐵達尼號的導演”', 
                                                                            dependencies=[], 
                                                                            status='completed', 
                                                                            raw_result={'query': '使用 tavily_search 搜尋“鐵達尼號的導演”', 
                                                                                        'follow_up_questions': None, 
                                                                                        'answer': None, 
                                                                                        'images': [], 
                                                                                        'results': [{'url': 'https://zh.wikipedia.org/zh-tw/%E6%B3%B0%E5%9D%A6%E5%B0%BC%E5%85%8B%E5%8F%B7_(1997%E5%B9%B4%E7%94%B5%E5%BD%B1)', 
                                                                                                    'title': '鐵達尼號(1997年電影) - 維基百科', 
                                                                                                    'content': '《鐵達尼號》（英語：Titanic）是一部於1997年上映的美國史詩浪漫災難電影，由詹姆士·卡麥隆創作、導演、監製、共同製作及共同編輯。電影部分情節是根據1912年4月14日至15', 
                                                                                                    'score': 0.6227582, 
                                                                                                    'raw_content': None}, 
                                                                                                    {'url': 'https://www.threads.com/@daniel.moviegoer/post/DGsj2UbzHd-/6-titanic%E9%90%B5%E9%81%94%E5%B0%BC%E8%99%9F%E5%B0%8E%E6%BC%94james-cameron-19977-schindlers-list-%E8%BE%9B%E5%BE%B7%E5%8B%92%E7%9A%84%E5%90%8D%E5%96%AE-%E5%B0%8E%E6%BC%94steven-spielberg-199', 
                                                                                                    'title': '6. 《Titanic鐵達尼號》（導演：James Cameron, 1997) 7. ...', 
                                                                                                    'content': "6. 《Titanic鐵達尼號》（導演：James Cameron, 1997) 7. 《Schindler's List 辛德勒的名單》 (導演：Steven Spielberg, 1993) 8. 《The", 
                                                                                                    'score': 0.6150191, 
                                                                                                    'raw_content': None}, 
                                                                                                    {'url': 'https://tw.news.yahoo.com/%E3%80%8C%E8%AD%A6%E5%91%8A%E9%83%BD%E4%B8%8D%E8%81%BD%E3%80%8D%E5%90%8D%E5%B0%8E%E8%A9%B9%E5%A7%86%E6%96%AF%E5%8D%A1%E9%BA%A5%E9%9A%86%E6%AD%8E%EF%BC%9A%E6%B3%B0%E5%9D%A6%E8%99%9F%E6%82%B2%E5%8A%87%E5%A6%82%E9%90%B5%E9%81%94%E5%B0%BC%E8%99%9F%E4%BA%8B%E4%BB%B6%E9%87%8D%E6%BC%94-024027016.html', 
                                                                                                    'title': '「警告都不聽」名導詹姆斯‧卡麥隆歎：泰坦號悲劇如鐵達尼號 ...', 
                                                                                                    'content': '電影《鐵達尼號》（Titanic）導演兼深海探險家詹姆斯‧ 卡麥隆說，關於這具旅遊潛水器的許多安全警告均遭忽視，並感歎泰坦號悲劇如鐵達尼事件重演。泰坦號', 
                                                                                                    'score': 0.55072874, 
                                                                                                    'raw_content': None}], 
                                                                                                    'response_time': 0.77, 
                                                                                                    'request_id': 'dca737c5-588e-4db2-aa13-94c34abcf495'}, 
                                                                            result_summary='Successfully found that the director of Titanic is James Cameron.'), 
                                                                    SubGoal(goal_id=2, 
                                                                            description='總結搜尋結果並回答導演是誰', 
                                                                            dependencies=[1], 
                                                                            status='PENDING', 
                                                                            raw_result=None, 
                                                                            result_summary='')]), 
                                'working_memory': {'goal_1_summary': 'Successfully found that the director of Titanic is James Cameron.'}, 
                                'next_action': 'CONTINUE', 
                                'replan_count': 0, 
                                'sub_task_raw_result': None,
                                'is_human_intervention_needed': False,
                                "main_goal": "鐵達尼號的導演是誰",
                                'current_sub_goal_id': 1
                                }
    # print(Actions.executive_node(After_reflection_state))
    # Actions.executive_node(After_reflection_state)
    After_executive2_state = {'plan': HierarchicalPlan(main_goal='找出鐵達尼號的導演是誰', 
                                                        sub_goals=[SubGoal(goal_id=1, 
                                                                            description='使用tavily_search 搜尋“鐵達尼號的導演”', 
                                                                            dependencies=[], 
                                                                            status='completed', 
                                                                            raw_result={'query': '使用 tavily_search 搜尋“鐵達尼號的導演”', 
                                                                                        'follow_up_questions': None, 
                                                                                        'answer': None, 
                                                                                        'images': [], 
                                                                                        'results': [{'url': 'https://zh.wikipedia.org/zh-tw/%E6%B3%B0%E5%9D%A6%E5%B0%BC%E5%85%8B%E5%8F%B7_(1997%E5%B9%B4%E7%94%B5%E5%BD%B1)', 
                                                                                                    'title': '鐵達尼號(1997年電影) - 維基百科', 
                                                                                                    'content': '《鐵達尼號》（英語：Titanic）是一部於1997年上映的美國史詩浪漫災難電影，由詹姆士·卡麥隆創作、導演、監製、共同製作及共同編輯。電影部分情節是根據1912年4月14日至15', 
                                                                                                    'score': 0.6227582, 
                                                                                                    'raw_content': None}, 
                                                                                                    {'url': 'https://www.threads.com/@daniel.moviegoer/post/DGsj2UbzHd-/6-titanic%E9%90%B5%E9%81%94%E5%B0%BC%E8%99%9F%E5%B0%8E%E6%BC%94james-cameron-19977-schindlers-list-%E8%BE%9B%E5%BE%B7%E5%8B%92%E7%9A%84%E5%90%8D%E5%96%AE-%E5%B0%8E%E6%BC%94steven-spielberg-199', 
                                                                                                    'title': '6. 《Titanic鐵達尼號》（導演：James Cameron, 1997) 7. ...', 
                                                                                                    'content': "6. 《Titanic鐵達尼號》（導演：James Cameron, 1997) 7. 《Schindler's List 辛德勒的名單》 (導演：Steven Spielberg, 1993) 8. 《The", 
                                                                                                    'score': 0.6150191, 
                                                                                                    'raw_content': None}, 
                                                                                                    {'url': 'https://tw.news.yahoo.com/%E3%80%8C%E8%AD%A6%E5%91%8A%E9%83%BD%E4%B8%8D%E8%81%BD%E3%80%8D%E5%90%8D%E5%B0%8E%E8%A9%B9%E5%A7%86%E6%96%AF%E5%8D%A1%E9%BA%A5%E9%9A%86%E6%AD%8E%EF%BC%9A%E6%B3%B0%E5%9D%A6%E8%99%9F%E6%82%B2%E5%8A%87%E5%A6%82%E9%90%B5%E9%81%94%E5%B0%BC%E8%99%9F%E4%BA%8B%E4%BB%B6%E9%87%8D%E6%BC%94-024027016.html', 
                                                                                                    'title': '「警告都不聽」名導詹姆斯‧卡麥隆歎：泰坦號悲劇如鐵達尼號 ...', 
                                                                                                    'content': '電影《鐵達尼號》（Titanic）導演兼深海探險家詹姆斯‧ 卡麥隆說，關於這具旅遊潛水器的許多安全警告均遭忽視，並感歎泰坦號悲劇如鐵達尼事件重演。泰坦號', 
                                                                                                    'score': 0.55072874, 
                                                                                                    'raw_content': None}], 
                                                                                                    'response_time': 0.77, 
                                                                                                    'request_id': 'dca737c5-588e-4db2-aa13-94c34abcf495'}, 
                                                                            result_summary='Successfully found that the director of Titanic is James Cameron.'), 
                                                                    SubGoal(goal_id=2, 
                                                                            description='總結搜尋結果並回答導演是誰', 
                                                                            dependencies=[1], 
                                                                            status='PENDING', 
                                                                            raw_result=None, 
                                                                            result_summary='')]), 
                                'working_memory': {'goal_1_summary': 'Successfully found that the director of Titanic is James Cameron.'}, 
                                'next_action': 'CONTINUE', 
                                'replan_count': 0, 
                                'sub_task_raw_result': None,
                                'is_human_intervention_needed': False,
                                "main_goal": "鐵達尼號的導演是誰",
                                'current_sub_goal_id': 2
                                }
    # print(Actions.execute_subgraph_node(After_executive2_state))



    

   
