import logging
import json
from typing import TypedDict, Annotated, List, Union, Any, Literal, Optional
from functools import lru_cache

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 導入LLM 服務和工具列表
from core.services import get_langchain_gemini_pro, get_langchain_gemini_flash, get_langchain_gemini_flash_lite
from core.tool_registry import ALL_TOOLS
from core.utils import load_prompt

system_prompt_content = load_prompt("agent_system.txt") 

# 狀態和結構的定義
class RouteDecision(BaseModel):
    decision: Literal["simple_query", "complex_task", "knowledge_update_task"] = Field(description="根據使用者目標的複雜度，決定要走的路徑。")
    reasoning: str = Field(description="做出此決策的簡要理由，不超過 30 字。")

class ToolSelection(BaseModel):
    tool_name: str = Field(description="從可用工具列表中選擇的最合適的工具的確切名稱。")
    reasoning: str = Field(description="對為什麼選擇這個工具的簡要說明（約 25 字）。")

class ExecutiveDecision(BaseModel):
    decision: Literal["CONTINUE", "REPLAN", "SYNTHESIZE"] = Field(description="根據當前計畫狀態和工作記憶體做出的核心決策。")
    next_goal_id: Optional[int] = Field(None, description="如果決策是 'CONTINUE'，則指定下一個要執行的子目標 ID。")
    reasoning: str = Field(description="做出此決策的簡要理由，用於日誌記錄和除錯。")

class SubGoal(BaseModel):
    goal_id: int
    description: str
    dependencies: List[int] = Field(default_factory=list)
    status: str = "todo"
    raw_result: Union[dict, str, None] = None
    result_summary: str = ""

class HierarchicalPlan(BaseModel):
    main_goal: str
    sub_goals: List[SubGoal]

class Verdict(BaseModel):
    summary: str = Field(description="A brief summary of the sub-task's result.")
    new_status: str = Field(description="The new status for the sub-goal, e.g., 'completed' or 'failed'.")
    next_action: str = Field(description="The next action to take: 'CONTINUE', 'REPLAN', or 'ABORT'.")
    reasoning: str = Field(description="A detailed reasoning for your decision.")

class AgentState(TypedDict):
    main_goal: str
    plan: Union[HierarchicalPlan, None]
    working_memory: Annotated[dict, lambda left, right: {**left, **right}]
    current_sub_goal_id: Union[int, None]
    sub_goal_failure_counts: Annotated[dict, lambda left, right: {**left, **right}]
    sub_task_raw_result: Union[dict, str, None]
    replan_count: int
    is_human_intervention_needed: bool
    response: str
    route_decision: str
    

# 輔助函式
def find_next_executable_goal(plan: HierarchicalPlan) -> Union[SubGoal, None]:
    completed_ids = {g.goal_id for g in plan.sub_goals if g.status == "completed"}
    for goal in sorted(plan.sub_goals, key=lambda g: g.goal_id):
        if goal.status.lower() in ["pending", "todo", "待執行"] and all(dep in completed_ids for dep in goal.dependencies):
            return goal
    return None

def is_plan_stuck(plan: HierarchicalPlan) -> bool:
    has_pending_goals = any(g.status.lower() in ["pending", "todo", "待執行"] for g in plan.sub_goals)
    next_executable = find_next_executable_goal(plan)
    return has_pending_goals and next_executable is None

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
        if response and hasattr(response, 'tool_name') and response.tool_name in {tool.name for tool in ALL_TOOLS}: # pyright: ignore[reportAttributeAccessIssue] # pyright: ignore[reportAttributeAccessIssue]
            logging.info(f"--- LLM 路由決策: 工具 '{response.tool_name}'. 理由: {response.reasoning} ---") # pyright: ignore[reportAttributeAccessIssue]
            return response.tool_name # pyright: ignore[reportAttributeAccessIssue]
        else:
            tool_name_str = getattr(response, 'tool_name', 'None')
            logging.warning(f"--- LLM 路由警告: 模型回傳了無效或不存在的工具 '{tool_name_str}'。將啟用備用方案。 ---")
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
    def __init__(self, max_replans=3, max_subgoal_retries=2, tools: Optional[List[BaseTool]] = None):
        self.MAX_REPLANS = max_replans
        self.MAX_SUBGOAL_RETRIES = max_subgoal_retries
        self.tool = {tool.name: tool for tool in tools} # pyright: ignore[reportOptionalIterable]
        self.prompts = {"router_prompt": load_prompt("agent_router.txt"),
                        "simple_query_prompt": load_prompt("agent_simple_executor.txt"),
                        "planner_prompt": load_prompt("agent_planner.txt"),
                        "executive_prompt": load_prompt("agent_executive.txt"),}
    
    async def router_node(self, state: AgentState) -> dict:
        logging.info("--- 路由節點：評估任務複雜度 ---")
        goal = state['main_goal']        
        try:
            router_llm = get_langchain_gemini_flash_lite().with_structured_output(RouteDecision)
            response = await router_llm.ainvoke(self.prompts["router_prompt"].format(goal=goal))
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
        messages = [SystemMessage(content=self.prompts["simple_query_prompt"]),
                    HumanMessage(content=state['main_goal']),]
        try:
            llm_with_tools = get_langchain_gemini_flash_lite().bind_tools(ALL_TOOLS)
            logging.info("--- 快速路徑：LLM 正在決策... ---")
            response_message = await llm_with_tools.ainvoke(messages)
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
                            observation = await tool_to_call.ainvoke(tool_call['args'])
                        except Exception as e:
                            observation = f"工具 '{tool_name}' 執行時發生錯誤: {e}"
                            logging.error(observation)
                    tool_messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call.get("id"))) # pyright: ignore[reportPossiblyUnboundVariable]
                
                messages.extend(tool_messages)
                logging.info("--- 快速路徑：將工具結果傳回 LLM 進行綜合整理 ---")
                final_response_message = await llm_with_tools.ainvoke(messages)
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
        structured_planner_llm = get_langchain_gemini_flash().with_structured_output(HierarchicalPlan)
        previous_plan_summary = ""
        if state.get('plan') and state.get('working_memory'):
            previous_plan_summary = ("先前的計畫執行遭遇困難或結果不佳。以下是相關的歷史執行摘要，請務必基於此進行根源分析並提出改良計畫：\n"
                                     f"```json\n{json.dumps(state['working_memory'], indent=2, ensure_ascii=False)}\n```")
        prompt = self.prompts["planner_system"].format(main_goal=state['main_goal'],
                                                       previous_plan_summary=previous_plan_summary,)
        logging.info(f"--- [DEBUG] 準備發送給 Planner 的最終提示詞: ---\n{prompt}\n--- [DEBUG] ---")
        
        try:
            plan = await structured_planner_llm.ainvoke(prompt)
            plan_dict = plan.model_dump() if hasattr(plan, 'model_dump') else plan # pyright: ignore[reportAttributeAccessIssue]
            logging.info(f"--- [DEBUG] 元規劃器生成的計畫詳情: ---\n{json.dumps(plan_dict, indent=2, ensure_ascii=False)}\n--- [DEBUG] ---")
            if not plan or not plan.sub_goals: # pyright: ignore[reportAttributeAccessIssue]
                logging.error("元規劃器未能生成有效的計畫內容 (回傳為空或沒有子目標)。")
                return {"is_human_intervention_needed": True,
                        "response": "Fatal error in planning: The meta planner failed to generate a valid plan structure.",}
            logging.info(f"--- 元規劃器：成功生成計畫，包含 {len(plan.sub_goals)} 個子目標 ---") # pyright: ignore[reportAttributeAccessIssue]
            return {"plan": plan, "is_human_intervention_needed": False}
        except Exception as e:
            logging.error(f"元規劃器發生嚴重錯誤: {e}")
            return {"is_human_intervention_needed": True, "response": f"Fatal error in planning: {e}"}

    async def executive_node(self, state: AgentState) -> dict:
        logging.info("--- 執行官：由 LLM 評估計畫並決策下一步 ---")
        if state.get('is_human_intervention_needed'):
            logging.warning("--- 執行官：偵測到需要人工介入，跳過決策 ---")
            return {}
        plan = state.get('plan')
        if not plan:
            logging.error("--- 執行官：未找到有效計畫，任務終止 ---")
            return {"is_human_intervention_needed": True, "response": "Execution stopped due to a missing plan."}
        plan_json = json.dumps(plan.model_dump(), indent=2, ensure_ascii=False)
        working_memory_json = json.dumps(state.get('working_memory', {}), indent=2, ensure_ascii=False)
        prompt = self.prompts["executive_prompt"].format(main_goal=state['main_goal'],
                                                         plan_json=plan_json,
                                                         working_memory_json=working_memory_json)
        try:
            structured_executive_llm = get_langchain_gemini_flash().with_structured_output(ExecutiveDecision)
            logging.info("--- 執行官：正在請求 LLM 進行 A.C.T. 決策... ---")
            response_model = await structured_executive_llm.ainvoke(prompt)
            decision = response_model.decision # pyright: ignore[reportAttributeAccessIssue]
            reasoning = response_model.reasoning # pyright: ignore[reportAttributeAccessIssue]
            next_goal_id = response_model.next_goal_id # pyright: ignore[reportAttributeAccessIssue]
            logging.info(f"--- 執行官 LLM 決策: '{decision}'. 理由: {reasoning} ---")

            if decision == "CONTINUE":
                if next_goal_id is None:
                    logging.error("--- 執行官錯誤：決策為 CONTINUE 但未提供 next_goal_id，強制重新規劃 ---")
                    return {"next_action": "REPLAN"}
                logging.info(f"--- 執行官：分派子任務 ID '{next_goal_id}' ---")
                return {"current_sub_goal_id": next_goal_id}
            
            elif decision == "REPLAN":
                logging.warning("--- 執行官：根據 LLM 判斷，計畫需要重新規劃 ---")
                return {"next_action": "REPLAN"} 

            elif decision == "SYNTHESIZE":
                logging.info("--- 執行官：所有任務完成，準備綜合報告 ---")
                return {"current_sub_goal_id": None}
            
            else:
                logging.error(f"--- 執行官：收到未知的 LLM 決策 '{decision}'，觸發重新規劃作為備用方案 ---")
                return {"next_action": "REPLAN"}

        except Exception as e:
            logging.error(f"--- 執行官節點 LLM 決策失敗: {e}。將觸發重新規劃。 ---", exc_info=True)
            return {"next_action": "REPLAN"}
    
    async def execute_subgraph_node(self, state: AgentState) -> dict:
        goal_id = state['current_sub_goal_id']
        plan = state['plan']
        assert plan is not None, "Plan cannot be None in executor"
        assert goal_id is not None, "Goal ID cannot be None in executor"
        
        current_goal = next(g for g in plan.sub_goals if g.goal_id == goal_id)
        chosen_tool_name = await get_specialist_for_goal_llm(current_goal)
        logging.info(f"--- 專家 [{chosen_tool_name}]: 開始處理 '{current_goal.description}' ---")
        
        try:
            if chosen_tool_name and chosen_tool_name in self.tool:
                tool_to_call = self.tool[chosen_tool_name]
                tool_schema_properties = tool_to_call.get_input_schema().schema().get('properties', {})
                invoke_input = {}
                if 'context' in tool_schema_properties:
                    invoke_input['context'] = state.get('working_memory', {})
                primary_input_key = next((key for key in tool_schema_properties if key != 'context'), None)
                if primary_input_key:
                    invoke_input[primary_input_key] = current_goal.description
                if len(tool_schema_properties) == 1 and 'context' not in tool_schema_properties:
                    single_key = list(tool_schema_properties.keys())[0]
                    final_input = {single_key: current_goal.description}
                elif not invoke_input and tool_schema_properties:
                     logging.warning(f"--- 執行官警告: 無法為工具 '{chosen_tool_name}' 建構有效的輸入。")
                     final_input = current_goal.description
                else:
                    final_input = invoke_input
                logging.info(f"--- 執行官：準備以如下參數呼叫工具 '{chosen_tool_name}': {list(final_input.keys()) if isinstance(final_input, dict) else 'String Input'} ---")
                result = await tool_to_call.ainvoke(final_input)
                return {"sub_task_raw_result": result}
            else:
                logging.warning(f"--- 未找到目標 '{current_goal.description}' 的特定工具，使用通用 LLM 處理 ---")
                general_llm = get_langchain_gemini_flash()
                result = await general_llm.ainvoke(current_goal.description)
                return {"sub_task_raw_result": result.content}
        except Exception as e:
            logging.error(f"[{chosen_tool_name}] 工具執行失敗: {e}")
            return {"sub_task_raw_result": f"Error executing tool '{chosen_tool_name}': {e}"}

    async def reflection_node(self, state: AgentState) -> dict:
        logging.info("--- 高級反思器：評估子任務結果 ---")
        goal_id = state['current_sub_goal_id']
        plan = state['plan']
        assert plan is not None, "Plan cannot be None in reflector"
        assert goal_id is not None, "Goal ID cannot be None in reflector"
        main_goal = state['main_goal']
        current_goal = next(g for g in plan.sub_goals if g.goal_id == goal_id)
        working_memory_str = json.dumps(state.get('working_memory', {}), indent=2, ensure_ascii=False)
        raw_result = str(state.get('sub_task_raw_result', ''))
        failure_counts = state.get('sub_goal_failure_counts', {})
        current_failures = failure_counts.get(goal_id, 0)
        is_fatal_error = raw_result.strip().startswith("FATAL:")

        if is_fatal_error:
            logging.error(f"--- 反思器：偵測到子任務 {goal_id} 的致命錯誤，將直接判定失敗。---")
            verdict = {
                "summary": f"工具執行遭遇不可恢復的錯誤: {raw_result}",
                "new_status": "failed",
                "next_action": "ABORT", # 強制中止
                "reasoning": "A fatal, non-recoverable error was returned by the tool."
            }
        else:
            prompt = f"""
            你是一位經驗豐富的專案經理，你的核心職責是評估子任務的執行結果，並決定專案的下一步走向。
            你需要有智慧地判斷，而不是機械地追求完美。

            **1. 專案的完整上下文:**
            - **總體目標 (Main Goal):** {main_goal}
            - **當前子目標 (Current Sub-goal):** {current_goal.description}
            - **已知的背景資訊 (Working Memory):** ```json
            {working_memory_str}
            ```
            - **子任務的原始執行結果 (Result to Evaluate):** ```
            {raw_result[:2500]}
            ```
            
            **2. 你的決策框架 (DECISION FRAMEWORK):**
            - **A. 任務類型判斷 (Task Type Analysis):**
            - 這是**初步的資訊蒐集任務**嗎 (例如，使用搜尋工具)？
            - 還是**後續的處理/分析任務** (例如，整理、規劃、總結)？

            - **B. 核心評估標準 (CRITICAL EVALUATION CRITERIA):**
            - **對於資訊蒐集任務:**
                - **相關性 (Relevance):** 結果是否與「當前子目標」高度相關？
                - **充分性 (Sufficiency):** 結果是否提供了【足夠的基礎資訊】，讓【下一個】子任務可以繼續進行？(注意：這裡不需要「完全詳盡」，只需要「足夠下一步」即可)。
            - **對於處理/分析任務:**
                - **目標達成度 (Goal Completion):** 結果是否【直接且完整地】回答了「當前子目標」？
                - **品質 (Quality):** 結果是否清晰、結構化且沒有明顯錯誤？

            - **C. 內容安全審核 (Content Safety Check):**
                - 結果中是否包含任何不當、攻擊性、或帶有強烈偏見的內容？

            **3. 你的行動指令 (ACTIONABLE INSTRUCTIONS):**
            - **第一步：生成摘要 (Generate Summary):**
            - 無論你的最終決策是什麼，都必須先根據「子任務的原始執行結果」，生成一份【客觀、中立、安全】的摘要，移除所有主觀或不安全的內容。
            - **第二步：做出裁決 (Make a Verdict):**
            - 如果結果是**完全無關**的、**錯誤的**，或包含**不安全內容**，則你的決策是 `REPLAN`。
            - 對於**資訊蒐集任務**，只要結果滿足【相關性】和【充分性】，即使不夠完美，你的決策也【應該是 `CONTINUE`】，以推動專案進程。
            - 對於**處理/分析任務**，你需要更嚴格地評估其【目標達成度】和【品質】，若不滿足則決策為 `REPLAN`。
            - 在絕大多數情況下，只要我們獲得了有用的新資訊，就應該選擇 `CONTINUE`。

            請以 JSON 格式回傳你的最終分析報告。JSON 必須包含以下欄位：
            - "summary": (字串) 你生成的【安全且中立】的摘要。
            - "new_status": (字串) 根據你的評估，將子目標的新狀態設為 'completed' 或 'failed'。
            - "next_action": (字串) 你的最終決策：'CONTINUE' 或 'REPLAN'。
            - "reasoning": (字串) 你做出此決策的詳細理由，必須明確引用上述的決策框架。
            """
            structured_reflection_llm = get_langchain_gemini_flash().with_structured_output(Verdict)
            verdict_model = await structured_reflection_llm.ainvoke(prompt)
            if verdict_model is None:
                logging.error("--- 反思器嚴重錯誤: 模型未能生成有效的裁決 (可能觸發了內容安全審核). ---")
                verdict = {
                    "summary": "Reflection failed: The model did not return a valid verdict.",
                    "new_status": "failed",
                    "next_action": "REPLAN",
                    "reasoning": "The reflection model returned no output, possibly due to content safety filters or an API error."
                }
            else:
                verdict = verdict_model.model_dump() # pyright: ignore[reportAttributeAccessIssue]
        
        # 處理失敗計數與決策覆寫
        if verdict.get('new_status') == 'failed' or verdict.get('next_action') == 'ABORT':
            current_failures += 1
            failure_counts[goal_id] = current_failures
            logging.warning(f"--- 反思器：子任務 {goal_id} 失敗。這是第 {current_failures} 次失敗。---")

            if current_failures >= self.MAX_SUBGOAL_RETRIES:
                logging.error(f"--- 反思器：子任務 {goal_id} 已達最大重試次數 ({self.MAX_SUBGOAL_RETRIES})。強制中止！---")
                final_response_message = f"代理程式中止：子任務 '{current_goal.description}' 在嘗試 {current_failures} 次後仍然失敗。"
                updated_plan = update_plan_status(plan, goal_id, verdict, raw_result)
                return {
                    "plan": updated_plan,
                    "is_human_intervention_needed": True,
                    "response": final_response_message,
                    "sub_goal_failure_counts": failure_counts
                }

        next_action = verdict.get('next_action', 'REPLAN').upper()
        logging.info(f"--- 反思決策: {next_action}. 理由: {verdict.get('reasoning', 'N/A')} ---")
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
            "working_memory": {**state.get('working_memory',{}), f"goal_{goal_id}_summary": summary},
            "next_action": next_action,
            "replan_count": replan_count,
            "sub_task_raw_result": None,
            "sub_goal_failure_counts": failure_counts
        }

    async def synthesizer_node(self, state: AgentState) -> dict:
        logging.info("--- 綜合節點：生成最終報告 ---")
        final_prompt = f"""Synthesize the results from the working memory into a final answer for the user's goal. Goal: {state['main_goal']}. Working Memory: {json.dumps(state['working_memory'], indent=2, ensure_ascii=False)}"""
        logging.info(f"--- [DEBUG] 準備發送給 Synthesizer 的最終提示詞: ---\n{final_prompt}\n--- [DEBUG] ---")
        try:
            response = await get_langchain_gemini_pro().ainvoke(final_prompt)
            return {"response": response.content}
        except ValueError as e:
            if "No generations found in stream" in str(e):
                error_message = "綜合報告生成失敗：模型因內容審核或其他原因未回傳任何結果。可能是因為蒐集的資料中包含了觸發安全機制的內容。"
                logging.error(f"--- 綜合節點錯誤: {error_message} ---")
                return {"response": f"抱歉，在為您生成最終報告時發生錯誤。原因：{error_message}"}
            else:
                raise e
        except Exception as e:
            error_message = f"綜合報告生成時發生未預期的嚴重錯誤: {e}"
            logging.error(f"--- 綜合節點嚴重錯誤: {error_message} ---", exc_info=True)
            return {"response": f"抱歉，系統在生成最終報告時發生嚴重錯誤。"}
        
    def human_intervention_node(self, state: AgentState) -> dict:
        logging.error("--- 任務已暫停，需要人工介入 ---")
        logging.error(f"最終狀態摘要: {state.get('response', 'N/A')}")
        return {}


def create_master_graph():
    logging.info("正在初始化 Production-Grade DEHP Agent Graph...")
    
    nodes = AgentNodes(max_replans=3, max_subgoal_retries=2, tools = ALL_TOOLS)
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
                                             "complex_task": "meta_planner",
                                             "knowledge_update_task": "meta_planner"})
    
    workflow.add_edge(start_key="simple_executor", end_key=END)
    
    def route_from_planner(state: AgentState):
        if state.get("is_human_intervention_needed"):
            return "human_intervention"
        return "executive"
    workflow.add_conditional_edges(source="meta_planner", path=route_from_planner)
    
    def route_from_executive(state: AgentState):
        if state.get("is_human_intervention_needed"):
            return "human_intervention"
        if state.get("next_action") == "REPLAN":
            return "meta_planner"
        if state.get("current_sub_goal_id") is None:
            plan = state.get('plan')
            if plan and not find_next_executable_goal(plan):
                 return "synthesizer"
            else:
                # 如果 LLM 說完成了，但程式邏輯發現還有未完成的，就強制重新規劃
                logging.warning("--- 路由：執行官與程式邏輯判斷不一致，強制重新規劃！---")
                return "meta_planner"
        return "executor"
    workflow.add_conditional_edges("executive", route_from_executive)

    workflow.add_edge("executor", "reflector")

    def route_from_reflector(state: AgentState):
        if state.get("is_human_intervention_needed"):
            return "human_intervention"
        next_action = state.get("next_action")
        if next_action and next_action.upper() == "REPLAN": # pyright: ignore[reportOptionalMemberAccess]
            return "meta_planner"
        return "executive" # CONTINUE
    workflow.add_conditional_edges("reflector", route_from_reflector)

    workflow.add_edge("synthesizer", END)
    workflow.add_edge("human_intervention", END)
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
    # After_planner_state = {'plan': HierarchicalPlan(main_goal='找出鐵達尼號的導演是誰', 
    #                                                 sub_goals=[SubGoal(goal_id=1, description='使用 tavily_search 搜尋“鐵達尼號的導演”', dependencies=[], status='PENDING', raw_result=None, result_summary=''), 
    #                                                            SubGoal(goal_id=2, description='總結搜尋結果並回答導演是誰', dependencies=[1], status='PENDING', raw_result=None, result_summary='')]), 
    #                                                            'is_human_intervention_needed': False,
    #                                                            "main_goal": "鐵達尼號的導演是誰",}
    # print(Actions.executive_node(After_planner_state))
    # After_executive_state = {'plan': HierarchicalPlan(main_goal='找出鐵達尼號的導演是誰', 
    #                                                 sub_goals=[SubGoal(goal_id=1, description='使用 tavily_search 搜尋“鐵達尼號的導演”', dependencies=[], status='PENDING', raw_result=None, result_summary=''), 
    #                                                            SubGoal(goal_id=2, description='總結搜尋結果並回答導演是誰', dependencies=[1], status='PENDING', raw_result=None, result_summary='')]), 
    #                                                            'is_human_intervention_needed': False,
    #                                                            "main_goal": "鐵達尼號的導演是誰",
    #                                                            'current_sub_goal_id': 1,}
    # print(Actions.execute_subgraph_node(After_executive_state))
    # After_execute_state = {'plan': HierarchicalPlan(main_goal='找出鐵達尼號的導演是誰', 
    #                                                 sub_goals=[SubGoal(goal_id=1, description='使用 tavily_search 搜尋“鐵達尼號的導演”', dependencies=[], status='PENDING', raw_result=None, result_summary=''), 
    #                                                            SubGoal(goal_id=2, description='總結搜尋結果並回答導演是誰', dependencies=[1], status='PENDING', raw_result=None, result_summary='')]), 
    #                                                            'is_human_intervention_needed': False,
    #                                                            "main_goal": "鐵達尼號的導演是誰",
    #                                                            'current_sub_goal_id': 1,
    #                                                            'sub_task_raw_result': {'query': '使用 tavily_search 搜尋“鐵達尼號的導演”', 
    #                                                                                    'follow_up_questions': None, 
    #                                                                                    'answer': None, 
    #                                                                                    'images': [], 
    #                                                                                    'results': [{'url': 'https://zh.wikipedia.org/zh-tw/%E6%B3%B0%E5%9D%A6%E5%B0%BC%E5%85%8B%E5%8F%B7_(1997%E5%B9%B4%E7%94%B5%E5%BD%B1)', 
    #                                                                                                 'title': '鐵達尼號(1997年電影) - 維基百科', 
    #                                                                                                 'content': '《鐵達尼號》（英語：Titanic）是一部於1997年上映的美國史詩浪漫災難電影，由詹姆士·卡麥隆創作、導演、監製、共同製作及共同編輯。電影部分情節是根據1912年4月14日至15', 
    #                                                                                                 'score': 0.6227582, 
    #                                                                                                 'raw_content': None},
    #                                                                                                 {'url': 'https://www.threads.com/@daniel.moviegoer/post/DGsj2UbzHd-/6-titanic%E9%90%B5%E9%81%94%E5%B0%BC%E8%99%9F%E5%B0%8E%E6%BC%94james-cameron-19977-schindlers-list-%E8%BE%9B%E5%BE%B7%E5%8B%92%E7%9A%84%E5%90%8D%E5%96%AE-%E5%B0%8E%E6%BC%94steven-spielberg-199', 
    #                                                                                                  'title': '6. 《Titanic鐵達尼號》（導演：James Cameron, 1997) 7. ...', 
    #                                                                                                  'content': "6. 《Titanic鐵達尼號》（導演：James Cameron, 1997) 7. 《Schindler's List 辛德勒的名單》 (導演：Steven Spielberg, 1993) 8. 《The", 
    #                                                                                                  'score': 0.6150191, 
    #                                                                                                  'raw_content': None}, 
    #                                                                                                  {'url': 'https://tw.news.yahoo.com/%E3%80%8C%E8%AD%A6%E5%91%8A%E9%83%BD%E4%B8%8D%E8%81%BD%E3%80%8D%E5%90%8D%E5%B0%8E%E8%A9%B9%E5%A7%86%E6%96%AF%E5%8D%A1%E9%BA%A5%E9%9A%86%E6%AD%8E%EF%BC%9A%E6%B3%B0%E5%9D%A6%E8%99%9F%E6%82%B2%E5%8A%87%E5%A6%82%E9%90%B5%E9%81%94%E5%B0%BC%E8%99%9F%E4%BA%8B%E4%BB%B6%E9%87%8D%E6%BC%94-024027016.html', 
    #                                                                                                   'title': '「警告都不聽」名導詹姆斯‧卡麥隆歎：泰坦號悲劇如鐵達尼號 ...', 
    #                                                                                                   'content': '電影《鐵達尼號》（Titanic）導演兼深海探險家詹姆斯‧ 卡麥隆說，關於這具旅遊潛水器的許多安全警告均遭忽視，並感歎泰坦號悲劇如鐵達尼事件重演。泰坦號', 
    #                                                                                                   'score': 0.55072874, 
    #                                                                                                   'raw_content': None}], 
    #                                                                                     'response_time': 0.77, 
    #                                                                                     'request_id': 'dca737c5-588e-4db2-aa13-94c34abcf495'},
    #                                                                                     }
    # print(Actions.reflection_node(After_execute_state))
    # After_reflection_state = {'plan': HierarchicalPlan(main_goal='找出鐵達尼號的導演是誰', 
    #                                                     sub_goals=[SubGoal(goal_id=1, 
    #                                                                         description='使用tavily_search 搜尋“鐵達尼號的導演”', 
    #                                                                         dependencies=[], 
    #                                                                         status='completed', 
    #                                                                         raw_result={'query': '使用 tavily_search 搜尋“鐵達尼號的導演”', 
    #                                                                                     'follow_up_questions': None, 
    #                                                                                     'answer': None, 
    #                                                                                     'images': [], 
    #                                                                                     'results': [{'url': 'https://zh.wikipedia.org/zh-tw/%E6%B3%B0%E5%9D%A6%E5%B0%BC%E5%85%8B%E5%8F%B7_(1997%E5%B9%B4%E7%94%B5%E5%BD%B1)', 
    #                                                                                                 'title': '鐵達尼號(1997年電影) - 維基百科', 
    #                                                                                                 'content': '《鐵達尼號》（英語：Titanic）是一部於1997年上映的美國史詩浪漫災難電影，由詹姆士·卡麥隆創作、導演、監製、共同製作及共同編輯。電影部分情節是根據1912年4月14日至15', 
    #                                                                                                 'score': 0.6227582, 
    #                                                                                                 'raw_content': None}, 
    #                                                                                                 {'url': 'https://www.threads.com/@daniel.moviegoer/post/DGsj2UbzHd-/6-titanic%E9%90%B5%E9%81%94%E5%B0%BC%E8%99%9F%E5%B0%8E%E6%BC%94james-cameron-19977-schindlers-list-%E8%BE%9B%E5%BE%B7%E5%8B%92%E7%9A%84%E5%90%8D%E5%96%AE-%E5%B0%8E%E6%BC%94steven-spielberg-199', 
    #                                                                                                 'title': '6. 《Titanic鐵達尼號》（導演：James Cameron, 1997) 7. ...', 
    #                                                                                                 'content': "6. 《Titanic鐵達尼號》（導演：James Cameron, 1997) 7. 《Schindler's List 辛德勒的名單》 (導演：Steven Spielberg, 1993) 8. 《The", 
    #                                                                                                 'score': 0.6150191, 
    #                                                                                                 'raw_content': None}, 
    #                                                                                                 {'url': 'https://tw.news.yahoo.com/%E3%80%8C%E8%AD%A6%E5%91%8A%E9%83%BD%E4%B8%8D%E8%81%BD%E3%80%8D%E5%90%8D%E5%B0%8E%E8%A9%B9%E5%A7%86%E6%96%AF%E5%8D%A1%E9%BA%A5%E9%9A%86%E6%AD%8E%EF%BC%9A%E6%B3%B0%E5%9D%A6%E8%99%9F%E6%82%B2%E5%8A%87%E5%A6%82%E9%90%B5%E9%81%94%E5%B0%BC%E8%99%9F%E4%BA%8B%E4%BB%B6%E9%87%8D%E6%BC%94-024027016.html', 
    #                                                                                                 'title': '「警告都不聽」名導詹姆斯‧卡麥隆歎：泰坦號悲劇如鐵達尼號 ...', 
    #                                                                                                 'content': '電影《鐵達尼號》（Titanic）導演兼深海探險家詹姆斯‧ 卡麥隆說，關於這具旅遊潛水器的許多安全警告均遭忽視，並感歎泰坦號悲劇如鐵達尼事件重演。泰坦號', 
    #                                                                                                 'score': 0.55072874, 
    #                                                                                                 'raw_content': None}], 
    #                                                                                                 'response_time': 0.77, 
    #                                                                                                 'request_id': 'dca737c5-588e-4db2-aa13-94c34abcf495'}, 
    #                                                                         result_summary='Successfully found that the director of Titanic is James Cameron.'), 
    #                                                                 SubGoal(goal_id=2, 
    #                                                                         description='總結搜尋結果並回答導演是誰', 
    #                                                                         dependencies=[1], 
    #                                                                         status='PENDING', 
    #                                                                         raw_result=None, 
    #                                                                         result_summary='')]), 
    #                             'working_memory': {'goal_1_summary': 'Successfully found that the director of Titanic is James Cameron.'}, 
    #                             'next_action': 'CONTINUE', 
    #                             'replan_count': 0, 
    #                             'sub_task_raw_result': None,
    #                             'is_human_intervention_needed': False,
    #                             "main_goal": "鐵達尼號的導演是誰",
    #                             'current_sub_goal_id': 1
    #                             }
    # print(Actions.executive_node(After_reflection_state))
    # Actions.executive_node(After_reflection_state)
    # After_executive2_state = {'plan': HierarchicalPlan(main_goal='找出鐵達尼號的導演是誰', 
    #                                                     sub_goals=[SubGoal(goal_id=1, 
    #                                                                         description='使用tavily_search 搜尋“鐵達尼號的導演”', 
    #                                                                         dependencies=[], 
    #                                                                         status='completed', 
    #                                                                         raw_result={'query': '使用 tavily_search 搜尋“鐵達尼號的導演”', 
    #                                                                                     'follow_up_questions': None, 
    #                                                                                     'answer': None, 
    #                                                                                     'images': [], 
    #                                                                                     'results': [{'url': 'https://zh.wikipedia.org/zh-tw/%E6%B3%B0%E5%9D%A6%E5%B0%BC%E5%85%8B%E5%8F%B7_(1997%E5%B9%B4%E7%94%B5%E5%BD%B1)', 
    #                                                                                                 'title': '鐵達尼號(1997年電影) - 維基百科', 
    #                                                                                                 'content': '《鐵達尼號》（英語：Titanic）是一部於1997年上映的美國史詩浪漫災難電影，由詹姆士·卡麥隆創作、導演、監製、共同製作及共同編輯。電影部分情節是根據1912年4月14日至15', 
    #                                                                                                 'score': 0.6227582, 
    #                                                                                                 'raw_content': None}, 
    #                                                                                                 {'url': 'https://www.threads.com/@daniel.moviegoer/post/DGsj2UbzHd-/6-titanic%E9%90%B5%E9%81%94%E5%B0%BC%E8%99%9F%E5%B0%8E%E6%BC%94james-cameron-19977-schindlers-list-%E8%BE%9B%E5%BE%B7%E5%8B%92%E7%9A%84%E5%90%8D%E5%96%AE-%E5%B0%8E%E6%BC%94steven-spielberg-199', 
    #                                                                                                 'title': '6. 《Titanic鐵達尼號》（導演：James Cameron, 1997) 7. ...', 
    #                                                                                                 'content': "6. 《Titanic鐵達尼號》（導演：James Cameron, 1997) 7. 《Schindler's List 辛德勒的名單》 (導演：Steven Spielberg, 1993) 8. 《The", 
    #                                                                                                 'score': 0.6150191, 
    #                                                                                                 'raw_content': None}, 
    #                                                                                                 {'url': 'https://tw.news.yahoo.com/%E3%80%8C%E8%AD%A6%E5%91%8A%E9%83%BD%E4%B8%8D%E8%81%BD%E3%80%8D%E5%90%8D%E5%B0%8E%E8%A9%B9%E5%A7%86%E6%96%AF%E5%8D%A1%E9%BA%A5%E9%9A%86%E6%AD%8E%EF%BC%9A%E6%B3%B0%E5%9D%A6%E8%99%9F%E6%82%B2%E5%8A%87%E5%A6%82%E9%90%B5%E9%81%94%E5%B0%BC%E8%99%9F%E4%BA%8B%E4%BB%B6%E9%87%8D%E6%BC%94-024027016.html', 
    #                                                                                                 'title': '「警告都不聽」名導詹姆斯‧卡麥隆歎：泰坦號悲劇如鐵達尼號 ...', 
    #                                                                                                 'content': '電影《鐵達尼號》（Titanic）導演兼深海探險家詹姆斯‧ 卡麥隆說，關於這具旅遊潛水器的許多安全警告均遭忽視，並感歎泰坦號悲劇如鐵達尼事件重演。泰坦號', 
    #                                                                                                 'score': 0.55072874, 
    #                                                                                                 'raw_content': None}], 
    #                                                                                                 'response_time': 0.77, 
    #                                                                                                 'request_id': 'dca737c5-588e-4db2-aa13-94c34abcf495'}, 
    #                                                                         result_summary='Successfully found that the director of Titanic is James Cameron.'), 
    #                                                                 SubGoal(goal_id=2, 
    #                                                                         description='總結搜尋結果並回答導演是誰', 
    #                                                                         dependencies=[1], 
    #                                                                         status='PENDING', 
    #                                                                         raw_result=None, 
    #                                                                         result_summary='')]), 
    #                             'working_memory': {'goal_1_summary': 'Successfully found that the director of Titanic is James Cameron.'}, 
    #                             'next_action': 'CONTINUE', 
    #                             'replan_count': 0, 
    #                             'sub_task_raw_result': None,
    #                             'is_human_intervention_needed': False,
    #                             "main_goal": "鐵達尼號的導演是誰",
    #                             'current_sub_goal_id': 2
    #                             }
    # print(Actions.execute_subgraph_node(After_executive2_state))



    

   
