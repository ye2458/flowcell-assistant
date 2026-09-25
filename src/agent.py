"""
LangGraph 에이전트.

[전체 흐름]
  질문 → [agent 노드] LLM이 "도구를 부를지 / 바로 답할지" 결정
                ├─ 도구 호출이 필요하면 → [tools 노드] 실행 → 다시 agent 노드
                └─ 더 부를 도구가 없으면 → 종료(최종 답변)

이 왕복 구조를 ReAct 루프라고 부릅니다. 왜 루프인가?
  "conf 0.05를 왜 골랐고, 0.2로 올리면 어떻게 되나?" 같은 질문은
  문서 검색(도구 A) → 시뮬레이션(도구 C) 두 번을 거쳐야 답이 나옵니다.
  한 번만 부르고 끝내면 이런 질문에서 반드시 틀립니다.

[ABSA 경험의 이식]
  ABSA에서는 Supervisor/Analyzer/Critic 으로 나누고 reason_code 로 실패를 분류했습니다.
  여기서는 역할을 나누는 대신 '도구 선택'과 '실패 분류'에 같은 발상을 씁니다.
  → 팀 프로젝트의 설계를 새 문제에 옮겼다는 것이 면접에서 쓸 서사입니다.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src import config
from src.tools import TOOLS

# ★ 이 프롬프트의 규칙 순서는 평가 결과를 보고 고친 것입니다(경위는 아래 주석).
SYSTEM_PROMPT = """당신은 이차전지 셀 결함검사 시스템의 운영 어시스턴트입니다.

규칙:
- 이 시스템의 문서·검사 결과에 관한 질문이면 **먼저 도구를 사용하세요.**
  도구를 부르기 전에 "문서에 없다"고 단정하지 마세요. 검색해 보기 전에는 알 수 없습니다.
- 숫자를 세거나 계산해야 하는 질문은 추측하지 말고 반드시 도구를 사용하세요.
- 출처는 **문서 검색 결과에만** 붙이세요. 도구가 계산한 수치(recall, 재검 부하 등)는
  문서에서 인용한 것이 아니므로 출처를 붙이지 마세요.
- 다음은 도구로도 알 수 없으므로 **도구를 부르지 말고** 모른다고 답하세요.
  (1) 미래 예측  (2) 이 시스템 외부의 정보(경쟁사 등)  (3) 다른 도메인에의 적용 판단
- 지어내지 마세요.
"""

# [프롬프트 수정 경위 — 2026-09-25, 평가 1차 결과]
#   증상: "TensorRT FP16 효과가 1.15배에 그친 이유"(rag #5)에서 도구를 부르지 않고 거절.
#         그런데 답은 04_deployment_spec.md:16 에 실제로 적혀 있었습니다.
#   원인: 거절 규칙("모르면 모른다고 답하라")이 검색 규칙보다 먼저 적용됐습니다.
#         LLM은 규칙의 '순서'에 영향을 받으므로, 거절을 앞에 두면 과잉 거절이 납니다.
#   조치: 검색 규칙을 맨 위로 올리고, 거절 대상을 3가지로 '열거'했습니다.
#         "모르는 것"처럼 모호하게 두면 LLM이 범위를 스스로 넓힙니다.
#   ★ 이 수정의 위험: 거절 범위를 좁히면 refuse 3문항이 깨질 수 있습니다.
#     그래서 고친 뒤 평가를 다시 돌려 rag 와 refuse 를 함께 확인했습니다(README 참조).


class State(TypedDict):
    # add_messages: 새 메시지를 리스트에 '누적'하라는 표시.
    # 이게 없으면 매 노드마다 메시지가 덮어써져 대화 맥락이 사라집니다.
    messages: Annotated[list[AnyMessage], add_messages]


def build_agent():
    llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0)
    # bind_tools: LLM에게 "이런 도구들이 있다"고 알려줍니다.
    # 그러면 LLM이 답변 대신 '도구를 부르겠다'는 응답을 낼 수 있게 됩니다(Function Calling).
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent_node(state: State):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}

    def should_continue(state: State) -> str:
        """마지막 메시지에 도구 호출이 붙어 있으면 tools 노드로, 아니면 종료."""
        # [흐름] agent 노드가 방금 만든 메시지 1개를 보고 '다음에 갈 노드 이름'만 돌려줍니다.
        #        이 함수는 아무것도 실행하지 않습니다. 문자열 하나를 고르는 게 전부입니다.
        last_message = state["messages"][-1]

        # ★ 루프를 도는 조건 전체가 이 한 줄에 들어 있습니다.
        #   LLM이 도구를 부르기로 했으면 tool_calls 에 목록이 채워지고,
        #   바로 답하기로 했으면 빈 리스트([])가 됩니다.
        #   빈 리스트는 파이썬에서 False 로 취급되므로 if 하나로 갈립니다.
        #
        #   getattr(..., []) 를 쓴 이유: 마지막 메시지가 항상 AIMessage 인 것은 아닙니다.
        #   tool_calls 속성이 없는 메시지 타입이 오면 last_message.tool_calls 는
        #   AttributeError 로 죽습니다. 기본값 [] 로 '도구 없음'으로 안전하게 처리합니다.
        if getattr(last_message, "tool_calls", []):
            return "tools"

        # 도구 호출이 없다 = LLM이 최종 답변을 썼다 = 그래프를 끝낼 때.
        return END

    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")  # 도구 실행 후 다시 agent 로 (= 루프)

    return graph.compile()


_agent = None


def ask(question: str) -> dict:
    """질문 하나를 처리하고, 답변과 '실제로 호출된 도구 목록'을 함께 반환합니다.

    ★ 호출된 도구를 같이 반환하는 이유:
      평가셋에서 '도구 선택 정확도'를 채점하려면 어떤 도구를 골랐는지 알아야 합니다.
      답변 텍스트만 봐서는 우연히 맞은 것인지 구분할 수 없습니다.
    """
    global _agent
    if _agent is None:
        _agent = build_agent()

    result = _agent.invoke({"messages": [("user", question)]})

    used_tools = []
    for m in result["messages"]:
        for call in getattr(m, "tool_calls", []) or []:
            used_tools.append(call["name"])

    return {
        "answer": result["messages"][-1].content,
        "used_tools": used_tools,
    }


if __name__ == "__main__":
    out = ask("conf 임계값을 0.05로 정한 이유가 뭔가요?")
    print(out["answer"])
    print("사용한 도구:", out["used_tools"])
