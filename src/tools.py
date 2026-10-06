"""
에이전트가 쓸 도구 정의.

★ 이 파일이 이 프로젝트의 핵심입니다.
면접에서 "왜 도구를 이렇게 나눴습니까"를 물으면 여기 설계로 답하게 됩니다.

────────────────────────────────────────────────────────
[도구 경계 설계 — 이 프로젝트의 논증]

도구 A: search_inspection_docs  → 문서에서 '근거/판단 이유'를 찾는다
도구 B: query_defect_stats      → 표에서 '숫자를 센다'
도구 C: simulate_threshold      → 임계값을 바꿨을 때 '숫자를 다시 계산한다'

왜 하나로 합치지 않는가?
  "confidence 0.3 이하인 REJECT 셀이 몇 개인가?" 를 벡터 검색으로 풀면 반드시 틀립니다.
  벡터 검색은 '의미가 비슷한 문장'을 찾을 뿐, 세거나 더하지 못합니다.
  숫자를 세는 일은 결정적(deterministic)이어야 하고, 그건 코드가 할 일입니다.
  LLM에게는 '어떤 도구를 부를지 고르는 판단'만 맡기고, 계산은 코드가 합니다.

  → 이것이 RAG 챗봇과 Agent의 차이입니다.
    RAG는 검색해서 요약하고, Agent는 무엇을 할지 고르고 실행합니다.

[공통 설계 원칙 — 세 도구가 전부 str 을 반환하는 이유]
  반환값을 읽는 것은 사람이 아니라 LLM입니다. LLM은 dict 나 DataFrame 을 못 읽고
  결국 문자열로 바꿔서 봅니다. 그렇다면 그 변환을 LLM에게 맡기지 말고
  우리가 직접 통제해야 합니다. "무엇을 보여줄지"를 도구가 정하면
  답변 품질이 재현 가능해지고, 틀렸을 때 원인을 도구 하나로 좁힐 수 있습니다.
────────────────────────────────────────────────────────
"""

import pandas as pd
from langchain_core.tools import tool

from src import config
from src.ingest import get_vectorstore

# 표 데이터는 한 번만 읽어 메모리에 올려둡니다(수백 행 규모라 충분).
_df: pd.DataFrame | None = None


def _results() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(config.RESULTS_CSV)
    return _df


# 벡터DB 핸들도 한 번만 열어 재사용합니다.
_store = None


def _vectorstore():
    """열려 있는 벡터DB를 돌려줍니다. 처음 호출될 때 한 번만 엽니다.

    ★ 왜 캐시하는가 (2026-10-06 수정)
      전에는 검색할 때마다 get_vectorstore() 를 불렀습니다. 그 함수는 호출될 때마다
      OpenAIEmbeddings 객체와 Chroma 클라이언트를 **새로 만듭니다.**
      기능은 맞지만 질문 한 번에 DB를 다시 여는 셈이라, 서버로 띄워 두고
      여러 요청을 받을수록 손해가 쌓입니다.

      표(_results)를 이미 같은 방식으로 캐시하고 있었으므로, 읽기 전용 자원은
      같은 규칙으로 다루는 편이 코드가 읽기 쉽습니다.

      주의: 색인을 다시 만들면(python -m src.ingest) 이 프로세스는 옛 핸들을 들고
      있을 수 있습니다. 색인은 별도 프로세스로 돌리므로 실제로는 문제가 없지만,
      서버를 띄운 채 재색인했다면 서버를 다시 시작해야 합니다.
    """
    global _store
    if _store is None:
        _store = get_vectorstore()
    return _store


# ★ 계산 결과임을 도구 출력 자체에 박아 넣는 표시.
#
# [왜 프롬프트가 아니라 여기인가 — 두 번 배운 교훈]
# 평가 중 LLM이 simulate_threshold 의 '계산값'에까지
# "[출처: 04_deployment_spec.md]" 를 붙이는 출처 오귀속이 반복됐습니다.
# 시스템 프롬프트에 "계산 결과에는 출처를 붙이지 말 것"을 명시했으나 고쳐지지 않았습니다.
# (도구 선택 문제를 프롬프트로 고치려다 실패한 것과 같은 패턴입니다.)
#
# LLM은 컨텍스트에 '있는 것'에 훨씬 강하게 반응합니다. 그래서 규칙을 프롬프트에 두는 대신
# 도구가 돌려주는 문자열의 첫 줄에 넣었습니다. 지켜야 할 제약은 지시가 아니라
# 데이터로 주는 편이 안정적입니다.
_NO_SOURCE_NOTE = "[계산 결과 — 표에서 직접 계산한 값이므로 문서 출처가 없습니다. 출처를 붙이지 마세요.]"


# ══════════════════════════════════════════════════════
# 도구 A — 문서 검색 (RAG)
# ══════════════════════════════════════════════════════
@tool
def search_inspection_docs(query: str) -> str:
    """이 프로젝트의 내부 기록에서 '판단 근거'를 찾습니다. 아래 주제가 전부 들어 있으므로,
    해당하는 질문이면 답을 모른다고 판단하기 전에 **반드시 먼저 이 도구를 호출하세요.**

    - 평가 프로토콜: 평가 분할, 지표 정의, 지표 함정 감사 절차
    - 실험 이력: CT / EXT 파이프라인, 두 파이프라인을 분리한 이유, SAHI 타일 분할
    - 라벨 품질: 오라벨 셀 발견 경위, 제외 후 재측정 결과
    - 배포 사양: 처리 시간 실측, CPU/GPU 병목 분석, **TensorRT FP16 등 최적화 시도**,
      인스턴스 선정, 운영점(conf threshold) 결정 근거, 파이프라인 5단계

    수치를 세거나 계산하는 질문에는 사용하지 마세요(그건 다른 두 도구의 일입니다).

    Args:
        query: 찾고자 하는 내용 (예: "conf 임계값을 0.05로 정한 이유")
    """
    # [흐름] 질문(str) → 벡터DB 검색 → Document 리스트 → 출처가 붙은 문자열 1개
    documents = _vectorstore().similarity_search(query, k=config.TOP_K)

    # 검색 결과가 0건일 때 빈 문자열을 돌려주지 않습니다.
    # 빈 문자열을 받은 LLM은 '근거가 없다'가 아니라 '도구가 고장났다'고 해석하거나,
    # 최악의 경우 자기 지식으로 답을 지어냅니다. 없으면 없다고 말로 알려줘야 합니다.
    if not documents:
        return "관련 문서를 찾지 못했습니다."

    # ★ 출처를 본문과 함께 묶는 지점.
    #   출처가 없으면 (1) 답변이 환각인지 검증할 수 없고
    #   (2) eval 단계에서 '근거 있음/없음'을 채점할 수 없습니다.
    #   출처는 LLM의 편의가 아니라 '검증 가능성'을 위해 붙입니다.
    passages = [
        f"[출처: {doc.metadata['source']}]\n{doc.page_content}"
        for doc in documents
    ]

    # 청크 사이를 빈 줄로 벌립니다. 경계가 흐릿하면 LLM이 두 청크의 내용을 섞어
    # 엉뚱한 파일을 출처로 붙입니다(실제로 자주 나는 오류입니다).
    return "\n\n".join(passages)


# ══════════════════════════════════════════════════════
# 도구 B — 통계 조회
# ══════════════════════════════════════════════════════
@tool
def query_defect_stats(
    verdict: str | None = None,
    defect_type: str | None = None,
    max_confidence: float | None = None,
) -> str:
    """검사 결과 표에서 조건에 맞는 셀의 개수와 요약 통계를 반환합니다.
    '몇 개', '평균', '분포' 처럼 세거나 집계하는 질문에 사용하세요.

    Args:
        verdict: "PASS" 또는 "REJECT" 로 필터
        defect_type: 결함 종류로 필터 (예: "Pollution", "Damaged")
        max_confidence: 이 값 이하의 confidence 만 필터
    """
    # [흐름] 인자 0~3개 → 조건에 맞는 행만 남은 DataFrame → 사람이 읽는 요약 문자열
    df = _results()

    # ★ 인자를 전부 Optional 로 두고 '들어온 것만' 필터를 겁니다.
    #   필수 인자로 만들면 LLM이 모르는 값을 지어내서 채웁니다(환각의 흔한 원인).
    #
    #   applied_filters 는 '어떤 조건으로 센 결과인지'를 답변에 되돌려주기 위한 것입니다.
    #   이게 없으면 LLM이 조건을 빼먹고 "43건입니다"라고만 답해서,
    #   사용자는 무엇을 센 숫자인지 확인할 수 없게 됩니다.
    applied_filters: list[str] = []

    if verdict is not None:
        # .upper() : "reject" 처럼 소문자로 들어와도 동작하게. LLM 출력은 일정하지 않습니다.
        df = df[df["true_label"] == verdict.upper()]
        applied_filters.append(f"판정={verdict.upper()}")

    if defect_type is not None:
        # ★ 대소문자를 무시하고 맞춥니다 (2026-10-06 버그 수정).
        #   표에는 "Pollution" 으로 저장돼 있는데, LLM 이 질문 문장을 그대로 옮겨
        #   "pollution" 을 넘기면 정확히 일치하는 행이 없어 **0건**이 나옵니다.
        #   문제는 에러가 아니라 "0건입니다" 라는 **그럴듯한 오답**이 나간다는 점입니다.
        #   verdict 는 .upper() 로 막아뒀는데 여기만 빠져 있어 동작이 비일관했습니다.
        #
        #   casefold() 를 쓴 이유: lower() 보다 넓은 범위를 정규화합니다.
        #   defect_type 이 비어 있는 행(PASS 셀)은 NaN 이라 비교 결과가 False 가 되므로
        #   따로 걸러낼 필요가 없습니다.
        matches = df["defect_type"].str.casefold() == defect_type.casefold()
        found = df.loc[matches, "defect_type"]
        # 표에 실제로 적힌 철자를 되돌려줍니다. 사용자가 무엇으로 센 것인지 분명해집니다.
        canonical = found.iloc[0] if len(found) else defect_type
        df = df[matches]
        applied_filters.append(f"결함종류={canonical}")

    if max_confidence is not None:
        df = df[df["pred_score"] <= max_confidence]
        applied_filters.append(f"confidence<={max_confidence}")

    condition_text = ", ".join(applied_filters) if applied_filters else "조건 없음(전체)"

    # 0건일 때 평균을 구하면 NaN 이 나옵니다. NaN 을 그대로 문자열에 넣으면
    # LLM이 "평균은 nan입니다"라고 그대로 읽어버리므로 여기서 갈라줍니다.
    if len(df) == 0:
        return f"조건({condition_text})에 맞는 셀이 없습니다. (0건)"

    return (
        f"{_NO_SOURCE_NOTE}\n"
        f"조건: {condition_text}\n"
        f"건수: {len(df)}건\n"
        f"confidence 평균: {df['pred_score'].mean():.4f} "
        f"(최소 {df['pred_score'].min():.4f} / 최대 {df['pred_score'].max():.4f})"
    )


# ══════════════════════════════════════════════════════
# 도구 C — 운영점 시뮬레이션  ★ 이 프로젝트의 차별화 지점
# ══════════════════════════════════════════════════════
@tool
def simulate_threshold(confidence_threshold: float) -> str:
    """confidence 임계값을 바꿨을 때 recall 과 재검 부하가 어떻게 변하는지 계산합니다.
    '임계값을 0.2로 올리면 어떻게 되나' 같은 운영 판단 질문에 사용하세요.

    Args:
        confidence_threshold: 시뮬레이션할 임계값 (0.0 ~ 1.0)
    """
    # [흐름] 임계값(float) → 전체 1000행에 판정 규칙 적용 → 지표 계산 → 문자열
    df = _results()

    # 길이 1000짜리 불리언 시리즈 두 개를 만듭니다(True/False 가 행마다 하나씩).
    #   is_actual_defect : 정답 라벨이 REJECT 인가          → 임계값과 무관한 '사실'
    #   is_flagged       : 이 임계값에서 REJECT 로 판정되는가 → 임계값에 따라 변하는 '판단'
    # 이 둘을 분리해 둔 것이 이 함수의 뼈대입니다. 사실은 고정, 판단만 움직입니다.
    is_actual_defect = df["true_label"] == "REJECT"
    is_flagged = df["pred_score"] >= confidence_threshold

    # & 는 두 시리즈를 행 단위로 AND. 즉 '실제 불량이면서 잡아내기도 한' 행.
    caught_defects = int((is_actual_defect & is_flagged).sum())
    total_defects = int(is_actual_defect.sum())
    flagged_count = int(is_flagged.sum())

    recall = caught_defects / total_defects

    # ★ 재검 부하 = 사람이 다시 봐야 하는 셀의 비율. precision 을 쓰지 않았습니다.
    #   precision 의 분모에는 '정답'이 필요한데, 운영 현장에는 정답이 없습니다
    #   (정답을 안다면 애초에 검사가 필요 없습니다).
    #   반면 '오늘 몇 개를 재검 대기로 넘겼는가'는 라벨 없이 그 자리에서 셀 수 있고,
    #   그대로 인건비로 환산됩니다.
    #   → 운영 중에 실제로 계산 가능하고 비용에 직결되는 지표를 골랐습니다.
    review_load = flagged_count / len(df)

    missed_defects = total_defects - caught_defects

    return (
        f"{_NO_SOURCE_NOTE}\n"
        f"임계값 {confidence_threshold} 적용 결과\n"
        f"- Recall: {recall:.3f} (실제 불량 {total_defects}건 중 {caught_defects}건 검출)\n"
        f"- 재검 부하: {review_load:.1%} (전체 {len(df)}건 중 {flagged_count}건이 재검 대상)\n"
        f"- 놓친 불량: {missed_defects}건"
    )


# 에이전트에 넘길 도구 목록.
# ★ 3개를 넘기지 마세요. 도구가 늘수록 LLM의 선택 정확도가 떨어지고,
#   평가셋으로 원인을 추적하기도 어려워집니다.
TOOLS = [search_inspection_docs, query_defect_stats, simulate_threshold]
