"""
평가 실행기. ★ 이 프로젝트에서 가장 값이 나가는 산출물입니다.

실행:  python -m eval.run_eval

[왜 평가셋을 만드는가]
"LLM 프로젝트를 해봤다"는 지원자는 많고, "LLM 시스템을 평가했다"는 지원자는 거의 없습니다.
챗봇은 아무 질문에나 그럴듯한 문장을 내놓기 때문에, 눈으로 몇 번 보고
'잘 되는 것 같다'고 판단하면 반드시 틀립니다.

[무엇을 자동으로 채점하고, 무엇을 손으로 채점하는가]
- 자동: **도구 선택** / **출처** / **사실 근거**  ← 아래 세 judge_* 함수
- 수동: **답변 정확도 / 환각 여부**. 결과 CSV의 빈 칸을 직접 채웁니다.
        LLM에게 채점을 맡기면 채점기를 또 평가해야 합니다.

[2026-09-28 보강 — 왜 지표를 늘렸는가] ★
실사용 중 평가셋이 잡지 못한 실패를 발견했습니다.

    질문: "0.651→0.798은 재학습이 아니라 평가 데이터 정제인가?"
    답변: "...동분포와 홀드아웃 평가의 차이를 반영하며..." [출처: 01_evaluation_protocol.md]

인용된 01 문서에는 그 수치가 아예 없습니다. 즉 **출처를 잘못 붙이고 근거를 지어냈습니다.**
(실제 원인은 라벨 오류 셀 9개의 이미지를 **학습에서 빼고 재학습**한 것이고 평가셋은 그대로입니다.
 당시 03 문서가 이를 "평가에서 제외"로 잘못 적고 있었고, 2026-09-29 원자료 기준으로 정정했습니다.)

기존 20문항이 이것을 못 잡은 이유는 두 가지입니다.
 1) 질문이 전부 중립적으로 쓰여 있어서, **거짓 전제를 들이밀었을 때**를 시험하지 않았다.
 2) 채점이 '도구를 골랐는가'까지만 자동이어서, **출처가 맞는가**는 사람이 봐야 했다.

그래서 두 가지를 추가했습니다.
 - `leading` 카테고리 5문항 — 거짓 전제를 담은 유도 질문
 - `expected_source` / `must_contain` 자동 채점 — 출처와 사실 근거를 기계가 확인

**지표는 관측된 실패 유형에 맞춰 늘려야 합니다.** 평가셋도 한 번 만들고 끝나지 않습니다.

[한계 — 면접에서 먼저 밝힐 것]
25문항은 통계적 유의성을 주장할 표본이 아닙니다.
설정을 바꿨을 때 망가진 곳을 찾는 **회귀 테스트** 용도입니다.
"성능을 증명했다"가 아니라 "무너지는 지점을 감시했다"가 정확한 표현입니다.
또한 `must_contain` 은 문자열 포함 검사라 '의미가 맞는지'까지 보지 못합니다.
대신 LLM 채점기를 쓰지 않으므로 몇 번 돌려도 같은 값이 나옵니다.
"""

import csv
import json
from datetime import datetime

from src import config
from src.agent import ask

QUESTIONS_PATH = config.EVAL_DIR / "questions.jsonl"
CATEGORIES = ["rag", "stats", "simulation", "refuse", "leading"]


def load_questions() -> list[dict]:
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def judge_tool(expected: str | None, used: list[str]) -> bool:
    """도구 선택이 맞았는지 판정.

    expected 가 None(거절해야 하는 질문)이면 '도구를 하나도 안 불렀을 때' 정답입니다.
    데이터에 없는 것을 도구로 뒤지기 시작하면 환각으로 이어집니다.
    """
    if expected is None:
        return len(used) == 0
    return expected in used


# 코퍼스 파일 전체 목록. '엉뚱한 문서를 같이 인용했는지' 를 보려면 후보를 알아야 합니다.
CORPUS_FILES = [
    "01_evaluation_protocol.md",
    "02_experiment_log.md",
    "03_mislabel_issue.md",
    "04_deployment_spec.md",
]


def judge_source(expected: str | None, answer: str) -> bool | None:
    """답변이 **올바른 출처만** 인용했는지.

    ★ 이 검사가 위 '0.651' 사례를 잡습니다. 내용이 그럴듯해도 출처가 다른 파일이면
      근거를 확인할 수 없고, 읽는 사람을 엉뚱한 문서로 보냅니다.

    [2026-09-28 강화] 처음에는 `expected in answer` 만 봤는데, 그러면
    **맞는 출처와 틀린 출처를 같이 인용해도 통과**합니다. 실제로 21번 답변이
    `01` 과 `03` 을 나란히 인용했고(해당 수치는 `03` 에만 있음) 그대로 통과했습니다.
    → 기대 출처가 있어야 하고, **다른 코퍼스 문서가 섞이면 실패**로 봅니다.

    전제: 채점 대상 문항은 정답 출처가 하나입니다(현재 13문항 모두 해당).
          한 답에 여러 문서가 정당하게 필요한 질문을 추가하려면 이 함수를 손봐야 합니다.
    expected_source 가 없는 문항(stats/simulation/refuse)은 채점 대상이 아니므로 None.
    """
    if not expected:
        return None
    if expected not in answer:
        return False
    strays = [f for f in CORPUS_FILES if f != expected and f in answer]
    return not strays


def judge_facts(must_all: list[str] | None, must_any: list[str] | None, answer: str) -> bool | None:
    """답변에 **사실 앵커**가 들어 있는지.

    - `must_contain`     : 전부 있어야 함 (AND)
    - `must_contain_any` : 하나만 있으면 됨 (OR)

    앵커는 '그 질문의 정답 대목에만 등장하는 문자열'로 골랐습니다.
    예: 24번(FP16 3배냐?)의 앵커는 "1.1" 입니다. 거짓 전제에 동의하면
    이 문자열이 나올 수 없으므로, **유도 질문에 끌려갔는지**가 여기서 드러납니다.

    [2026-09-28 OR 추가 — 측정기 버그를 고친 것] ★
    22번 앵커를 "분리" 하나로 잡았더니, 시스템이 정확히
    "하나의 멀티모달 모델로 **묶지 않고** 두 개의 **독립** 파이프라인으로 설계"
    라고 옳게 답했는데도 실패로 찍혔습니다. **같은 뜻을 다른 낱말로 쓴 것**뿐입니다.
    T8에서 정답 문자열을 잘못 골라 검색을 의심했던 것과 **같은 실수의 재발**입니다.
    → 표현이 갈릴 수 있는 앵커는 OR 목록으로 두어, 측정기가 문체에 흔들리지 않게 합니다.
    """
    if must_all and not all(token in answer for token in must_all):
        return False
    if must_any and not any(token in answer for token in must_any):
        return False
    if not must_all and not must_any:
        return None
    return True


def _rate(rows: list[dict], key: str) -> str:
    """해당 컬럼에서 채점 대상인 행만 모아 비율 문자열을 만듭니다."""
    scored = [r for r in rows if r[key] != ""]
    if not scored:
        return "     -"
    hit = sum(int(r[key]) for r in scored)
    return f"{hit / len(scored):>5.0%} ({hit}/{len(scored)})"


def main() -> None:
    config.check()
    questions = load_questions()
    rows = []

    for q in questions:
        try:
            result = ask(q["question"])
            answer, used = result["answer"], result["used_tools"]
            error = ""
        except Exception as e:  # 도구 예외도 결과로 남깁니다. 조용히 넘어가면 원인을 못 찾습니다.
            answer, used, error = "", [], f"{type(e).__name__}: {e}"

        tool_ok = judge_tool(q["expected_tool"], used)
        src_ok = judge_source(q.get("expected_source"), answer)
        fact_ok = judge_facts(q.get("must_contain"), q.get("must_contain_any"), answer)

        mark = lambda v: "-" if v is None else ("O" if v else "X")
        print(
            f"[{q['id']:>2}] {q['category']:<10} "
            f"tool={mark(tool_ok)} src={mark(src_ok)} fact={mark(fact_ok)}  used={used}"
        )

        rows.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "expected_tool": q["expected_tool"] or "(none)",
            "used_tools": "|".join(used),
            "tool_correct": int(tool_ok),
            "expected_source": q.get("expected_source", ""),
            "source_correct": "" if src_ok is None else int(src_ok),
            "must_contain": "|".join(q.get("must_contain", []) + q.get("must_contain_any", [])),
            "facts_correct": "" if fact_ok is None else int(fact_ok),
            "false_premise": q.get("false_premise", ""),
            # 답변에서 실제로 인용된 코퍼스 파일. 아래 answer 는 400자에서 잘리기 때문에
            # 출처가 뒤쪽에 있으면 CSV 에서 보이지 않습니다(채점은 원문 전체로 합니다).
            # 사람이 눈으로 확인할 때 이 컬럼이 없으면 출처를 검증할 수 없습니다.
            "cited_sources": "|".join(f for f in CORPUS_FILES if f in answer),
            "answer": answer.replace("\n", " ")[:400],
            "error": error,
            # ↓ 아래 2개는 사람이 채웁니다. LLM 채점기를 쓰면 채점기를 또 평가해야 합니다.
            "answer_correct_MANUAL": "",   # 답변 내용이 맞는가 (1/0)
            "hallucination_MANUAL": "",    # 문서·표에 없는 내용을 지어냈는가 (1/0)
        })

    stamp = datetime.now().strftime("%m%d_%H%M")
    out = config.EVAL_DIR / f"results_{stamp}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ── 요약: 카테고리별 × 지표별 ──
    print(f"\n{'':<12}{'도구 선택':>13}{'출처':>13}{'사실 근거':>13}")
    print("─" * 51)
    for cat in CATEGORIES:
        sub = [r for r in rows if r["category"] == cat]
        if sub:
            print(f"{cat:<12}{_rate(sub, 'tool_correct'):>13}{_rate(sub, 'source_correct'):>13}"
                  f"{_rate(sub, 'facts_correct'):>13}")
    print("─" * 51)
    print(f"{'전체':<11}{_rate(rows, 'tool_correct'):>13}{_rate(rows, 'source_correct'):>13}"
          f"{_rate(rows, 'facts_correct'):>13}")

    # 실패한 문항은 따로 모아 보여줍니다. 요약표만 보면 '어디가' 깨졌는지 모릅니다.
    failed = [r for r in rows
              if not r["tool_correct"]
              or r["source_correct"] == 0
              or r["facts_correct"] == 0]
    if failed:
        print("\n── 실패 문항 ──")
        for r in failed:
            why = []
            if not r["tool_correct"]:
                why.append(f"도구({r['used_tools'] or '없음'} ≠ {r['expected_tool']})")
            if r["source_correct"] == 0:
                why.append(f"출처(기대 {r['expected_source']})")
            if r["facts_correct"] == 0:
                why.append(f"사실앵커({r['must_contain']} 누락)")
            print(f"  [{r['id']:>2}] {r['category']:<10} {' / '.join(why)}")
            print(f"       {r['question']}")

    print(f"\n결과 저장 → {out}")
    print("이제 CSV를 열어 answer_correct_MANUAL / hallucination_MANUAL 을 채우세요.")


if __name__ == "__main__":
    main()
