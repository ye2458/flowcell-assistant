"""
평가 실행기. ★ 이 프로젝트에서 가장 값이 나가는 산출물입니다.

실행:  python -m eval.run_eval

[왜 평가셋을 만드는가]
"LLM 프로젝트를 해봤다"는 지원자는 많고, "LLM 시스템을 평가했다"는 지원자는 거의 없습니다.
챗봇은 아무 질문에나 그럴듯한 문장을 내놓기 때문에, 눈으로 몇 번 보고
'잘 되는 것 같다'고 판단하면 반드시 틀립니다.

[무엇을 자동으로 채점하고, 무엇을 손으로 채점하는가]
- 자동: **도구 선택 정확도**. 기대한 도구를 실제로 불렀는가.
        답변 텍스트만 보면 우연히 맞은 것과 구분할 수 없으므로 이 지표가 필요합니다.
- 수동: **답변 정확도 / 환각 여부**. 결과 CSV의 빈 칸을 직접 채우세요.
        20문항이면 15분이면 됩니다. LLM에게 채점을 맡기면 채점기를 또 평가해야 합니다.

[한계 — 면접에서 먼저 밝힐 것]
20문항은 통계적 유의성을 주장할 표본이 아닙니다.
설정을 바꿨을 때 망가진 곳을 찾는 **회귀 테스트** 용도입니다.
"성능을 증명했다"가 아니라 "무너지는 지점을 감시했다"가 정확한 표현입니다.
"""

import csv
import json
from datetime import datetime

from src import config
from src.agent import ask

QUESTIONS_PATH = config.EVAL_DIR / "questions.jsonl"


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

        ok = judge_tool(q["expected_tool"], used)
        print(f"[{q['id']:>2}] {q['category']:<10} tool={'O' if ok else 'X'}  used={used}")

        rows.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "expected_tool": q["expected_tool"] or "(none)",
            "used_tools": "|".join(used),
            "tool_correct": int(ok),
            "answer": answer.replace("\n", " ")[:400],
            "error": error,
            # ↓ 아래 3개는 사람이 채웁니다. LLM에게 채점을 맡기면 채점기를 또 평가해야 합니다.
            "answer_correct_MANUAL": "",   # 답변 내용이 맞는가 (1/0)
            "hallucination_MANUAL": "",    # 문서·표에 없는 내용을 지어냈는가 (1/0)
            # source_correct: 출처를 맞게 붙였는가 (1/0, 출처가 필요 없는 문항은 공란)
            # ★ 이 컬럼은 1차 평가에서 '출처 오귀속'이 반복 관측돼 추가했습니다.
            #   내용은 맞는데 출처 파일명만 틀린 경우가 있었고, 위 두 컬럼으로는
            #   그게 잡히지 않았습니다(내용 정답=1, 환각=0 인데 인용이 틀림).
            #   지표는 관측된 실패 유형에 맞춰 늘려야 합니다.
            "source_correct_MANUAL": "",
        })

    stamp = datetime.now().strftime("%m%d_%H%M")
    out = config.EVAL_DIR / f"results_{stamp}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ── 요약: 카테고리별 도구 선택 정확도 ──
    print("\n── 도구 선택 정확도 ──")
    for cat in ["rag", "stats", "simulation", "refuse"]:
        sub = [r for r in rows if r["category"] == cat]
        if sub:
            acc = sum(r["tool_correct"] for r in sub) / len(sub)
            print(f"  {cat:<11} {acc:>6.0%}  ({sum(r['tool_correct'] for r in sub)}/{len(sub)})")
    total = sum(r["tool_correct"] for r in rows) / len(rows)
    print(f"  {'전체':<11} {total:>6.0%}")
    print(f"\n결과 저장 → {out}")
    print("이제 CSV를 열어 answer_correct_MANUAL / hallucination_MANUAL 을 채우세요.")


if __name__ == "__main__":
    main()
