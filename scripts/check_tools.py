"""
도구 회귀 테스트.  실행: python -m scripts.check_tools

[왜 이 파일이 필요한가]
이 프로젝트의 수치 주장(conf 0.05 → recall 0.977 / 재검 부하 6.6% / 놓친 불량 1건)은
전부 `simulate_threshold` 와 `query_defect_stats` 가 계산한 값입니다.
그런데 그 계산을 검증하는 유일한 수단이 `eval/run_eval.py` 뿐이었습니다.

평가는 **LLM API 를 25번 호출**합니다. 돈이 들고, 몇 분 걸리고, 네트워크가 필요합니다.
그래서 자주 못 돌립니다. 자주 못 도는 검사는 **없는 것과 비슷합니다.**

이 파일은 **API 를 한 번도 부르지 않습니다.** 표를 읽어 계산하는 두 도구만 확인하므로
몇 초면 끝나고, 공짜고, 비행기 안에서도 돕니다. 계산식을 건드린 뒤 이것부터 돌리면
평가를 돌리기 전에 틀린 것을 잡을 수 있습니다.

[왜 pytest 를 쓰지 않는가]
확인할 것이 10여 개뿐이라 프레임워크가 주는 이득이 없습니다. 의존성을 하나 늘리면
설치 문서와 CI 설정이 따라옵니다. `assert` 로 충분하고, 실패하면 어느 줄인지 그대로 나옵니다.

[무엇을 확인하지 않는가 — 솔직하게]
`search_inspection_docs` 는 임베딩 API 가 필요하므로 여기서 빠졌습니다.
검색 품질은 `python -m eval.retrieval_check` 가 따로 봅니다.
"""

import io
import sys

from src.tools import _NO_SOURCE_NOTE, query_defect_stats, simulate_threshold

# Windows 콘솔(cp949)에서 한글·기호 출력이 죽지 않게 UTF-8 로 감쌉니다.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_passed = 0


def check(label: str, got, expected) -> None:
    """기대값과 실제값을 비교하고, 다르면 **즉시 멈춥니다.**

    실패를 모아서 끝에 보고하지 않는 이유: 계산이 틀어지면 뒤 항목도 줄줄이 틀립니다.
    첫 번째 불일치가 거의 항상 원인이므로, 거기서 멈추는 쪽이 읽기 쉽습니다.
    """
    global _passed
    assert got == expected, f"\n  [실패] {label}\n    기대: {expected!r}\n    실제: {got!r}"
    _passed += 1
    print(f"  OK  {label}")


def contains(label: str, haystack: str, needle: str) -> None:
    global _passed
    assert needle in haystack, f"\n  [실패] {label}\n    '{needle}' 가 없음\n    실제: {haystack!r}"
    _passed += 1
    print(f"  OK  {label}")


def main() -> None:
    print("\n── query_defect_stats ──")
    # 통과 기준의 출처: data/inspection_results.csv 를 pandas 로 직접 세어 확인한 값.
    all_cells = query_defect_stats.invoke({})
    contains("인자 없음 → 전체 1,000건", all_cells, "건수: 1000건")
    contains("계산 결과 표시가 붙는다", all_cells, _NO_SOURCE_NOTE)

    reject = query_defect_stats.invoke({"verdict": "REJECT"})
    contains("verdict=REJECT → 43건", reject, "건수: 43건")

    # ★ 대소문자 회귀. LLM 이 질문 문장을 그대로 옮기면 소문자가 들어옵니다.
    #   예전에는 여기서 조용히 0건이 나왔습니다.
    contains(
        "verdict=reject (소문자) → 같은 43건",
        query_defect_stats.invoke({"verdict": "reject"}),
        "건수: 43건",
    )
    lower_type = query_defect_stats.invoke({"defect_type": "pollution"})
    contains("defect_type=pollution (소문자) → 14건", lower_type, "건수: 14건")
    contains("표에 적힌 철자로 되돌려준다", lower_type, "결함종류=Pollution")

    contains(
        "defect_type=Damaged → 10건",
        query_defect_stats.invoke({"defect_type": "Damaged"}),
        "건수: 10건",
    )
    contains(
        "조건 2개 결합 (REJECT + confidence<=0.3) → 16건",
        query_defect_stats.invoke({"verdict": "REJECT", "max_confidence": 0.3}),
        "건수: 16건",
    )
    # 0건일 때 NaN 이 새어 나가지 않아야 합니다.
    none_found = query_defect_stats.invoke({"defect_type": "Unicorn"})
    contains("없는 결함종류 → 0건 안내", none_found, "0건")
    assert "nan" not in none_found.lower(), f"0건 응답에 nan 이 섞였습니다: {none_found!r}"
    print("  OK  0건 응답에 nan 이 없다")

    print("\n── simulate_threshold ──")
    # ★ 채택 운영점. 이 세 줄이 README·포트폴리오에 인용되는 수치입니다.
    op = simulate_threshold.invoke({"confidence_threshold": 0.05})
    contains("0.05 → Recall 0.977", op, "Recall: 0.977")
    contains("0.05 → 재검 부하 6.6%", op, "재검 부하: 6.6%")
    contains("0.05 → 놓친 불량 1건", op, "놓친 불량: 1건")
    contains("계산 결과 표시가 붙는다", op, _NO_SOURCE_NOTE)

    t03 = simulate_threshold.invoke({"confidence_threshold": 0.3})
    contains("0.3 → Recall 0.628", t03, "Recall: 0.628")
    contains("0.3 → 재검 부하 2.7%", t03, "재검 부하: 2.7%")
    contains("0.3 → 놓친 불량 16건", t03, "놓친 불량: 16건")

    contains(
        "0.5 → 놓친 불량 21건",
        simulate_threshold.invoke({"confidence_threshold": 0.5}),
        "놓친 불량: 21건",
    )

    # 경계값. 모든 셀을 REJECT 로 판정하면 놓친 불량이 0이어야 합니다.
    floor = simulate_threshold.invoke({"confidence_threshold": 0.0})
    contains("임계값 0.0 → 전부 재검, 놓친 불량 0건", floor, "놓친 불량: 0건")
    contains("임계값 0.0 → Recall 1.000", floor, "Recall: 1.000")

    print(f"\n{_passed}개 확인 전부 통과.")


if __name__ == "__main__":
    main()
