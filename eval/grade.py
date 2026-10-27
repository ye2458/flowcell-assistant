"""
수동 채점 결과를 결과 CSV에 반영합니다.  실행: python -m eval.grade

[왜 이 파일이 따로 있는가]
`run_eval.py` 가 자동으로 채우는 것은 **형식**입니다 — 도구를 골랐는가,
출처 파일명이 맞는가, 사실 앵커가 들어 있는가. 전부 문자열 검사입니다.

그런데 형식을 전부 통과하고도 **내용이 틀릴 수 있습니다.** 실제로 그랬습니다.
3번과 7번은 올바른 도구를 부르고, 올바른 출처를 인용하고, 앵커까지 포함했는데
**근거로 제시한 내용이 그 질문과 무관한 절에서 왔습니다.**
문자열 검사로는 원리적으로 잡히지 않는 유형입니다.

그래서 사람이 원문과 대조한 판정을 여기에 표로 남깁니다. 채팅이나 기억이 아니라
**파일로** 남겨야 나중에 "왜 이 문항을 0점 줬나"를 되짚을 수 있습니다.

[왜 LLM 채점기를 쓰지 않는가]
채점기를 쓰면 **채점기를 또 평가해야 합니다.** 20~25문항 규모에서는
사람이 원문을 열어 대조하는 편이 빠르고, 몇 번 돌려도 같은 값이 나옵니다.
"""

import csv
import glob
import io
import os
import sys

from src import config

# id: (answer_correct, hallucination, 판정 근거)
#   answer_correct : 답변 내용이 맞는가
#   hallucination  : 문서에 없는 내용을 사실처럼 진술했는가
#                    (사실 자체는 문서에 있어도, 없는 인과관계를 만들면 1)
GRADES = {
    1:  (1, 0, "비용 비대칭·F1 최대점(conf 0.12) 미채택 모두 04 문서와 일치."),
    2:  (0, 1, "홀드아웃 성능 저하·오탐은 정확하나, 3번으로 든 '라벨 품질 이슈'는 "
               "tiled 미배포 근거가 아니다. 문서에 없는 인과관계이고 출처도 03 이 섞였다."),
    3:  (1, 0, "unpaired·대응관계만 정확히 설명. 2026-09-29 코퍼스 정정 전에는 "
               "무관한 'SAHI 오탐'을 근거로 붙였는데, 원문을 절 단위로 정리하자 사라졌다."),
    4:  (1, 0, "셀 ID 9개(208~214,219,223) 정확."),
    5:  (1, 0, "CPU 병목이 원인. 04 문서와 일치."),
    6:  (1, 0, "과대평가 방지 / 홀드아웃 기준 배포 판단. 01 문서와 일치."),
    7:  (1, 0, "120장·독립검증 없음·Damaged F1 0.455 모두 EXT 절 내용과 일치. "
               "이전 실행에서 CT 쪽 타일 오탐을 끌어붙이던 혼동이 해소됐다."),
    8:  (1, 0, "g4dn.2xlarge / SQS 비동기 / 4.2분 대기 정확."),
    9:  (1, 0, "43건. CSV 실측 일치."),
    10: (1, 0, "14건. 실측 일치."),
    11: (1, 0, "16건. 실측 일치."),
    12: (1, 0, "0.0289. 실측 일치."),
    13: (1, 0, "10건 / 0.0874~0.7581. 실측 일치."),
    14: (1, 0, "6건. 실측 일치."),
    15: (1, 0, "recall 0.837 / 재검부하 4.7% / 놓친 불량 7건. 실측 일치."),
    16: (1, 0, "21건. 실측 일치."),
    17: (1, 0, "0.05(0.977/6.6%/1건), 0.3(0.628/2.7%/16건) 모두 일치. 도구 2회 호출."),
    18: (1, 0, "미래 예측 거절. 도구 미호출."),
    19: (1, 0, "외부 정보 거절. 도구 미호출."),
    20: (1, 0, "타 도메인 적용 판단 거절. 도구 미호출."),
    21: (1, 0, "거짓 전제(평가 데이터 정제)를 정확히 부정하고 '학습 데이터에서 제외 후 "
               "재학습, 평가는 동일 홀드아웃'을 명시. 다만 01 을 불필요하게 함께 인용했다."),
    22: (1, 0, "거짓 전제(합쳐서 학습)를 부정하고 독립 파이프라인 설계를 정확히 설명."),
    23: (1, 0, "tiled 가 챔피언이 아님을 명시하고 배포 챔피언 seg-v2 를 제시."),
    24: (1, 0, "3배가 아니라 1.15배임을 명시하고 CPU 병목까지 설명."),
    25: (1, 0, "F1 최대점(conf 0.12)이 아님을 명시하고 비용 비대칭으로 설명."),
}


def latest_results() -> str:
    """가장 최근 results_*.csv. 최종 채점본(FINAL)은 입력에서 제외합니다."""
    files = [f for f in glob.glob(str(config.EVAL_DIR / "results_*.csv")) if "FINAL" not in f]
    if not files:
        sys.exit("results_*.csv 가 없습니다. 먼저 python -m eval.run_eval 을 실행하세요.")
    return sorted(files)[-1]


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else latest_results()
    print(f"채점 대상: {src}")

    with io.open(src, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    fields = list(rows[0].keys())
    for col in ("grading_note_MANUAL", "graded_from_MANUAL"):
        if col not in fields:
            fields.append(col)

    for r in rows:
        qid = int(r["id"])
        if qid not in GRADES:
            sys.exit(f"{qid}번 문항의 채점 결과가 GRADES 에 없습니다. 먼저 채점하세요.")
        ac, hl, why = GRADES[qid]
        r["answer_correct_MANUAL"] = ac
        r["hallucination_MANUAL"] = hl
        r["grading_note_MANUAL"] = why
        # ★ 어느 실행을 채점한 것인지 각 행에 남깁니다.
        #   수동 채점은 '그 실행의 답변'을 보고 매긴 것이라, 평가를 다시 돌리면
        #   답변이 달라져 점수가 낡습니다. 출처를 적어두면 그걸 알아챌 수 있습니다.
        r["graded_from_MANUAL"] = os.path.basename(src)

    dst = config.EVAL_DIR / f"results_FINAL_chunk{config.CHUNK_SIZE}_topk{config.TOP_K}.csv"
    with io.open(dst, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    ac = sum(int(r["answer_correct_MANUAL"]) for r in rows)
    hl = sum(int(r["hallucination_MANUAL"]) for r in rows)

    print(f"\n답변 정확도 : {ac}/{n} = {ac/n:.0%}")
    print(f"환각 발생   : {hl}/{n} = {hl/n:.0%}")
    print("\n── 감점 문항 ──")
    for r in rows:
        if int(r["answer_correct_MANUAL"]) == 0 or int(r["hallucination_MANUAL"]) == 1:
            print(f"  [{r['id']:>2}] {r['category']:<10} {r['grading_note_MANUAL']}")
    print(f"\n최종본 저장 → {dst}")


if __name__ == "__main__":
    main()
