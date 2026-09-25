"""
검색 품질(retrieval) 측정기.  실행: python -m eval.retrieval_check

[왜 이 파일이 따로 있는가 — T8에서 부딪힌 문제]
`run_eval.py` 가 재는 것은 **도구 선택 정확도**입니다. 그런데 T8에서
설정 A(chunk 500 / top_k 4)와 설정 B(chunk 1000 / top_k 2)가 **둘 다 100%** 가 나왔습니다.

지표가 포화되면(모두 만점) 그 지표로는 두 설정을 비교할 수 없습니다.
"어느 쪽이 나은지 모른다"가 정직한 결론이지만, 그 상태로는 CHUNK_SIZE 를
무엇으로 정할지 답할 수 없습니다. 그래서 **구분되는 지표를 하나 더** 만듭니다.

[무엇을 재는가]
도구 선택은 LLM의 판단이고, 검색은 그보다 앞단입니다. 두 층을 분리해서 봅니다.

  1층 검색:  질문 → 청크 k개              ← 이 파일이 재는 곳
  2층 판단:  청크 → 도구 선택 / 최종 답변   ← run_eval.py 가 재는 곳

1층이 답을 못 가져오면 2층은 아무리 똑똑해도 맞힐 수 없습니다.
그래서 rag 8문항마다 '답이 들어 있어야만 하는 문자열'을 하나 정해 두고,
검색된 청크 안에 그게 실제로 들어왔는지 봅니다.

  - hit rate    : 정답 문자열이 검색 결과에 들어온 비율  (높을수록 좋음)
  - 평균 문자수 : LLM에게 넘기는 컨텍스트 크기          (낮을수록 싸고 빠름)

[한계 — 먼저 밝힐 것]
문자열 포함 여부는 '의미가 맞는지'까지 보지 못합니다.
정답 문자열을 사람이 골랐으므로 그 선택에 편향이 들어갈 수도 있습니다.
그래도 도구 선택 정확도가 포화된 상황에서 **두 설정을 구분해 주는** 지표이고,
채점에 LLM을 쓰지 않으므로(채점기를 또 평가할 필요가 없으므로) 재현됩니다.
"""

import io

from src import config
from src.ingest import get_vectorstore

# rag 8문항 × (질문, 검색 결과에 반드시 들어와야 하는 문자열)
# 문자열은 '그 질문의 답이 적힌 바로 그 대목'에서 골랐습니다.
# 예: 5번은 원인이 "병목이 GPU가 아니라 CPU"라는 대목에만 '1.15배'가 붙어 있습니다.
RAG_PROBES = [
    (1, "conf 임계값을 0.05로 정한 이유가 무엇인가요?", "대칭"),
    # 2번 주의: 처음에 정답 문자열을 '오탐'으로 잡았다가 틀렸습니다.
    # 오탐(각인·이음매)은 '타일 경계 처리' 이야기이고, tiled 모델을 배포하지 않은
    # 실제 근거는 '홀드아웃에서 붕괴'(02_experiment_log.md:13,16)입니다.
    # → 정답 문자열을 잘못 고르면 검색이 잘 됐는데도 실패로 잡힙니다(측정기의 버그).
    (2, "tiled 모델을 배포하지 않기로 한 근거는 무엇인가요?", "붕괴"),
    (3, "CT와 EXT를 하나의 모델로 합치지 않고 분리한 이유는?", "unpaired"),
    (4, "라벨 오류로 평가에서 제외한 셀 ID를 알려주세요.", "223"),
    (5, "TensorRT FP16 변환 효과가 1.15배에 그친 이유가 뭔가요?", "1.15배"),
    (6, "동분포 평가와 홀드아웃 평가를 왜 나눠서 보고하나요?", "홀드아웃"),
    (7, "EXT 점 recall 0.774 수치에 어떤 한계가 있나요?", "0.774"),
    (8, "최종 배포 인스턴스와 처리 방식은 어떻게 정해졌나요?", "g4dn.2xlarge"),
]


def main() -> None:
    store = get_vectorstore()
    setting = f"chunk={config.CHUNK_SIZE} / overlap={config.CHUNK_OVERLAP} / top_k={config.TOP_K}"
    print(f"설정: {setting}\n")

    hits = 0
    total_chars = 0
    lines = []

    for qid, question, must_contain in RAG_PROBES:
        documents = store.similarity_search(question, k=config.TOP_K)
        context = "\n".join(d.page_content for d in documents)
        sources = [d.metadata["source"] for d in documents]

        found = must_contain in context
        hits += found
        total_chars += len(context)

        mark = "O" if found else "X"
        print(f"[{qid:>2}] {mark}  '{must_contain}' / {len(context):>5}자 / {sources}")
        lines.append(f"| {qid} | `{must_contain}` | {mark} | {len(context)} |")

    hit_rate = hits / len(RAG_PROBES)
    avg_chars = total_chars / len(RAG_PROBES)

    print(f"\n검색 정확도(hit rate): {hit_rate:.0%}  ({hits}/{len(RAG_PROBES)})")
    print(f"평균 컨텍스트 크기   : {avg_chars:.0f}자")

    # 결과를 파일로 남깁니다. 두 설정을 비교하려면 A를 돌리고 B를 돌린 뒤
    # 두 파일을 나란히 봐야 하는데, 콘솔 출력은 사라지기 때문입니다.
    out = config.EVAL_DIR / f"retrieval_chunk{config.CHUNK_SIZE}_topk{config.TOP_K}.md"
    with io.open(out, "w", encoding="utf-8") as f:
        f.write(f"# 검색 품질 — {setting}\n\n")
        f.write("| id | 정답 문자열 | 검색됨 | 컨텍스트 문자수 |\n|---|---|---|---|\n")
        f.write("\n".join(lines))
        f.write(f"\n\n- hit rate: **{hit_rate:.0%}** ({hits}/{len(RAG_PROBES)})\n")
        f.write(f"- 평균 컨텍스트: **{avg_chars:.0f}자**\n")
    print(f"\n저장 → {out}")


if __name__ == "__main__":
    main()
