"""
단계별 동작 확인. 막혔을 때 '어디까지 되는지'를 30초 안에 확인하는 용도입니다.

실행:  python -m scripts.smoke_test

[왜 이런 파일을 두는가]
초보자가 시간을 가장 많이 날리는 지점은 코드 오류가 아니라
"어디서 잘못됐는지 모르는 상태"입니다.
단계를 나눠 확인하면 문제 구간이 한 줄로 좁혀집니다.
"""

from src import config


def step(n: int, name: str) -> None:
    print(f"\n{'='*50}\n[{n}] {name}\n{'='*50}")


def main() -> None:
    step(1, "설정 확인 (.env, 경로)")
    config.check()

    step(2, "검사 결과 CSV 읽기")
    import pandas as pd
    df = pd.read_csv(config.RESULTS_CSV)
    print(f"[OK] {len(df)}행 / 컬럼 {list(df.columns)}")
    print(df.head(3).to_string(index=False))

    step(3, "벡터DB 열기 + 검색 1회")
    from src.ingest import get_vectorstore
    store = get_vectorstore()
    hits = store.similarity_search("운영점 임계값을 정한 이유", k=2)
    if not hits:
        raise RuntimeError("검색 결과가 비었습니다. 먼저 `python -m src.ingest` 를 실행하세요.")
    for h in hits:
        print(f"  [{h.metadata.get('source')}] {h.page_content[:80]}...")
    print("[OK] 벡터DB 정상")

    step(4, "LLM 호출 1회")
    from langchain_openai import ChatOpenAI
    msg = ChatOpenAI(model=config.LLM_MODEL, temperature=0).invoke("'준비 완료'라고만 답하세요.")
    print(f"[OK] LLM 응답: {msg.content}")

    step(5, "에이전트 1회 (도구 호출 포함)")
    from src.agent import ask
    out = ask("conf 임계값을 0.05로 정한 이유가 뭔가요?")
    print(f"사용한 도구: {out['used_tools']}")
    print(f"답변: {out['answer'][:300]}")

    print("\n" + "="*50)
    print("모든 단계 통과. 이제 `python -m eval.run_eval` 로 평가를 돌리세요.")


if __name__ == "__main__":
    main()
