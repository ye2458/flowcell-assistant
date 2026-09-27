"""
CLI 진입점. 터미널에서 질문 한 번 던지고 답을 받습니다.

    python -m scripts.ask "라벨 품질 이슈가 뭐였죠?"

[왜 이 파일이 있는가]
`src/api.py` 로 서버를 띄우면 브라우저를 열고 JSON 본문을 채워야 합니다.
"문서 내용을 빠르게 확인하는" 용도로는 그 과정이 번거롭습니다.
같은 `ask()` 를 부르는 얇은 껍데기를 하나 둬서, 자료를 찾는 사람이
서버를 신경 쓰지 않게 했습니다.

[왜 로직을 여기 두지 않는가]
질문 처리 로직은 `src/agent.py` 의 `ask()` 하나뿐입니다.
CLI와 API가 **같은 함수**를 부르므로, 두 경로의 답이 갈릴 수 없습니다.
진입점마다 로직을 복사하면 "CLI에서는 되는데 API에서는 안 된다"가 생깁니다.
"""

import io
import sys

from src import config
from src.agent import ask

# ★ Windows 콘솔은 기본 인코딩이 cp949 라서, 한글 답변을 그대로 print 하면
#   UnicodeEncodeError 로 죽습니다(실제로 이 프로젝트에서 겪은 오류입니다).
#   출력 스트림을 UTF-8 로 다시 감싸서 환경에 상관없이 같게 동작하게 합니다.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main() -> None:
    # sys.argv[0] 은 스크립트 이름이므로 1번째부터가 질문입니다.
    # 여러 단어를 공백으로 이어 붙여, 따옴표를 빼먹어도 동작하게 합니다.
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('사용법: python -m scripts.ask "질문을 여기에"')
        sys.exit(1)

    config.check()
    result = ask(question)

    print(f"\n질문: {question}")
    print("─" * 68)
    print(result["answer"])
    print("─" * 68)

    # ★ 호출된 도구를 항상 같이 보여줍니다.
    #   답변만 보면 '문서에서 찾은 내용'과 '표에서 계산한 숫자'를 구분할 수 없습니다.
    #   이 구분이 중요한 이유: 표는 샘플 데이터이므로 그 숫자를 외부 문서에
    #   인용하면 안 됩니다. 어느 도구가 답했는지 보이면 그 실수를 막을 수 있습니다.
    used = result["used_tools"]
    print(f"사용한 도구: {used if used else '(없음 — 도구 없이 답하거나 거절함)'}")

    if any(t in used for t in ("query_defect_stats", "simulate_threshold")):
        print("\n[주의] 이 답의 수치는 data/inspection_results.csv 에서 계산한 것입니다.")
        print("       이 표는 샘플 데이터이므로 실제 모델 성능으로 인용하지 마세요.")


if __name__ == "__main__":
    main()
