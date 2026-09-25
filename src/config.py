"""
설정을 한 파일에 모읍니다.

[왜 이렇게 하는가]
노트북에서 작업할 때는 경로와 모델명을 셀마다 다시 적게 됩니다.
그러면 "어느 셀의 설정이 진짜인지" 알 수 없어져서, 실행 순서에 따라 결과가 달라집니다.
설정을 한 곳에 모으면 (1) 바꿀 때 한 군데만 고치면 되고
(2) 나중에 Docker/서버로 옮길 때 환경변수만 갈아끼우면 됩니다.
이게 '노트북 코드'와 '서비스 코드'의 첫 번째 차이입니다.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# .env 파일을 읽어 os.environ 에 올립니다. 프로젝트에서 딱 한 번만 호출됩니다.
load_dotenv()

# ── 경로 ──────────────────────────────────────────────
# __file__ = 이 파일의 위치(src/config.py)
# .parent  = src/ ,  .parent.parent = 프로젝트 루트
# 상대경로("./data")를 쓰지 않는 이유: 터미널을 어느 폴더에서 열었느냐에 따라
# 상대경로는 깨집니다. 절대경로로 고정해야 어디서 실행해도 동작합니다.
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
VECTORDB_DIR = DATA_DIR / "vectordb"
RESULTS_CSV = DATA_DIR / "inspection_results.csv"
EVAL_DIR = ROOT_DIR / "eval"

# ── 모델 ──────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ── RAG 파라미터 ──────────────────────────────────────
# 이 세 값은 나중에 평가셋으로 한 번은 바꿔서 돌려보세요.
# "감으로 정했습니다"와 "20문항으로 두 조합을 비교했습니다"는 면접에서 급이 다릅니다.
CHUNK_SIZE = 500          # 청크 하나의 글자 수
CHUNK_OVERLAP = 80        # 청크 간 겹침. 문장이 경계에서 잘리는 것을 완화합니다.
TOP_K = 4                 # 검색해올 청크 개수

COLLECTION_NAME = "cellnex_docs"


def check() -> None:
    """설정이 제대로 잡혔는지 확인합니다. smoke_test 에서 호출합니다."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY 가 없습니다. .env.example 을 .env 로 복사하고 키를 채우세요."
        )
    if not DOCS_DIR.exists():
        raise RuntimeError(f"문서 폴더가 없습니다: {DOCS_DIR}")
    print(f"[OK] 설정 정상 / LLM={LLM_MODEL} / 문서폴더={DOCS_DIR}")
