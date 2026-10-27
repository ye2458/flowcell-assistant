"""
FastAPI 서빙.

실행:  uvicorn src.api:app --reload
확인:  브라우저에서 http://127.0.0.1:8000/docs

[왜 API로 감싸는가]
노트북 안의 모델은 그 노트북을 연 사람만 쓸 수 있습니다.
API로 감싸는 순간 대시보드, 다른 서비스, 다른 팀이 같은 기능을 호출할 수 있습니다.
"모델을 만들었다"와 "모델을 남이 쓸 수 있게 만들었다" 사이의 경계가 여기입니다.

[왜 FastAPI 인가]
- 타입 힌트만 적으면 요청/응답 검증(Pydantic)과 문서(/docs)가 자동 생성됩니다.
  10일짜리 프로젝트에서 스키마 문서를 손으로 쓸 시간은 없습니다.
- /docs 화면이 곧 데모 화면이 됩니다. 그래서 프론트엔드를 안 만들어도 됩니다.

[한계 — 면접에서 먼저 밝힐 것]
- 동기 처리라 요청이 몰리면 그대로 대기가 쌓입니다.
  실서비스라면 작업 큐(SQS 등)로 비동기 전환이 필요합니다.
  ← 코퍼스의 배포 사양 문서가 SQS 비동기 큐를 택한 근거와 같은 이야기입니다.
- 인증이 없습니다. 사내망 가정입니다.
"""

import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config
from src.agent import ask


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버가 뜰 때 설정을 먼저 확인합니다.

    ★ 왜 여기서 확인하는가 (2026-10-06 추가)
      전에는 .env 가 없어도 서버가 멀쩡히 뜨고, **첫 요청에서야** 알 수 없는 오류로
      터졌습니다. 띄우는 사람과 호출하는 사람이 다를 수 있으므로,
      잘못된 설정은 **가장 이른 시점에, 읽을 수 있는 말로** 실패해야 합니다.
    """
    config.check()
    yield


app = FastAPI(
    title="CELLNEX Inspection Assistant",
    description="이차전지 셀 검사 운영 어시스턴트 (RAG + Tool-calling Agent)",
    version="0.1.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="검사 운영에 대한 질문")


class AskResponse(BaseModel):
    answer: str
    used_tools: list[str] = Field(..., description="실제로 호출된 도구 이름")


@app.get("/health")
def health() -> dict:
    """서버가 살아있는지 확인하는 최소 엔드포인트."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="질문이 비어 있습니다.")

    try:
        result = ask(question)
    except Exception as e:
        # ★ 외부 호출(LLM API)은 언제든 실패합니다 — 쿼터 소진, 네트워크, 타임아웃.
        #   처리하지 않으면 FastAPI 가 500 과 함께 스택트레이스를 띄웁니다.
        #   그러면 (1) 호출한 쪽은 무엇을 해야 할지 모르고
        #   (2) 내부 경로와 라이브러리 구성이 밖으로 드러납니다.
        #
        #   그래서 전체 내용은 **서버 로그에만** 남기고, 응답에는 오류 종류만 돌려줍니다.
        #   502 를 쓰는 이유: 우리 코드가 아니라 **상위 서비스(LLM API)가 실패**한 것이고,
        #   호출한 쪽은 '잠시 후 재시도'가 올바른 대응이기 때문입니다.
        traceback.print_exc()
        raise HTTPException(
            status_code=502,
            # 원인을 단정하지 않습니다. 벡터DB 누락·CSV 손상 같은 로컬 문제도
            # 같은 경로로 들어오는데, "크레딧을 확인하라"고만 하면 엉뚱한 곳을 보게 됩니다.
            detail=f"질의 처리에 실패했습니다 ({type(e).__name__}). "
                   "서버 로그에서 전체 오류를 확인하세요. "
                   "자주 있는 원인: LLM API 장애·크레딧 소진, 벡터DB 미생성(python -m src.ingest).",
        ) from e

    return AskResponse(**result)
