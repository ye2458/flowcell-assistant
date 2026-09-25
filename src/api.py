"""
FastAPI 서빙.

실행:  uvicorn src.api:app --reload
확인:  브라우저에서 http://127.0.0.1:8000/docs

[왜 API로 감싸는가 — 자소서에 그대로 쓸 문장]
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
  ← 원 프로젝트 배포 산정에서 SQS 비동기를 택했던 근거와 같은 이야기입니다.
- 인증이 없습니다. 사내망 가정입니다.
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.agent import ask

app = FastAPI(
    title="CELLNEX Inspection Assistant",
    description="이차전지 셀 검사 운영 어시스턴트 (RAG + Tool-calling Agent)",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., description="검사 운영에 대한 질문")


class AskResponse(BaseModel):
    answer: str
    used_tools: list[str] = Field(..., description="실제로 호출된 도구 이름")


@app.get("/health")
def health() -> dict:
    """서버가 살아있는지 확인하는 최소 엔드포인트."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    result = ask(req.question)
    return AskResponse(**result)
