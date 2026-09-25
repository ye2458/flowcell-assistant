# TASKS — 진행 상태

각 단계에는 **통과 기준**이 있습니다. 기준을 만족하지 못하면 다음으로 넘어가지 않습니다.
단계를 마치면 체크박스를 갱신하세요.

**실질 마감: 9/10(목) 밤.** 9/11~13은 작업 불가이고, 9/14는 자소서를 쓰는 날입니다.

---

## T0 — 환경 ✅ 완료

- [x] OpenAI API 키 발급
- [x] `.venv` 생성·활성화
- [x] `pip install -r requirements.txt`
- [x] `.env`에 키 입력 → `config.check()` 통과
- [x] **크레딧 충전** (해결됨)
- [x] `pip freeze | Out-File -Encoding utf8 requirements.lock.txt` (109개 패키지)

---

## T1 — 인덱싱 ✅ 완료

```powershell
python -m src.ingest
```

- [x] 실행 성공
- ✅ **통과 기준**: `[3/3] 벡터DB 저장 완료` 출력, `data/vectordb/` 폴더 생성

> 확인 완료 (2026-09-25): `cellnex_docs` 컬렉션에 청크 10개가 저장돼 있습니다.

---

## T2 — TODO 1: RAG 검색 (`src/tools.py`) ✅ 완료

`search_inspection_docs` 구현.

- [x] 벡터DB에서 관련 청크 검색
- [x] **출처 파일명을 답변 문자열에 포함**
- ✅ **통과 기준**: 직접 호출했을 때 `[출처: 04_deployment_spec.md]` 형태가 붙어 나옴

**생각해볼 것**: 출처를 왜 반드시 붙여야 하는가? 붙이지 않으면 나중에 무엇을 할 수
없게 되는가?

---

## T3 — TODO 4: 라우팅 (`src/agent.py`) ✅ 완료

`should_continue` 구현 (2줄).

- [x] 마지막 메시지에 도구 호출이 있으면 `"tools"`, 없으면 `END`
- [x] `python -m scripts.smoke_test` — 5단계 전부 통과
- ✅ **통과 기준**: 5단계 전부 통과, 마지막에 `사용한 도구: ['search_inspection_docs']`

**생각해볼 것**: 왜 도구 실행 후 다시 `agent` 노드로 돌아가는가? 한 번만 부르고
끝내면 어떤 질문에서 틀리는가?

> 🚩 **여기까지가 1차 관문입니다.** RAG + Agent 루프가 도는 상태.

---

## T4 — TODO 2: 통계 도구 ✅ 완료

`query_defect_stats` 구현 (pandas 필터).

- [x] 세 인자를 조건부로 적용 (43건 확인)
- ✅ **통과 기준**: "REJECT 셀 몇 개?" 질문에 **43** 이 나옴

**생각해볼 것**: 인자를 왜 전부 Optional로 뒀는가? 필수로 만들면 LLM이 어떻게
행동하는가?

---

## T5 — TODO 3: 임계값 시뮬레이션 ★ ✅ 완료

`simulate_threshold` 구현.

- [x] `pred_score >= threshold` 를 REJECT로 판정
- [x] recall, 재검 부하, 놓친 불량 수 계산 (0.05 → 0.977 / 6.6% / 1건 확인)
- ✅ **통과 기준**: `0.05` 입력 시 **recall 0.977 / 재검 부하 6.6% / 놓친 불량 1건**

> 정답을 미리 알고 시작하는 셈이라 디버깅이 쉽습니다. 숫자가 안 맞으면 계산식이
> 틀린 겁니다.

**생각해볼 것**: 재검 부하를 왜 precision 대신 썼는가? 운영 중에 어느 쪽이
계산 가능한가?

---

## T6 — API 서빙 ✅ 완료

```powershell
uvicorn src.api:app --reload
```

- [x] `/health` → `{"status":"ok"}`, `/ask` → 정상 응답 확인 (curl 검증)
- [ ] **스크린샷 저장** (포트폴리오용) ← **직접 해야 하는 유일한 작업**
      `uvicorn src.api:app --reload` 실행 → `http://127.0.0.1:8000/docs` 접속
      → `/ask` 에 질문 입력 후 응답 화면 캡처 → `docs/screenshot_api.png`
- ✅ **통과 기준**: 브라우저에서 질문 입력 → 답변 반환

---

## T7 — 평가 ★ 가장 값나가는 단계 ✅ 완료

```powershell
python -m eval.run_eval
```

- [x] 카테고리별 도구 선택 정확도: rag/stats/simulation/refuse **전부 100%**
- [x] 수동 채점 완료 → `eval/results_FINAL_chunk500_topk4.csv`
      답변 정확도 19/20(95%) · 환각 0/20 · 출처 정확도 7/8(88%)
- [x] 1차 평가에서 rag #5 실패 → 도구 docstring 수정으로 88%→100% (TROUBLESHOOTING 4)
- ✅ **통과 기준**: rag/stats/simulation/refuse 4개 카테고리 정확도가 나옴

**주목할 것**: `refuse` 3문항의 정확도. 여기가 낮으면 시스템 프롬프트나 도구 설명을
고쳐야 합니다.

**이미 관측된 이슈 (2026-09-25)**: 복합 질문에서 LLM이 `simulate_threshold` 의
계산 결과에까지 `(출처: 04_deployment_spec.md)` 를 붙였습니다. 계산값은 문서 출처가
없는데도 붙인 것이므로 **출처 오귀속**입니다. 시스템 프롬프트에서
"문서 검색 결과에만 출처를 붙이고, 계산 결과에는 붙이지 말 것"을 명시해 고칩니다.

---

## T8 — 파라미터 비교 (30분, 생략 금지) ✅ 완료

- [x] 설정 B(1000/2) 로 재인덱싱 후 재평가, 설정 A(500/4) 로 복원
- [x] 도구 선택 정확도는 양쪽 100%로 **포화** → 비교 불가 판정
- [x] 검색 품질 지표(`eval/retrieval_check.py`)를 새로 만들어 구분: A 100% vs B 75%
- [x] `docs/EXPERIMENTS.md` 에 기록 → **설정 A 채택**
- ✅ **통과 기준**: 두 설정의 도구 선택 정확도 비교표

> **이 30분이 "튜토리얼 복붙 아니냐"는 의심을 지웁니다.** 면접에서 "청크 크기는
> 어떻게 정했습니까"에 답할 근거가 여기서 나옵니다.

---

## T9 — 기록 ✅ 거의 완료

- [x] `docs/TROUBLESHOOTING.md` 항목 5개로 정리 → README에 트러블슈팅 절 추가
- [x] 평가 결과표 + 파라미터 비교표를 README에 붙이기
- [x] `git init` + 첫 커밋 완료 (28개 파일, 커밋 `0e5c833`)
- [x] **확인 완료**: `.env` / `docs/_private/` / `data/vectordb/` 모두 제외됨
- [x] `CLAUDE.md` 와 `docs/QUIZ.md` 도 제외 (작업 방식·학습용 문서 — 공개 여부는 본인 판단)
- [ ] 깃허브 원격 저장소 생성 후 push ← **본인 계정 인증 필요**
      원격: `ye2458/flowcell-assistant` (푸시 완료)

---

## 남은 작업 (2개, 본인이 해야 함)

1. **API 스크린샷** (T6) — `uvicorn src.api:app --reload` → `/docs` 접속 → `/ask` 호출 후 캡처
2. **GitHub push** — 위 명령 (private 로 먼저 만들고 내용 확인 후 public 전환 권장)

## 다음 단계: 이해도 확인

`docs/QUIZ.md` 의 문항에 답하기. 정답은 `docs/_private/QUIZ_ANSWERS.md` 에 있으나
**먼저 답한 뒤에** 펼칠 것.

---

## 보너스 — T7까지 끝난 뒤에만

- [ ] Dockerfile
- [ ] MCP 서버화

**T7 이전에는 절대 손대지 않습니다.**
