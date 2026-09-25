"""
문서를 벡터DB에 넣는 스크립트. 프로젝트에서 딱 한 번(문서가 바뀔 때마다) 실행합니다.

실행:  python -m src.ingest

[전체 흐름]
  data/docs/*.md  →  읽기  →  청크로 자르기  →  임베딩(숫자 벡터로 변환)  →  Chroma에 저장

[왜 청크로 자르는가]
문서 하나를 통째로 임베딩하면 벡터 하나가 문서 전체를 뭉뚱그려 표현합니다.
"conf 0.05를 왜 골랐나"를 물었을 때 문서 전체가 딸려오면
LLM에게 쓸데없는 내용까지 주게 되고, 정확도와 비용이 동시에 나빠집니다.
문단 단위로 잘라야 질문에 맞는 조각만 골라올 수 있습니다.

[왜 Chroma 인가]
- 별도 서버를 띄울 필요가 없습니다(파일로 저장). 10일짜리 프로젝트에 인프라를 늘리지 않습니다.
- 메타데이터(어느 문서에서 왔는지)를 같이 저장할 수 있어 '출처 인용'이 가능합니다.
  출처가 없으면 환각인지 아닌지 확인할 방법이 없습니다.
- 대안: FAISS는 더 빠르지만 메타데이터를 직접 관리해야 해서 코드가 늘어납니다.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config


def load_documents() -> list[Document]:
    """data/docs 안의 .md 파일을 모두 읽어 Document 객체로 만듭니다."""
    docs = []
    for path in sorted(config.DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        # metadata 에 파일명을 남깁니다. 나중에 답변에 "출처: 03_mislabel_issue.md" 로 붙습니다.
        docs.append(Document(page_content=text, metadata={"source": path.name}))
    if not docs:
        raise RuntimeError(f"{config.DOCS_DIR} 에 .md 파일이 없습니다.")
    print(f"[1/3] 문서 {len(docs)}개 로드")
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """
    RecursiveCharacterTextSplitter 를 쓰는 이유:
    문단(\n\n) → 줄(\n) → 문장 → 글자 순서로 자를 자리를 찾습니다.
    그냥 500자마다 끊으면 문장 한가운데가 잘려 검색 품질이 떨어집니다.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"[2/3] 청크 {len(chunks)}개 생성 (chunk_size={config.CHUNK_SIZE})")
    return chunks


def build_vectorstore(chunks: list[Document]) -> Chroma:
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.VECTORDB_DIR),
    )
    print(f"[3/3] 벡터DB 저장 완료 → {config.VECTORDB_DIR}")
    return store


def get_vectorstore() -> Chroma:
    """이미 만들어진 벡터DB를 열기만 합니다. tools.py 에서 사용합니다."""
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.VECTORDB_DIR),
    )


if __name__ == "__main__":
    config.check()
    build_vectorstore(split_documents(load_documents()))
