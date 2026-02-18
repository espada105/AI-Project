# RAG (Retrieval-Augmented Generation) 파이프라인 실습 가이드

함수 중심으로 구현된 RAG 시스템을 단계별로 학습하고 실습할 수 있는 코드입니다.

## 📋 목차

1. [설치 방법](#설치-방법)
2. [RAG 파이프라인 구조](#rag-파이프라인-구조)
3. [함수별 설명](#함수별-설명)
4. [실습 예제](#실습-예제)
5. [핵심 개념 정리](#핵심-개념-정리)

---

## 🚀 설치 방법

### 1. 필요한 패키지 설치

```bash
cd LangChain
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일에 OpenAI API 키가 설정되어 있어야 합니다:

```
OPENAI_API_KEY=your-api-key-here
```

---

## 📊 RAG 파이프라인 구조

```
┌─────────────────────────────────────────────────────────┐
│                    RAG 파이프라인                        │
└─────────────────────────────────────────────────────────┘

[1단계] 인덱싱 (Indexing)
    ↓
    Load → Split → Embed → Store
    ↓
    [VectorStore + BM25 Index 생성]

[2단계] 검색 (Retrieve)
    ↓
    Hybrid Search (BM25 + Dense Embedding)
    ↓
    RRF (Reciprocal Rank Fusion)
    ↓
    ReRanking (선택적)
    ↓
    [상위 N개 문서 선정]

[3단계] 생성 (Generation)
    ↓
    Context Injection → Prompt Engineering → LLM Inference
    ↓
    [최종 답변 생성]
```

---

## 🔧 함수별 설명

### 1단계: 인덱싱 함수들

#### `load_documents() -> List[str]`
- **역할**: 문서를 로드하는 함수
- **실습 포인트**: 실제로는 PDF, TXT, DB 등에서 데이터를 로드합니다
- **반환값**: 문서 문자열 리스트

#### `split_documents(documents, chunk_size, chunk_overlap) -> List[Document]`
- **역할**: 문서를 작은 청크로 분할
- **파라미터**:
  - `chunk_size`: 각 청크의 최대 크기 (기본값: 200)
  - `chunk_overlap`: 청크 간 겹치는 부분 (기본값: 50)
- **실습 포인트**: `overlap` 크기를 조절하여 의미가 끊기지 않도록 합니다
- **반환값**: Document 객체 리스트

#### `embed_and_store(chunks, persist_directory) -> Chroma`
- **역할**: 텍스트를 벡터로 변환하여 VectorStore에 저장
- **파라미터**:
  - `chunks`: 분할된 문서 청크들
  - `persist_directory`: 벡터 DB 저장 경로 (기본값: "./chroma_db")
- **반환값**: Chroma 벡터 스토어 객체

#### `create_bm25_retriever(chunks) -> BM25Retriever`
- **역할**: BM25 기반 키워드 검색 인덱스 생성
- **반환값**: BM25Retriever 객체

---

### 2단계: 검색 함수들

#### `hybrid_search(query, vectorstore, bm25_retriever, top_k) -> List[Document]`
- **역할**: BM25와 Dense Embedding을 결합한 하이브리드 검색
- **파라미터**:
  - `query`: 사용자 질문
  - `vectorstore`: 벡터 스토어 (Dense 검색용)
  - `bm25_retriever`: BM25 검색기 (Sparse 검색용)
  - `top_k`: 반환할 상위 결과 수 (기본값: 10)
- **알고리즘**: RRF (Reciprocal Rank Fusion)
- **가중치**: BM25 40%, Dense 60%
- **반환값**: 검색된 Document 리스트

#### `rerank_documents(query, documents, top_n) -> List[Document]`
- **역할**: 검색 결과를 재정렬하여 관련성 높은 문서를 상단에 배치
- **파라미터**:
  - `query`: 사용자 질문
  - `documents`: 초기 검색 결과
  - `top_n`: 최종 반환할 문서 수 (기본값: 3)
- **목적**: 'Lost in the Middle' 현상 방지
- **반환값**: 재정렬된 Document 리스트

---

### 3단계: 생성 함수들

#### `create_prompt_template() -> ChatPromptTemplate`
- **역할**: RAG를 위한 프롬프트 템플릿 생성
- **핵심 제약**: 주어진 문서 내용만 바탕으로 답변

#### `generate_answer(query, context_docs, llm) -> str`
- **역할**: 최종 답변 생성
- **파라미터**:
  - `query`: 사용자 질문
  - `context_docs`: 검색된 문서들
  - `llm`: LLM 모델 객체
- **프로세스**: Context Injection → Prompt Engineering → LLM Inference
- **반환값**: 생성된 답변 문자열

---

### 통합 함수

#### `run_rag_pipeline(query, use_reranker) -> Tuple[str, List[Document]]`
- **역할**: 전체 RAG 파이프라인을 한 번에 실행
- **파라미터**:
  - `query`: 사용자 질문
  - `use_reranker`: ReRanker 사용 여부 (기본값: True)
- **반환값**: (답변, 검색된 문서들)

---

## 💡 실습 예제

### 기본 실행

```python
from rag_pipeline import run_rag_pipeline

# 전체 파이프라인 실행
answer, docs = run_rag_pipeline("RAG가 무엇인가요?", use_reranker=True)
```

### 단계별 실습

`rag_example.py` 파일에서 각 단계를 개별적으로 실습할 수 있습니다:

```python
from rag_example import (
    example_1_indexing,      # 인덱싱만
    example_2_retrieval,     # 검색만
    example_3_reranking,     # ReRanking만
    example_4_full_pipeline, # 전체 파이프라인
    example_5_comparison     # ReRanker 전/후 비교
)

# 예제 실행
example_1_indexing()
```

### 직접 실습하기

```python
from rag_pipeline import *

# 1. 문서 로드
documents = load_documents()

# 2. 문서 분할
chunks = split_documents(documents, chunk_size=200, chunk_overlap=50)

# 3. 벡터 스토어 생성
vectorstore = embed_and_store(chunks)

# 4. BM25 Retriever 생성
bm25_retriever = create_bm25_retriever(chunks)

# 5. 하이브리드 검색
query = "하이브리드 검색은 어떻게 작동하나요?"
results = hybrid_search(query, vectorstore, bm25_retriever, top_k=10)

# 6. ReRanking
reranked = rerank_documents(query, results, top_n=3)

# 7. 답변 생성
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
answer = generate_answer(query, reranked, llm)
print(answer)
```

---

## 📚 핵심 개념 정리

### Sparse vs Dense Retrieval

| 구분 | Sparse (BM25) | Dense (Embedding) |
|------|---------------|-------------------|
| **특징** | 키워드 중심 (Exact Match) | 의미 중심 (Semantic Match) |
| **장점** | 고유 명사, 전문 용어에 강함 | 오타나 유사어 대응 가능, 맥락 이해 |
| **단점** | 동의어 처리 불가 | 키워드 매칭이 정확하지 않을 수 있음 |
| **보정** | **ReRanker**를 통해 최종 정합성 및 순위 보정 | |

### 하이브리드 검색 (Hybrid Search)

- **BM25 + Dense Embedding** 결합
- **RRF (Reciprocal Rank Fusion)** 알고리즘으로 순위 통합
- 두 방식의 장점을 모두 활용

### ReRanker의 역할

1. **Lost in the Middle 현상 방지**: 중요한 정보가 중간에 있으면 LLM이 무시하는 경향
2. **정밀한 관련성 평가**: Cross-Encoder 같은 더 복잡한 모델 사용
3. **최상위 문서 선정**: 가장 관련성 높은 문서를 상단에 배치

### RAG 파이프라인 플로우

```
사용자 질문
    ↓
[인덱싱된 문서들]
    ↓
하이브리드 검색 (BM25 + Dense)
    ↓
RRF로 순위 통합
    ↓
ReRanking (선택적)
    ↓
상위 N개 문서 선정
    ↓
Context Injection
    ↓
Prompt Engineering
    ↓
LLM Inference
    ↓
최종 답변
```

---

## 🎯 다음 단계

1. **실제 데이터로 실습**: `load_documents()` 함수를 수정하여 실제 PDF나 TXT 파일 로드
2. **ReRanker 개선**: Cohere Rerank API나 bge-reranker 모델 사용
3. **평가 루프 구축**: Retrieval Precision, Hallucination 체크 로직 추가
4. **성능 최적화**: 청크 크기, overlap 크기, top_k 값 등 하이퍼파라미터 튜닝

---

## 📝 참고 자료

- [LangChain 공식 문서](https://python.langchain.com/)
- [Chroma 벡터 DB](https://www.trychroma.com/)
- [BM25 알고리즘](https://en.wikipedia.org/wiki/Okapi_BM25)
- [RRF 알고리즘](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
