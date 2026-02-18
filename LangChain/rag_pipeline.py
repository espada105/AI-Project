"""
RAG (Retrieval-Augmented Generation) 파이프라인 실습 코드
함수 중심으로 각 단계를 구현하여 플로우를 이해할 수 있도록 구성
"""

from dotenv import load_dotenv
import os
from typing import List, Dict, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi
import numpy as np

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")


# ============================================================================
# 1단계: 인덱싱 (Indexing) - LSES 파이프라인
# ============================================================================

def load_documents() -> List[str]:
    """
    Load: 문서를 로드하는 함수
    실습용 예제 데이터를 반환합니다.
    실제로는 PDF, TXT, DB 등에서 데이터를 로드합니다.
    """
    documents = [
        "LangChain은 대규모 언어 모델(LLM)을 활용한 애플리케이션 개발을 위한 프레임워크입니다. "
        "LangChain은 체인(Chain) 개념을 통해 여러 컴포넌트를 연결하여 복잡한 작업을 수행할 수 있습니다.",
        
        "RAG(Retrieval-Augmented Generation)는 검색 증강 생성 기술로, "
        "외부 지식 베이스에서 관련 정보를 검색하여 LLM의 답변 품질을 향상시킵니다. "
        "RAG는 특히 도메인 특화 지식이나 최신 정보가 필요한 경우에 유용합니다.",
        
        "벡터 임베딩(Vector Embedding)은 텍스트를 고차원 벡터 공간으로 변환하는 기술입니다. "
        "의미적으로 유사한 텍스트는 벡터 공간에서 가까운 위치에 배치됩니다. "
        "이를 통해 의미 기반 검색이 가능해집니다.",
        
        "BM25는 정보 검색에서 사용되는 순위 함수입니다. "
        "TF-IDF를 개선한 알고리즘으로, 키워드 기반 검색에 효과적입니다. "
        "BM25는 희소 검색(Sparse Retrieval)의 대표적인 방법입니다.",
        
        "하이브리드 검색(Hybrid Search)은 BM25와 Dense Embedding을 결합한 검색 방식입니다. "
        "RRF(Reciprocal Rank Fusion) 알고리즘을 사용하여 두 검색 결과의 순위를 통합합니다. "
        "이를 통해 키워드 매칭과 의미 기반 검색의 장점을 모두 활용할 수 있습니다.",
        
        "ReRanker는 초기 검색 결과를 더 정확한 모델로 재정렬하는 컴포넌트입니다. "
        "Cross-Encoder 모델을 사용하여 질문과 문서의 관련성을 더 정밀하게 평가합니다. "
        "Lost in the Middle 현상을 방지하기 위해 가장 관련성 높은 문서를 상단에 배치합니다.",
        
        "Chroma는 오픈소스 벡터 데이터베이스입니다. "
        "임베딩 벡터를 효율적으로 저장하고 검색할 수 있도록 설계되었습니다. "
        "LangChain과 통합되어 쉽게 사용할 수 있습니다.",
        
        "프롬프트 엔지니어링(Prompt Engineering)은 LLM에게 효과적인 지시를 제공하는 기술입니다. "
        "RAG에서는 검색된 문서를 컨텍스트로 제공하고, "
        "LLM에게 '주어진 문서 내용만 바탕으로 답변하라'는 제약을 부여합니다."
    ]
    return documents


def split_documents(documents: List[str], chunk_size: int = 200, chunk_overlap: int = 50) -> List[Document]:
    """
    Split (Chunking): 문서를 작은 청크로 분할하는 함수
    - chunk_size: 각 청크의 최대 크기
    - chunk_overlap: 청크 간 겹치는 부분 (의미 연속성 유지)
    
    실습 포인트: overlap 크기를 조절하여 의미가 끊기지 않도록 합니다.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    # Document 객체로 변환
    docs = [Document(page_content=text) for text in documents]
    
    # 청크로 분할
    chunks = text_splitter.split_documents(docs)
    
    print(f"✅ 문서 분할 완료: {len(documents)}개 문서 → {len(chunks)}개 청크")
    return chunks


def embed_and_store(chunks: List[Document], persist_directory: str = "./chroma_db") -> Chroma:
    """
    Embed & Store: 텍스트를 벡터로 변환하여 VectorStore에 저장하는 함수
    
    - Embedding: OpenAI의 text-embedding-ada-002 모델 사용
    - Store: Chroma 벡터 데이터베이스에 저장
    """
    embeddings = OpenAIEmbeddings()
    
    # Chroma 벡터 스토어 생성 및 저장
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"✅ 벡터 임베딩 및 저장 완료: {len(chunks)}개 청크")
    return vectorstore


def create_bm25_retriever(chunks: List[Document]) -> BM25Retriever:
    """
    BM25 Retriever 생성 함수
    키워드 기반 희소 검색을 위한 BM25 인덱스 생성
    """
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = 5  # 상위 5개 결과 반환
    
    print("✅ BM25 Retriever 생성 완료")
    return retriever


# ============================================================================
# 2단계: 검색 (Retrieve) - 하이브리드 서치 + ReRank
# ============================================================================

def hybrid_search(
    query: str,
    vectorstore: Chroma,
    bm25_retriever: BM25Retriever,
    top_k: int = 10
) -> List[Document]:
    """
    Hybrid Search: BM25와 Dense Embedding을 결합한 하이브리드 검색
    
    - Sparse Retrieval (BM25): 키워드 일치 기반
    - Dense Retrieval (Embedding): 의미적 유사성 기반
    - RRF(Reciprocal Rank Fusion): 두 결과의 순위 통합
    """
    # Ensemble Retriever 생성 (RRF 알고리즘 사용)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.4, 0.6]  # BM25 40%, Dense 60% 가중치
    )
    
    # 하이브리드 검색 실행
    results = ensemble_retriever.get_relevant_documents(query)
    
    print(f"✅ 하이브리드 검색 완료: '{query}' → {len(results)}개 결과")
    return results


def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: int = 3
) -> List[Document]:
    """
    ReRanking: 검색 결과를 재정렬하는 함수
    
    실습용 간단한 ReRanker 구현
    실제로는 Cohere Rerank API나 bge-reranker 같은 모델을 사용합니다.
    
    여기서는 문서 길이와 키워드 매칭 점수를 기반으로 간단히 재정렬합니다.
    """
    def calculate_score(query: str, doc: Document) -> float:
        """간단한 관련성 점수 계산 (실습용)"""
        query_words = set(query.lower().split())
        doc_words = set(doc.page_content.lower().split())
        
        # 키워드 매칭 점수
        keyword_score = len(query_words & doc_words) / len(query_words) if query_words else 0
        
        # 문서 길이 정규화 (너무 짧거나 긴 문서는 페널티)
        length_score = 1.0 / (1.0 + abs(len(doc.page_content) - 300) / 100)
        
        return keyword_score * 0.7 + length_score * 0.3
    
    # 각 문서에 점수 부여
    scored_docs = [(doc, calculate_score(query, doc)) for doc in documents]
    
    # 점수 기준으로 정렬
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # 상위 N개 반환
    reranked = [doc for doc, score in scored_docs[:top_n]]
    
    print(f"✅ ReRanking 완료: {len(documents)}개 → {top_n}개로 축소")
    return reranked


# ============================================================================
# 3단계: 생성 (Generation) - Context Injection + LLM Inference
# ============================================================================

def create_prompt_template() -> ChatPromptTemplate:
    """
    Prompt Engineering: RAG를 위한 프롬프트 템플릿 생성
    
    핵심 제약 사항:
    - 주어진 문서 내용만 바탕으로 답변
    - 모르는 내용은 추측하지 말 것
    """
    template = """다음 문서들을 참고하여 질문에 답변해주세요.
문서 내용만 바탕으로 답변하고, 문서에 없는 내용은 추측하지 마세요.

문서 내용:
{context}

질문: {question}

답변:"""
    
    return ChatPromptTemplate.from_template(template)


def generate_answer(
    query: str,
    context_docs: List[Document],
    llm: ChatOpenAI
) -> str:
    """
    Generation: 최종 답변 생성 함수
    
    - Context Injection: ReRank된 문서들을 프롬프트에 주입
    - LLM Inference: LLM이 컨텍스트를 해석하여 답변 생성
    """
    # 컨텍스트 결합
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    # 프롬프트 생성
    prompt_template = create_prompt_template()
    prompt = prompt_template.format(context=context, question=query)
    
    # LLM 호출
    response = llm.invoke(prompt)
    
    return response.content


# ============================================================================
# 메인 파이프라인: 전체 RAG 플로우 실행
# ============================================================================

def run_rag_pipeline(query: str, use_reranker: bool = True):
    """
    전체 RAG 파이프라인 실행 함수
    
    플로우:
    1. 인덱싱: Load → Split → Embed & Store
    2. 검색: Hybrid Search (BM25 + Dense)
    3. ReRanking (선택적)
    4. 생성: Context Injection → LLM Inference
    """
    print("\n" + "="*60)
    print("🚀 RAG 파이프라인 시작")
    print("="*60)
    
    # ========== 1단계: 인덱싱 ==========
    print("\n📚 [1단계] 인덱싱 (Indexing)")
    print("-" * 60)
    
    # Load
    documents = load_documents()
    print(f"✅ 문서 로드 완료: {len(documents)}개")
    
    # Split
    chunks = split_documents(documents, chunk_size=200, chunk_overlap=50)
    
    # Embed & Store
    vectorstore = embed_and_store(chunks)
    
    # BM25 Retriever 생성
    bm25_retriever = create_bm25_retriever(chunks)
    
    # ========== 2단계: 검색 ==========
    print("\n🔍 [2단계] 검색 (Retrieve)")
    print("-" * 60)
    
    # Hybrid Search
    retrieved_docs = hybrid_search(query, vectorstore, bm25_retriever, top_k=10)
    
    # ReRanking (선택적)
    if use_reranker:
        retrieved_docs = rerank_documents(query, retrieved_docs, top_n=3)
    
    print(f"\n📄 검색된 문서 ({len(retrieved_docs)}개):")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n[{i}] {doc.page_content[:100]}...")
    
    # ========== 3단계: 생성 ==========
    print("\n💬 [3단계] 생성 (Generation)")
    print("-" * 60)
    
    # LLM 초기화
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
    )
    
    # 답변 생성
    answer = generate_answer(query, retrieved_docs, llm)
    
    print(f"\n❓ 질문: {query}")
    print(f"\n✅ 답변:\n{answer}")
    
    print("\n" + "="*60)
    print("✨ RAG 파이프라인 완료")
    print("="*60)
    
    return answer, retrieved_docs


# ============================================================================
# 실습 실행 코드
# ============================================================================

if __name__ == "__main__":
    # 실습 예제 질문들
    test_queries = [
        "RAG가 무엇인가요?",
        "하이브리드 검색은 어떻게 작동하나요?",
        "ReRanker의 역할은 무엇인가요?",
    ]
    
    # 첫 번째 질문으로 실습 실행
    query = test_queries[0]
    
    # RAG 파이프라인 실행
    answer, docs = run_rag_pipeline(query, use_reranker=True)
    
    print("\n" + "="*60)
    print("💡 다른 질문으로 테스트하려면:")
    print("   run_rag_pipeline('하이브리드 검색은 어떻게 작동하나요?')")
    print("="*60)
