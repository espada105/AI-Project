"""
RAG 파이프라인 실습 예제
각 단계를 개별적으로 실행하여 학습할 수 있습니다.
"""

from rag_pipeline import (
    load_documents,
    split_documents,
    embed_and_store,
    create_bm25_retriever,
    hybrid_search,
    rerank_documents,
    generate_answer,
    run_rag_pipeline
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def example_1_indexing():
    """예제 1: 인덱싱 단계만 실습"""
    print("\n" + "="*60)
    print("📚 예제 1: 인덱싱 단계 실습")
    print("="*60)
    
    # Load
    documents = load_documents()
    print(f"\n✅ 로드된 문서 수: {len(documents)}")
    
    # Split
    chunks = split_documents(documents, chunk_size=200, chunk_overlap=50)
    print(f"✅ 생성된 청크 수: {len(chunks)}")
    print(f"\n첫 번째 청크 예시:\n{chunks[0].page_content[:200]}...")
    
    # Embed & Store
    vectorstore = embed_and_store(chunks)
    print(f"✅ 벡터 스토어 생성 완료")
    
    return vectorstore, chunks


def example_2_retrieval():
    """예제 2: 검색 단계만 실습"""
    print("\n" + "="*60)
    print("🔍 예제 2: 검색 단계 실습")
    print("="*60)
    
    # 인덱싱 먼저 수행
    vectorstore, chunks = example_1_indexing()
    
    # BM25 Retriever 생성
    bm25_retriever = create_bm25_retriever(chunks)
    
    # 하이브리드 검색 테스트
    query = "RAG가 무엇인가요?"
    results = hybrid_search(query, vectorstore, bm25_retriever, top_k=5)
    
    print(f"\n📄 검색 결과 ({len(results)}개):")
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] {doc.page_content[:150]}...")
    
    return results


def example_3_reranking():
    """예제 3: ReRanking 단계 실습"""
    print("\n" + "="*60)
    print("🔄 예제 3: ReRanking 단계 실습")
    print("="*60)
    
    # 검색 먼저 수행
    results = example_2_retrieval()
    
    # ReRanking 수행
    query = "RAG가 무엇인가요?"
    reranked = rerank_documents(query, results, top_n=3)
    
    print(f"\n📊 ReRanking 결과:")
    print(f"   원본: {len(results)}개 → ReRanked: {len(reranked)}개")
    
    for i, doc in enumerate(reranked, 1):
        print(f"\n[{i}] {doc.page_content[:150]}...")
    
    return reranked


def example_4_full_pipeline():
    """예제 4: 전체 파이프라인 실습"""
    print("\n" + "="*60)
    print("🚀 예제 4: 전체 RAG 파이프라인 실습")
    print("="*60)
    
    query = "하이브리드 검색은 어떻게 작동하나요?"
    answer, docs = run_rag_pipeline(query, use_reranker=True)
    
    return answer, docs


def example_5_comparison():
    """예제 5: ReRanker 사용 전/후 비교"""
    print("\n" + "="*60)
    print("⚖️  예제 5: ReRanker 사용 전/후 비교")
    print("="*60)
    
    query = "ReRanker의 역할은 무엇인가요?"
    
    # ReRanker 없이 실행
    print("\n[ReRanker 없이]")
    answer_without, docs_without = run_rag_pipeline(query, use_reranker=False)
    
    print("\n\n[ReRanker 사용]")
    answer_with, docs_with = run_rag_pipeline(query, use_reranker=True)
    
    print("\n" + "="*60)
    print("📊 비교 결과:")
    print(f"   ReRanker 없이: {len(docs_without)}개 문서 사용")
    print(f"   ReRanker 사용: {len(docs_with)}개 문서 사용")
    print("="*60)


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         RAG 파이프라인 실습 예제                         ║
    ╚══════════════════════════════════════════════════════════╝
    
    실행할 예제를 선택하세요:
    
    1. 인덱싱 단계만 실습 (Load → Split → Embed & Store)
    2. 검색 단계만 실습 (Hybrid Search)
    3. ReRanking 단계만 실습
    4. 전체 파이프라인 실습 (인덱싱 → 검색 → 생성)
    5. ReRanker 사용 전/후 비교
    
    """)
    
    # 전체 파이프라인 실행 (기본 예제)
    example_4_full_pipeline()
    
    # 다른 예제를 실행하려면 주석을 해제하세요:
    # example_1_indexing()
    # example_2_retrieval()
    # example_3_reranking()
    # example_5_comparison()
