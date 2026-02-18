"""
RAG 파이프라인 실습 - 메인 실행 파일
"""

from rag_pipeline import run_rag_pipeline

if __name__ == "__main__":
    # 실습 예제 질문
    query = "RAG가 무엇인가요?"
    
    # RAG 파이프라인 실행
    print("🚀 RAG 파이프라인을 시작합니다...\n")
    answer, docs = run_rag_pipeline(query, use_reranker=True)
    
    print("\n" + "="*60)
    print("💡 다른 질문으로 테스트하려면:")
    print("   run_rag_pipeline('하이브리드 검색은 어떻게 작동하나요?')")
    print("="*60)