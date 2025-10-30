from typing import List, Optional

from langchain_core.documents import Document

from src.rag.vector_store import VectorStore


class Retriever:
    """간단한 유사도 검색기"""

    def __init__(self,store: VectorStore | None = None, k: int = 3):
        self.store = store or VectorStore()
        self.k = k

    def retrieve(self, query: str, k: Optional[int] = None, threshold: float = 0.3) -> List[Document]:
        """질문과 가장 가까운 문서d와 점수를 찾아 반환합니다."""
        top_k = k if k is not None else self.k
        candidates = self.store.retrieve_with_scores(query, k=top_k)
        return [(doc, score) for doc, score in candidates if score >= threshold]

def test(query: str | None = None, top_k: int = 3):
    if query is None:
        import argparse

        parser = argparse.ArgumentParser(description="간단한 PDF 벡터 검색 CLI")
        parser.add_argument("query", help="찾고 싶은 질문/키워드")
        parser.add_argument("--top-k", type=int, default=3, help="가져올 문서 수")
        args = parser.parse_args()
        query, top_k = args.query, args.top_k

    retriever = Retriever(k=top_k)
    results = retriever.retrieve(query)

    if not results:
        print("XXX 결과가 없습니다. XXX")
        return
    
    print("="*30)
    print(f"'{query}' 관련 문서 {len(results)}개")
    for idx, (doc, score) in enumerate(results, start=1):   
        page = doc.metadata.get("page", "?")
        preview = doc.page_content.strip().replace("\n", " ")
        if len(preview) > 200:
            preview = preview[:200] + "..."
        print(f"\n[{idx}] page {page} | score={score:.4f}")
        print(preview)


if __name__ == "__main__":
    test("모바일로 주식 거래하는 앱은 뭐라고 하나요?")
    # test("듀레이션이라는게 뭔가요?")
    # test("ㅁㄴㅇㄹㄷㄴ")
