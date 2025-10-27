"""
메인 실행 파일
데이터 수집 → DB 구축 → 웹 배포까지 전체 파이프라인 실행
"""
import argparse
from src.utils.logger import Logger
from src.data.crawler import Crawler
from src.database.vector_store import VectorStore
from src.core.pipeline import RAGPipeline
from src.web.app import WebApp


class MainApp:
    """어플리케이션의 메인 실행 흐름을 제어"""
    
    def __init__(self, enable_evaluation: bool = False):
        self.logger = Logger()
        self.crawler = Crawler()
        self.vector_store = VectorStore()
        self.pipeline = None
        self.web_app = None
        self.enable_evaluation = enable_evaluation
    
    def setup_data(self):
        """데이터 수집 및 DB 구축"""
        self.logger.info("=== 데이터 수집 시작 ===")
        
        # 데이터 크롤링
        documents = self.crawler.crawl_all()
        
        # 벡터 DB 구축
        self.vector_store.build(documents)
        
        self.logger.info("=== 데이터 수집 완료 ===")
    
    def initialize_pipeline(self):
        """RAG 파이프라인 초기화"""
        self.logger.info("=== RAG 파이프라인 초기화 ===")
        self.pipeline = RAGPipeline(enable_evaluation=self.enable_evaluation)
        self.logger.info("=== 파이프라인 초기화 완료 ===")
    
    def launch_web(self, share: bool = False, server_port: int = 7860):
        """웹 애플리케이션 실행"""
        self.logger.info("=== 웹 애플리케이션 시작 ===")
        
        if self.pipeline is None:
            self.initialize_pipeline()
        
        # 웹 앱 생성 및 실행
        self.web_app = WebApp(self.pipeline)
        self.web_app.launch(share=share, server_port=server_port)
    
    def run_console(self):
        """콘솔 모드 실행 (테스트용)"""
        self.logger.info("=== 콘솔 모드 시작 ===")
        
        if self.pipeline is None:
            self.initialize_pipeline()
        
        while True:
            query = input("\n질문: ")
            if query.lower() in ['exit', 'quit', '종료']:
                break
            
            try:
                response = self.pipeline.query(query)
                print(f"\n답변: {response}")
            except Exception as e:
                self.logger.error(f"오류 발생: {e}", exc_info=True)
                print(f"오류가 발생했습니다: {str(e)}")
        
        self.logger.info("=== 콘솔 모드 종료 ===")
    
    def run(self, args: argparse.Namespace):
        """애플리케이션 실행"""
        try:
            # 1단계: 데이터 수집 (옵션)
            if not args.skip_setup:
                self.setup_data()
            
            if args.setup_only:
                self.logger.info("데이터 수집만 완료하고 종료합니다.")
                return
            
            # 2단계: 실행 모드 선택
            if args.console:
                self.run_console()
            else:
                self.launch_web(share=args.share, server_port=args.port)
            
        except Exception as e:
            self.logger.error(f"실행 중 오류 발생: {e}", exc_info=True)
            raise


def main():
    parser = argparse.ArgumentParser(description="Knowledge Base QA Engine")
    
    # 데이터 설정
    parser.add_argument(
        "--skip-setup", 
        action="store_true", 
        help="데이터 수집 단계 건너뛰기"
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="데이터 수집만 실행하고 종료"
    )
    
    # 웹 서버 설정
    parser.add_argument(
        "--share", 
        action="store_true", 
        help="Gradio 공유 링크 생성"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=7860, 
        help="웹 서버 포트"
    )
    
    # 실행 모드 설정
    parser.add_argument(
        "--console",
        action="store_true",
        help="웹이 아닌 콘솔 모드로 실행"
    )
    
    # 평가 설정
    parser.add_argument(
        "--enable-evaluation",
        action="store_true",
        help="응답 평가 기능 활성화"
    )
    
    args = parser.parse_args()
    
    app = MainApp(enable_evaluation=args.enable_evaluation)
    app.run(args)


if __name__ == "__main__":
    main()