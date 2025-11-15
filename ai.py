import os

from dotenv import load_dotenv
import torch
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class RAGSystem:
    def __init__(self):
        self._setup_retriever()
        self._setup_chain()

    def _setup_retriever(self):
        # 임베딩 모델 우선 정의 (로드/생성 시 모두 필요)
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        model_name = "BAAI/bge-m3"
        model_kwargs = {"device": device}
        encode_kwargs = {"normalize_embeddings": True}

        hf_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        # 저장할 Vector DB 경로 설정
        DB_SAVE_PATH = "faiss_index_bge_m3"  # 사용한 임베딩 모델명 등을 넣어주면 좋습니다.

        if os.path.exists(DB_SAVE_PATH):
            # [빠른 로드] 저장된 DB가 있으면 불러오기
            print(f"'{DB_SAVE_PATH}'에서 기존 Vector DB를 로드합니다.")
            db = FAISS.load_local(
                DB_SAVE_PATH,
                hf_embeddings,
                # 최신 langchain에서 huggingface 임베딩을 로드할 때 필요할 수 있습니다.
                allow_dangerous_deserialization=True
            )

        else:
            # [최초 1회 실행] 저장된 DB가 없으면 새로 생성
            print(f"'{DB_SAVE_PATH}'를 찾을 수 없습니다. 새로운 Vector DB를 생성합니다.")

            path = "교리/"
            text_loader_kwargs = {"autodetect_encoding": True}
            loader = DirectoryLoader(
                path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                silent_errors=True,
                loader_kwargs=text_loader_kwargs
            )
            print("문서 로드 중...")
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=50,
            )
            print("문서 분할 중...")
            split_docs = text_splitter.split_documents(docs)

            print("Vector DB 생성 및 임베딩 중... (시간이 오래 걸릴 수 있습니다)")
            db = FAISS.from_documents(split_docs, hf_embeddings)

            print(f"Vector DB를 '{DB_SAVE_PATH}'에 저장합니다.")
            db.save_local(DB_SAVE_PATH)  # <- 핵심: DB를 파일로 저장

        # 리트리버 설정
        base_retriever = db.as_retriever(search_kwargs={"k": 10})

        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
        compressor = CrossEncoderReranker(model=model, top_n=3)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )

    def _setup_chain(self):
        load_dotenv()

        google_api_key = os.getenv("GOOGLE_AI_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_AI_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")

        template = """
                당신은 웨스트민스터 신앙고백과 대소요리문답에 해박한 장로교 교리 전문가입니다. 주어진 문서를 바탕으로 다음 질문에 대해 신학적으로 답변해주세요.
                이해하기 쉽도록 설명해주고, 당신이 참고한 문서는 말 끝에 글로 나열해서 표시해주세요\n\n
                #[질문] : {question}\n\n
                #[문서] : {document}\n\n
                """

        prompt = PromptTemplate(
            template=template,
            input_variables=["question", "document"],
        )

        llm = ChatGoogleGenerativeAI(model="models/gemma-3-27b-it", google_api_key=google_api_key)
        self.chain = prompt | llm

    def ask(self, question):
        similar_docs = self.retriever.invoke(question)
        similar_texts = '\n\n'.join([doc.page_content for doc in similar_docs])
        response = self.chain.stream({"question": question, "document": similar_texts})
        return response

if __name__ == "__main__":
    print("RAG 초기화 중... 잠시만 기다려주세요")
    rag_system = RAGSystem()
    print("당신의 신앙을 도와주는 AI입니다. 무엇이 궁금하신가요?")

    while True:
        question = input("\n질문(q or quit to exit) :")
        if question in ['q', 'quit']:
            break
        response = rag_system.ask(question)
        for token in response:
            print(token.content, end='', flush=True)