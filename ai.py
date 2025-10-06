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
        path = "교리/"

        text_loader_kwargs = {"autodetect_encoding": True}

        loader = DirectoryLoader(
            path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            silent_errors=True,
            loader_kwargs=text_loader_kwargs
        )

        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
        )

        split_docs = text_splitter.split_documents(docs)

        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        model_name = "BAAI/bge-m3"
        model_kwargs = {"device": device}
        encode_kwargs = {"normalize_embeddings": True}

        hf_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        db = FAISS.from_documents(split_docs, hf_embeddings)

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
                당신은 웨스트민스터 신앙고백과 대소요리문답에 해박한 장로교 교리 전문가입니다. 주어진 문서를 바탕으로 다음 질문에 대해 신학적으로 답변해주세요.\n\n
                [질문] : {question}\n\n
                [문서] : {document}\n\n
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