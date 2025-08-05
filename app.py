import os
import gradio as gr
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load biến môi trường
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Thiếu OPENAI_API_KEY")

# Load và vector hóa tài liệu
docs = []
for file in os.listdir("data"):
    if file.endswith(".pdf"):
        loader = UnstructuredPDFLoader(os.path.join("data", file))
        docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

embedding = OpenAIEmbeddings()
vector_db = Chroma.from_documents(chunks, embedding, persist_directory="chroma_db")

prompt = PromptTemplate(
    template="""
Bạn là trợ lý y tế thông minh. Dưới đây là nội dung tài liệu Bộ Y tế:

{context}

Câu hỏi: {question}
Hãy trả lời bằng tiếng Việt ngắn gọn, chính xác.
""",
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
    retriever=vector_db.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

def ask(query):
    return qa_chain.run(query)

gr.Interface(fn=ask, inputs="text", outputs="text", title="Trợ lý Y tế AI").launch(
    server_name="0.0.0.0", server_port=10000
)
