import os
import gradio as gr
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load biến môi trường
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Thiếu OPENAI_API_KEY")

# Load và chia nhỏ tài liệu PDF
docs = []
for file in os.listdir("data"):
    if file.endswith(".pdf"):
        loader = UnstructuredPDFLoader(os.path.join("data", file))
        docs.extend(loader.load())

# Cắt nhỏ các đoạn văn để vector hóa
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Tạo embedding và vector database
embedding = OpenAIEmbeddings()
vector_db = Chroma.from_documents(chunks, embedding, persist_directory="chroma_db")

# Tạo prompt tiếng Việt
prompt = PromptTemplate(
    template="""
Bạn là trợ lý y tế thông minh. Dưới đây là tài liệu tham khảo chính thức từ Bộ Y tế:

{context}

Câu hỏi: {question}

Hãy trả lời bằng tiếng Việt, ngắn gọn và chính xác theo chuyên môn y tế.
""",
    input_variables=["context", "question"]
)

# Tạo chain hỏi – đáp
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
    retriever=vector_db.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# Hàm xử lý câu hỏi
def ask_bot(question):
    return qa_chain.run(question)

# Giao diện Gradio
gr.Interface(
    fn=ask_bot,
    inputs="text",
    outputs="text",
    title="Trợ lý Y tế AI",
    description="Hỏi đáp y tế bằng tiếng Việt, dựa trên tài liệu Bộ Y tế Việt Nam."
).launch(server_name="0.0.0.0", server_port=10000)
