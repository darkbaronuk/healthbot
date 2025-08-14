import os
import gradio as gr
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader  # <-- ĐỔI SANG PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load biến môi trường
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Thiếu OPENAI_API_KEY")

print(f"🚀 Starting application...")

# Load và vector hóa tài liệu
docs = []
data_folder = "data"
if os.path.exists(data_folder):
    print(f"📂 Scanning {data_folder} folder...")
    for file in os.listdir(data_folder):
        if file.endswith(".pdf"):
            print(f"📄 Loading: {file}")
            try:
                loader = PyPDFLoader(os.path.join(data_folder, file))  # <-- SỬA ĐỔI
                docs.extend(loader.load())
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
    print(f"✅ Loaded {len(docs)} documents")
else:
    print(f"⚠️ {data_folder} folder not found")

if docs:
    print("✂️ Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    print("🔧 Creating embeddings...")
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
    
    print("🤖 Setting up QA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            model_name="gpt-4-turbo",
            max_tokens=200000,
            temperature=0.3
        ),
        retriever=vector_db.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )
    
    def ask(query):
        try:
            return qa_chain.run(query)
        except Exception as e:
            return f"Lỗi: {str(e)}. Vui lòng thử câu hỏi ngắn hơn."
else:
    def ask(query):
        return "Chưa có tài liệu PDF nào được tải lên."

# Lấy port từ biến môi trường
port = int(os.environ.get("PORT", 7860))
print(f"🚀 Starting server on port: {port}")

# Launch app
gr.Interface(
    fn=ask, 
    inputs=gr.Textbox(placeholder="Nhập câu hỏi y tế của bạn..."),
    outputs=gr.Textbox(),
    title="🏥 Trợ lý Y tế AI",
    description="Hỏi đáp về y tế dựa trên tài liệu chính thức"
).launch(
    server_name="0.0.0.0",
    server_port=port,
    share=False,
    show_error=True
)
