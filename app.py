import os
import gradio as gr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Đổi từ OPENAI_API_KEY
if not GOOGLE_API_KEY:
    raise ValueError("Thiếu GOOGLE_API_KEY")

print("🚀 Khởi động Medical Chatbot với Google Gemini...")

# Load documents
docs = []
data_folder = "data"
if os.path.exists(data_folder):
    print(f"📂 Quét thư mục {data_folder}...")
    for file in os.listdir(data_folder):
        if file.endswith(".pdf"):
            print(f"📄 Đang tải: {file}")
            try:
                loader = PyPDFLoader(os.path.join(data_folder, file))
                docs.extend(loader.load())
            except Exception as e:
                print(f"❌ Lỗi tải {file}: {e}")
    print(f"✅ Đã tải {len(docs)} trang tài liệu")
else:
    print(f"⚠️ Không tìm thấy thư mục {data_folder}")

if docs:
    print("✂️ Chia nhỏ tài liệu...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Chia thành {len(chunks)} đoạn")
    
    print("🔧 Tạo embeddings (miễn phí)...")
    # Dùng HuggingFace embeddings thay vì OpenAI (miễn phí)
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    print("💾 Tạo vector database...")
    vector_db = Chroma.from_documents(
        chunks, 
        embedding, 
        persist_directory="chroma_db"
    )
    
    # Prompt tối ưu cho Gemini
    prompt = PromptTemplate(
        template="""
Bạn là trợ lý y tế chuyên nghiệp của Bộ Y tế Việt Nam.

TÀI LIỆU THAM KHẢO:
{context}

CÂU HỎI: {question}

HƯỚNG DẪN TRẢ LỜI:
- Trả lời bằng tiếng Việt chính xác, chuyên nghiệp
- Dựa chủ yếu vào tài liệu được cung cấp
- Nếu không có thông tin trong tài liệu, hãy nói rõ "Thông tin này chưa có trong tài liệu tham khảo"
- Đưa ra lời khuyên y tế cẩn trọng và khuyến khích tham khảo bác sĩ khi cần

TRẢ LỜI:""",
        input_variables=["context", "question"]
    )
    
    print("🤖 Thiết lập Gemini AI...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",  # Model tốt nhất của Google
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
            max_output_tokens=8192  # Gemini có giới hạn khác
        ),
        retriever=vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Lấy 5 đoạn liên quan nhất
        ),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    def ask(query):
        if not query.strip():
            return "Vui lòng nhập câu hỏi."
        
        if len(query) > 1000:
            return "Câu hỏi quá dài. Vui lòng rút ngắn dưới 1000 ký tự."
        
        try:
            print(f"🔍 Đang xử lý: {query[:50]}...")
            result = qa_chain({"query": query})
            
            # Lấy câu trả lời
            answer = result["result"]
            
            # Thêm nguồn tài liệu (optional)
            sources = result.get("source_documents", [])
            if sources:
                source_files = set()
                for doc in sources:
                    if "source" in doc.metadata:
                        source_files.add(os.path.basename(doc.metadata["source"]))
                
                if source_files:
                    answer += f"\n\n📚 Nguồn tài liệu: {', '.join(source_files)}"
            
            return answer
            
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "limit" in error_msg:
                return "⚠️ Đã vượt quá giới hạn API. Vui lòng thử lại sau ít phút."
            elif "safety" in error_msg:
                return "⚠️ Câu hỏi có thể chứa nội dung nhạy cảm. Vui lòng diễn đạt khác."
            else:
                return f"❌ Lỗi: {str(e)}\nVui lòng thử lại hoặc đặt câu hỏi khác."
else:
    def ask(query):
        return "📋 Chưa có tài liệu PDF nào trong thư mục 'data'. Vui lòng thêm file PDF và khởi động lại."

# Lấy port từ biến môi trường (cho Render/Heroku)
port = int(os.environ.get("PORT", 7860))
print(f"🌐 Khởi động server trên port {port}")

# Tạo giao diện Gradio
interface = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Ví dụ: Triệu chứng của bệnh tiểu đường là gì?",
        label="💬 Câu hỏi y tế của bạn"
    ),
    outputs=gr.Textbox(
        lines=10,
        label="🩺 Trả lời từ trợ lý y tế",
        show_copy_button=True
    ),
    title="🏥 Trợ lý Y tế AI - Powered by Google Gemini",
    description="""
    🤖 Chatbot y tế thông minh dựa trên tài liệu chính thức của Bộ Y tế Việt Nam
    
    ⚠️ **Lưu ý quan trọng:** 
    - Thông tin chỉ mang tính tham khảo
    - Không thay thế cho tư vấn y tế chuyên nghiệp
    - Hãy tham khảo bác sĩ khi cần thiết
    """,
    examples=[
        "Triệu chứng của bệnh tiểu đường là gì?",
        "Cách phòng ngừa bệnh cao huyết áp?",
        "Thuốc nào điều trị viêm họng?",
        "Chế độ ăn cho người bệnh tim mạch?"
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

# Khởi động ứng dụng
print("✅ Medical Chatbot sẵn sàng!")
interface.launch(
    server_name="0.0.0.0",
    server_port=port,
    share=False,
    show_error=True,
    quiet=True
)
