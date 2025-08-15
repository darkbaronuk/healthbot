import os
import sys
import gradio as gr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil  # For cleaning ChromaDB

# Setup port cho Render - ĐỪNG TEST PORT BINDING
port = int(os.environ.get("PORT", 7860))
print(f"🔍 ENV PORT: {os.environ.get('PORT', 'Not set')}")
print(f"🔍 Using port: {port}")
print(f"🔍 Python version: {sys.version}")

# Bỏ port binding test - có thể gây conflict
sys.stdout.flush()

# Load biến môi trường
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("❌ GOOGLE_API_KEY chưa được thiết lập!")
    print("🔑 Vui lòng thêm GOOGLE_API_KEY vào Environment Variables")
    GOOGLE_API_KEY = "dummy"  # Placeholder

print("🚀 Khởi động Medical Chatbot với Google Gemini...")

# Global variables
qa_chain = None
vector_db = None

def initialize_system():
    """Khởi tạo hệ thống AI"""
    global qa_chain, vector_db
    
    try:
        # Clean old ChromaDB if exists (fix column error)
        chroma_path = "chroma_db"
        if os.path.exists(chroma_path):
            print("🧹 Cleaning old ChromaDB...")
            shutil.rmtree(chroma_path)
            print("✅ Old database cleaned")
        
        # Load documents
        docs = []
        data_folder = "data"
        
        if os.path.exists(data_folder):
            print(f"📂 Quét thư mục {data_folder}...")
            pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]
            
            if pdf_files:
                for file in pdf_files:
                    print(f"📄 Đang tải: {file}")
                    try:
                        loader = PyPDFLoader(os.path.join(data_folder, file))
                        file_docs = loader.load()
                        for doc in file_docs:
                            doc.metadata["source_file"] = file
                        docs.extend(file_docs)
                        print(f"   ✅ Thành công: {len(file_docs)} trang")
                    except Exception as e:
                        print(f"   ❌ Lỗi tải {file}: {e}")
                        
                print(f"✅ Tổng cộng: {len(docs)} trang từ {len(pdf_files)} file")
            else:
                print(f"⚠️ Không có file PDF trong {data_folder}")
        else:
            print(f"⚠️ Thư mục {data_folder} không tồn tại")
        
        if docs and GOOGLE_API_KEY != "dummy":
            print("✂️ Chia nhỏ tài liệu...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)
            print(f"✅ Chia thành {len(chunks)} đoạn")
            
            print("🔧 Tạo embeddings...")
            embedding = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            print("💾 Tạo vector database...")
            try:
                vector_db = Chroma.from_documents(
                    chunks, 
                    embedding, 
                    persist_directory=None  # Use in-memory để tránh lỗi
                )
                print("✅ Vector database created successfully")
            except Exception as e:
                print(f"❌ ChromaDB error: {e}")
                # Fallback to simple setup
                vector_db = None
                return False
            
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
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3,
                max_output_tokens=8192
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vector_db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                ),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            print("✅ Hệ thống AI đã sẵn sàng!")
            return True
        else:
            print("⚠️ Không có tài liệu hoặc API key không hợp lệ")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi khởi tạo hệ thống: {e}")
        import traceback
        traceback.print_exc()
        return False

def ask_question(query):
    """Xử lý câu hỏi từ người dùng"""
    if not query.strip():
        return "❓ Vui lòng nhập câu hỏi."
    
    if len(query) > 1000:
        return "📝 Câu hỏi quá dài. Vui lòng rút ngắn dưới 1000 ký tự."
    
    if GOOGLE_API_KEY == "dummy":
        return "⚙️ Hệ thống chưa được cấu hình đúng. Vui lòng kiểm tra GOOGLE_API_KEY."
    
    if not qa_chain:
        return "🔧 Hệ thống AI chưa sẵn sàng. Vui lòng chờ khởi tạo hoặc kiểm tra tài liệu PDF."
    
    try:
        print(f"🔍 Xử lý câu hỏi: {query[:50]}...")
        result = qa_chain({"query": query})
        
        answer = result["result"]
        
        # Thêm thông tin nguồn
        sources = result.get("source_documents", [])
        if sources:
            source_files = set()
            for doc in sources:
                if "source_file" in doc.metadata:
                    source_files.add(doc.metadata["source_file"])
            
            if source_files:
                answer += f"\n\n📚 Nguồn tài liệu: {', '.join(source_files)}"
        
        return answer
        
    except Exception as e:
        error_msg = str(e).lower()
        if "quota" in error_msg or "limit" in error_msg:
            return "⚠️ Đã vượt quá giới hạn API (15 requests/phút). Vui lòng chờ và thử lại sau."
        elif "safety" in error_msg:
            return "⚠️ Câu hỏi có thể chứa nội dung nhạy cảm. Vui lòng diễn đạt lại câu hỏi."
        elif "api" in error_msg or "key" in error_msg:
            return "🔑 Lỗi API Key. Vui lòng kiểm tra cấu hình GOOGLE_API_KEY."
        else:
            return f"❌ Lỗi: {str(e)}\n\n💡 Vui lòng thử lại hoặc đặt câu hỏi khác."

# Tạo giao diện Gradio TRƯỚC KHI khởi tạo system
print("🎨 Tạo giao diện Gradio...")
interface = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Ví dụ: Triệu chứng của bệnh tiểu đường là gì?",
        label="💬 Câu hỏi y tế của bạn",
        max_lines=5
    ),
    outputs=gr.Textbox(
        lines=10,
        label="🩺 Trả lời từ trợ lý y tế",
        show_copy_button=True
    ),
    title="🏥 Trợ lý Y tế AI - Powered by Google Gemini",
    description="""
    🤖 **Chatbot y tế thông minh** dựa trên tài liệu chính thức của Bộ Y tế Việt Nam
    
    📊 **Trạng thái hệ thống:** ⚙️ Đang khởi tạo...
    
    ⚠️ **Lưu ý quan trọng:** 
    - Thông tin chỉ mang tính tham khảo
    - Không thay thế cho tư vấn y tế chuyên nghiệp  
    - Hãy tham khảo bác sĩ khi cần thiết
    - Giới hạn: 15 câu hỏi/phút (API miễn phí)
    """,
    examples=[
        "Triệu chứng của bệnh tiểu đường là gì?",
        "Cách phòng ngừa bệnh cao huyết áp?", 
        "Thuốc nào điều trị viêm họng?",
        "Chế độ ăn cho người bệnh tim mạch?",
        "Cách sơ cứu khi bị đột quỵ?"
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

# Launch ứng dụng NGAY để Render detect port
if __name__ == "__main__":
    print(f"🚀 Launching Gradio on port {port}")
    print(f"📡 Server binding: 0.0.0.0:{port}")
    sys.stdout.flush()
    
    # Launch trước, khởi tạo system sau
    import threading
    
    def init_background():
        """Khởi tạo system trong background"""
        print("⚙️ Đang khởi tạo hệ thống trong background...")
        system_ready = initialize_system()
        if system_ready:
            print("✅ Hệ thống sẵn sàng!")
        else:
            print("⚠️ Chạy ở chế độ demo")
    
    # Start background initialization
    init_thread = threading.Thread(target=init_background)
    init_thread.daemon = True
    init_thread.start()
    
    try:
        # Launch với parameters tối giản
        interface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            show_error=True
        )
    except Exception as e:
        print(f"❌ Launch failed: {e}")
        # Try with basic settings
        try:
            interface.launch(
                server_name="0.0.0.0",
                server_port=port
            )
        except Exception as e2:
            print(f"❌ Second launch attempt failed: {e2}")
            sys.exit(1)
