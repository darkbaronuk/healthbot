import os
import sys
import gradio as gr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil
import threading
import time

# Setup port
port = int(os.environ.get("PORT", 7860))
print(f"🔍 ENV PORT: {os.environ.get('PORT', 'Not set')}")
print(f"🔍 Using port: {port}")

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("❌ GOOGLE_API_KEY chưa được thiết lập!")
    GOOGLE_API_KEY = "dummy"

print("🚀 Khởi động Medical Chatbot với Google Gemini...")

# Global variables
qa_chain = None
vector_db = None
initialization_status = "⚙️ Đang khởi tạo..."
system_ready = False

def initialize_system():
    """Khởi tạo hệ thống AI tối ưu cho Standard Plan (2GB RAM)"""
    global qa_chain, vector_db, initialization_status, system_ready
    
    print("🔄 FORCE INIT: Starting optimized system initialization for Standard Plan...")
    initialization_status = "📂 Đang quét thư mục PDF (Standard Plan - 2GB RAM)..."
    
    try:
        # Clean old ChromaDB
        chroma_path = "chroma_db"
        if os.path.exists(chroma_path):
            print("🧹 Cleaning old ChromaDB...")
            shutil.rmtree(chroma_path)
            print("✅ Old database cleaned")
        
        # Load documents
        docs = []
        data_folder = "data"
        initialization_status = "📄 Đang tải PDF files (tối ưu cho 2GB RAM)..."
        
        if os.path.exists(data_folder):
            print(f"📂 Quét thư mục {data_folder}...")
            pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]
            
            if pdf_files:
                # Process files with optimized settings for Standard Plan
                for file in pdf_files:
                    print(f"📄 Đang tải: {file}")
                    try:
                        loader = PyPDFLoader(os.path.join(data_folder, file))
                        file_docs = loader.load()
                        for doc in file_docs:
                            doc.metadata["source_file"] = file
                            doc.metadata["plan"] = "standard_2gb"
                        docs.extend(file_docs)
                        print(f"   ✅ Thành công: {len(file_docs)} trang")
                    except Exception as e:
                        print(f"   ❌ Lỗi tải {file}: {e}")
                        
                print(f"✅ Tổng cộng: {len(docs)} trang từ {len(pdf_files)} file")
            else:
                print(f"⚠️ Không có file PDF trong {data_folder}")
                initialization_status = "⚠️ Không tìm thấy PDF files"
                return False
        else:
            print(f"⚠️ Thư mục {data_folder} không tồn tại")
            initialization_status = "⚠️ Thư mục data không tồn tại"
            return False
        
        if docs and GOOGLE_API_KEY != "dummy":
            initialization_status = "✂️ Đang chia nhỏ tài liệu (tối ưu cho 2GB RAM)..."
            print("✂️ Chia nhỏ tài liệu với cấu hình tối ưu...")
            
            # Optimized settings for Standard Plan
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,    # Standard chunk size for 2GB RAM
                chunk_overlap=200,  # Good overlap for context
                length_function=len,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            chunks = splitter.split_documents(docs)
            print(f"✅ Chia thành {len(chunks)} đoạn")
            
            initialization_status = "🔧 Đang tạo embeddings (Standard Plan)..."
            print("🔧 Tạo embeddings với cấu hình tối ưu...")
            embedding = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            initialization_status = "💾 Đang tạo vector database (2GB RAM)..."
            print("💾 Tạo vector database...")
            try:
                vector_db = Chroma.from_documents(
                    chunks, 
                    embedding, 
                    persist_directory=None  # Use memory for better performance on Standard Plan
                )
                print("✅ Vector database created successfully")
            except Exception as e:
                print(f"❌ ChromaDB error: {e}")
                initialization_status = f"❌ Lỗi ChromaDB: {str(e)[:50]}..."
                return False
            
            initialization_status = "🤖 Đang thiết lập Gemini AI..."
            print("🤖 Thiết lập Gemini AI...")
            
            prompt = PromptTemplate(
                template="""
Bạn là trợ lý y tế AI chuyên nghiệp của Hội Thầy thuốc trẻ Việt Nam.

TÀI LIỆU THAM KHẢO:
{context}

CÂU HỎI: {question}

HƯỚNG DẪN TRẢ LỜI:
- Trả lời bằng tiếng Việt chính xác, chuyên nghiệp
- Dựa chủ yếu vào tài liệu được cung cấp
- Nếu không có thông tin trong tài liệu, hãy nói rõ "Thông tin này chưa có trong tài liệu tham khảo"
- Đưa ra lời khuyên y tế cẩn trọng và khuyến khích tham khảo Thầy thuốc khi cần

TRẢ LỜI:""",
                input_variables=["context", "question"]
            )
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3,
                max_output_tokens=8192
            )
            
            # Optimized retriever for Standard Plan
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vector_db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5, "fetch_k": 20}  # Better results with more candidates
                ),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            print("✅ Hệ thống AI đã sẵn sàng cho Standard Plan!")
            initialization_status = "✅ Sẵn sàng trả lời câu hỏi (Standard Plan - 2GB RAM)!"
            system_ready = True
            return True
        else:
            print("⚠️ Không có tài liệu hoặc API key không hợp lệ")
            initialization_status = "⚠️ API key không hợp lệ"
            return False
            
    except Exception as e:
        print(f"❌ Lỗi khởi tạo hệ thống: {e}")
        initialization_status = f"❌ Lỗi: {str(e)[:100]}..."
        import traceback
        traceback.print_exc()
        return False

def ask_question(query):
    """Xử lý câu hỏi từ người dùng"""
    global initialization_status, system_ready
    
    if not query.strip():
        return f"❓ Vui lòng nhập câu hỏi.\n\n📊 Trạng thái: {initialization_status}"
    
    if len(query) > 1000:
        return "📝 Câu hỏi quá dài. Vui lòng rút ngắn dưới 1000 ký tự."
    
    if GOOGLE_API_KEY == "dummy":
        return "⚙️ Hệ thống chưa được cấu hình đúng. Vui lòng kiểm tra GOOGLE_API_KEY."
    
    if not system_ready or not qa_chain:
        return f"""🔧 Hệ thống AI chưa sẵn sàng.

📊 Trạng thái hiện tại: {initialization_status}

💡 Hệ thống đang tối ưu cho Standard Plan (2GB RAM):
   • Load file PDF và tạo vector database
   • Khởi tạo AI model với cấu hình tối ưu
   • Ước tính thời gian: 1-2 phút

🔄 Vui lòng chờ và thử lại..."""
    
    try:
        print(f"🔍 Xử lý câu hỏi: {query[:50]}...")
        result = qa_chain.invoke({"query": query})
        
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

def create_thaythuoctre_interface():
    """Tạo interface với logo thật và font trắng cho Hội Thầy thuốc trẻ VN"""
    with gr.Blocks(
        theme=gr.themes.Soft(), 
        css="""
        .gradio-container { 
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }
        .custom-header {
            background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%);
            color: white;
            padding: 35px;
            border-radius: 20px;
            margin-bottom: 30px;
            box-shadow: 0 12px 40px rgba(29, 78, 216, 0.25);
        }
        .logo-section {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 25px;
            margin-bottom: 25px;
            flex-wrap: wrap;
        }
        .logo-circle {
            width: 85px;
            height: 85px;
            background: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            border: 3px solid rgba(255,255,255,0.3);
            padding: 8px;
        }
        .logo-circle img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .info-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.08);
            border-left: 5px solid #1d4ed8;
            margin-bottom: 20px;
        }
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .stat-item {
            background: #f8fafc;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e2e8f0;
        }
        @media (max-width: 768px) {
            .logo-section { flex-direction: column; gap: 15px; }
            .custom-header { padding: 25px 20px; }
        }
        """,
        title="🏥 Hội Thầy thuốc trẻ Việt Nam - AI Assistant"
    ) as interface:
        
        # CUSTOM HEADER VỚI LOGO THẬT VÀ FONT TRẮNG
        gr.HTML("""
        <div class="custom-header">
            <div class="logo-section">
                <div class="logo-circle">
                    <img src="http://thaythuoctre.vn/wp-content/uploads/2020/12/logo-ttt.png" 
                         alt="Logo Hội Thầy thuốc trẻ VN">
                </div>
                <div style="text-align: center;">
                    <h1 style="margin: 0; font-size: 32px; font-weight: 800; color: white; text-shadow: 2px 2px 6px rgba(0,0,0,0.3); letter-spacing: -0.5px;">
                        HỘI THẦY THUỐC TRẺ VIỆT NAM
                    </h1>
                    <p style="margin: 10px 0 0 0; font-size: 18px; color: white; opacity: 0.95; font-weight: 400;">
                        🤖 Trợ lý Y tế AI - Tư vấn sức khỏe thông minh 24/7
                    </p>
                    <p style="margin: 8px 0 0 0; font-size: 14px; color: white; opacity: 0.9;">
                        Được phát triển bởi các Thầy thuốc trẻ Việt Nam
                    </p>
                </div>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.2); backdrop-filter: blur(10px);">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; text-align: center;">
                    <div>
                        <div style="font-size: 24px; margin-bottom: 5px;">🌐</div>
                        <strong style="color: white;">Website chính thức</strong><br>
                        <a href="https://thaythuoctre.vn" target="_blank" style="color: #fbbf24; text-decoration: none; font-weight: 600;">
                            thaythuoctre.vn
                        </a>
                    </div>
                    <div>
                        <div style="font-size: 24px; margin-bottom: 5px;">🤖</div>
                        <strong style="color: white;">AI Technology</strong><br>
                        <span style="color: #34d399; font-weight: 600;">Google Gemini Pro</span>
                    </div>
                    <div>
                        <div style="font-size: 24px; margin-bottom: 5px;">📚</div>
                        <strong style="color: white;">Nguồn dữ liệu</strong><br>
                        <span style="color: #f87171; font-weight: 600;">Bộ Y tế Việt Nam</span>
                    </div>
                </div>
            </div>
        </div>
        """)
        
        # MAIN INTERFACE
        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    lines=4,
                    placeholder="💬 Đặt câu hỏi về: triệu chứng bệnh, thuốc men, chế độ dinh dưỡng, sơ cứu, phòng bệnh...",
                    label="🩺 Câu hỏi y tế của bạn",
                    max_lines=6,
                    show_label=True
                )
                
                with gr.Row():
                    submit_btn = gr.Button(
                        "🔍 Tư vấn với Thầy thuốc AI", 
                        variant="primary", 
                        size="lg"
                    )
                    clear_btn = gr.Button("🗑️ Xóa", variant="secondary")
            
            with gr.Column(scale=1):
                # THÔNG TIN HỘI VỚI LOGO THẬT
                gr.HTML(f"""
                <div class="info-card">
                    <div style="text-align: center; margin-bottom: 20px;">
                        <div style="width: 50px; height: 50px; background: white; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 10px; padding: 4px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                            <img src="http://thaythuoctre.vn/wp-content/uploads/2020/12/logo-ttt.png" 
                                 alt="Logo TTT" 
                                 style="width: 100%; height: 100%; object-fit: contain;">
                        </div>
                        <h3 style="color: #1e40af; margin: 0; font-size: 18px; font-weight: 700;">
                            Hội Thầy thuốc trẻ Việt Nam
                        </h3>
                    </div>
                    
                    <div style="space-y: 15px;">
                        <div style="margin-bottom: 15px;">
                            <strong style="color: #1e40af;">🌐 Website:</strong><br>
                            <a href="https://thaythuoctre.vn" target="_blank" style="color: #1d4ed8; text-decoration: none;">
                                thaythuoctre.vn
                            </a>
                        </div>
                        
                        <div style="margin-bottom: 15px;">
                            <strong style="color: #1e40af;">📧 Liên hệ:</strong><br>
                            <span style="color: #64748b;">info@thaythuoctre.vn</span>
                        </div>
                        
                        <div style="margin-bottom: 15px;">
                            <strong style="color: #1e40af;">🎯 Sứ mệnh:</strong><br>
                            <span style="color: #64748b; font-size: 14px;">
                                Nâng cao chất lượng chăm sóc sức khỏe<br>
                                và ứng dụng công nghệ trong y tế
                            </span>
                        </div>
                        
                        <div style="background: #f1f5f9; padding: 15px; border-radius: 10px; border-left: 4px solid #1d4ed8;">
                            <strong style="color: #1e40af;">📊 Trạng thái AI:</strong><br>
                            <span id="ai-status" style="color: #059669; font-weight: 600;">
                                {initialization_status}
                            </span>
                        </div>
                        
                        <div style="background: #e0f2fe; padding: 12px; border-radius: 8px; border-left: 4px solid #0891b2;">
                            <strong style="color: #0891b2;">🚀 Plan:</strong><br>
                            <span style="color: #0f766e; font-weight: 600;">Standard (2GB RAM)</span>
                        </div>
                    </div>
                    
                    <div class="stat-grid">
                        <div class="stat-item">
                            <div style="font-size: 20px; color: #1d4ed8; font-weight: 700;">24/7</div>
                            <div style="font-size: 12px; color: #64748b;">Hỗ trợ</div>
                        </div>
                        <div class="stat-item">
                            <div style="font-size: 20px; color: #059669; font-weight: 700;">AI</div>
                            <div style="font-size: 12px; color: #64748b;">Thông minh</div>
                        </div>
                        <div class="stat-item">
                            <div style="font-size: 20px; color: #dc2626; font-weight: 700;">VN</div>
                            <div style="font-size: 12px; color: #64748b;">Tiếng Việt</div>
                        </div>
                    </div>
                </div>
                """)
        
        # OUTPUT
        answer_output = gr.Textbox(
            lines=12,
            label="🩺 Tư vấn từ Thầy thuốc AI",
            show_copy_button=True,
            interactive=False,
            placeholder="Câu trả lời từ AI sẽ hiển thị ở đây..."
        )
        
        # EXAMPLES
        gr.Examples(
            examples=[
                "Triệu chứng của bệnh tiểu đường type 2 là gì?",
                "Cách phòng ngừa bệnh cao huyết áp ở người trẻ?",
                "Thuốc paracetamol có tác dụng phụ gì?",
                "Chế độ ăn uống cho người bệnh tim mạch?",
                "Cách sơ cứu ban đầu khi bị đột quỵ?",
                "Vaccine COVID-19 có an toàn không?",
                "Triệu chứng viêm gan B như thế nào?",
                "Cách chăm sóc trẻ em bị sốt cao?",
            ],
            inputs=question_input,
            label="💡 Câu hỏi mẫu - Click để thử ngay",
            examples_per_page=4
        )
        
        # FOOTER VỚI LOGO THẬT
        gr.HTML("""
        <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); padding: 30px; border-radius: 20px; margin-top: 30px; border-top: 4px solid #1d4ed8; text-align: center;">
            <div style="display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 20px; flex-wrap: wrap;">
                <div style="width: 50px; height: 50px; background: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; padding: 4px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <img src="http://thaythuoctre.vn/wp-content/uploads/2020/12/logo-ttt.png" 
                         alt="Logo TTT" 
                         style="width: 100%; height: 100%; object-fit: contain;">
                </div>
                <div>
                    <h4 style="margin: 0; color: #1e40af; font-size: 20px; font-weight: 700;">
                        Hội Thầy thuốc trẻ Việt Nam
                    </h4>
                    <p style="margin: 5px 0 0 0; color: #64748b; font-size: 14px;">
                        Vietnam Young Physicians' Association
                    </p>
                </div>
            </div>
            
            <div style="background: white; padding: 20px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
                <p style="color: #dc2626; margin: 0; font-weight: 600; font-size: 16px;">
                    ⚠️ LưU Ý QUAN TRỌNG
                </p>
                <p style="color: #64748b; margin: 10px 0 0 0; line-height: 1.6;">
                    Thông tin tư vấn từ AI chỉ mang tính chất <strong>tham khảo</strong> và <strong>không thay thế</strong> 
                    cho việc khám bệnh, tư vấn y tế trực tiếp từ Thầy thuốc.<br>
                    Hãy đến cơ sở y tế gần nhất khi có triệu chứng bất thường hoặc cần hỗ trợ y tế khẩn cấp.
                </p>
            </div>
            
            <div style="border-top: 1px solid #e2e8f0; padding-top: 20px; color: #94a3b8; font-size: 13px;">
                <p style="margin: 5px 0;">
                    🔒 Dữ liệu được bảo mật tuyệt đối | 🚀 Powered by Google Gemini AI | 🇻🇳 Made in Vietnam
                </p>
                <p style="margin: 5px 0;">
                    © 2024 Hội Thầy thuốc trẻ Việt Nam. Phát triển bởi các Thầy thuốc trẻ Việt Nam.
                </p>
                <p style="margin: 10px 0 0 0;">
                    <a href="https://thaythuoctre.vn" target="_blank" style="color: #1d4ed8; text-decoration: none;">
                        🌐 Truy cập website chính thức
                    </a>
                </p>
            </div>
        </div>
        """)
        
        # EVENT HANDLERS
        submit_btn.click(ask_question, inputs=question_input, outputs=answer_output)
        question_input.submit(ask_question, inputs=question_input, outputs=answer_output)
        clear_btn.click(lambda: ("", ""), outputs=[question_input, answer_output])
    
    return interface

# Tạo interface
interface = create_thaythuoctre_interface()

if __name__ == "__main__":
    print(f"🚀 Launching Gradio on port {port}")
    print(f"📡 Server binding: 0.0.0.0:{port}")
    print(f"💾 Optimized for Standard Plan (2GB RAM)")
    
    # FORCE start initialization BEFORE launch
    print("🔥 STARTING FORCED INITIALIZATION FOR STANDARD PLAN...")
    init_thread = threading.Thread(target=initialize_system)
    init_thread.daemon = True
    init_thread.start()
    
    # Small delay để thread bắt đầu
    time.sleep(0.5)
    
    try:
        interface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            show_error=True
        )
    except Exception as e:
        print(f"❌ Launch failed: {e}")
        try:
            interface.launch(
                server_name="0.0.0.0",
                server_port=port
            )
        except Exception as e2:
            print(f"❌ Second launch failed: {e2}")
            sys.exit(1)
