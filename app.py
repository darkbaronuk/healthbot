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
from queue import Queue

# Setup port cho Render
port = int(os.environ.get("PORT", 7860))
print(f"🔍 ENV PORT: {os.environ.get('PORT', 'Not set')}")
print(f"🔍 Using port: {port}")

# Load environment variables với improved handling
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

if not GOOGLE_API_KEY or GOOGLE_API_KEY == "":
    print("❌ GOOGLE_API_KEY chưa được thiết lập!")
    print("📝 Vui lòng kiểm tra Environment Variables trong Render Dashboard")
    GOOGLE_API_KEY = "dummy"
else:
    print(f"✅ GOOGLE_API_KEY loaded: {len(GOOGLE_API_KEY)} chars")
    if GOOGLE_API_KEY.startswith("AIza"):
        print("✅ API Key format valid")
    else:
        print("⚠️ API Key format may be invalid (should start with 'AIza')")

print("🚀 Khởi động Healthbot cho Hội Thầy thuốc trẻ Việt Nam...")

# Global variables
qa_chain = None
vector_db = None
initialization_status = "⚙️ Đang khởi tạo hệ thống..."
system_ready = False

def create_vector_db_with_timeout(chunks, embedding, timeout_seconds=120):
    """Tạo vector database với timeout protection sử dụng threading"""
    
    def worker(result_queue):
        try:
            start_time = time.time()
            print(f"💾 Creating vector database with {len(chunks)} chunks...")
            
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embedding,
                persist_directory=None
            )
            
            elapsed = time.time() - start_time
            result_queue.put(('success', vector_db, elapsed))
            
        except Exception as e:
            result_queue.put(('error', str(e), 0))
    
    result_queue = Queue()
    worker_thread = threading.Thread(target=worker, args=(result_queue,), daemon=True)
    
    worker_thread.start()
    worker_thread.join(timeout=timeout_seconds)
    
    if worker_thread.is_alive():
        print("⚠️ Vector database creation timeout - trying emergency mode")
        return None, "timeout"
    
    try:
        result_type, result_data, elapsed = result_queue.get_nowait()
        if result_type == 'success':
            print(f"✅ Vector database created in {elapsed:.1f}s")
            return result_data, 'success'
        else:
            print(f"❌ Vector database creation failed: {result_data}")
            return None, result_data
    except:
        return None, "queue_error"

def initialize_system():
    """Khởi tạo hệ thống tối ưu cho speed và stability"""
    global qa_chain, vector_db, initialization_status, system_ready
    
    print("\n⚡ STARTING OPTIMIZED INITIALIZATION")
    print("=" * 50)
    
    try:
        # Step 1: Clean old data
        initialization_status = "🧹 Cleaning old data..."
        chroma_path = "chroma_db"
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)
            print("✅ Old database cleaned")
        
        # Step 2: Load documents với smart limiting
        initialization_status = "📂 Smart document loading..."
        docs = []
        data_folder = "data"
        
        if not os.path.exists(data_folder):
            print(f"❌ Folder {data_folder} not found")
            initialization_status = "❌ Data folder not found"
            return False
        
        pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]
        if not pdf_files:
            print("❌ No PDF files found")
            initialization_status = "❌ No PDF files found"
            return False
        
        # Smart limiting: Chỉ load số lượng cần thiết
        max_files = 3  # Tăng lên 3 files
        limited_files = pdf_files[:max_files]
        print(f"📚 Processing {len(limited_files)} files for optimal performance")
        
        initialization_status = f"📄 Loading {len(limited_files)} PDF files..."
        
        for i, file in enumerate(limited_files):
            print(f"📄 Loading ({i+1}/{len(limited_files)}): {file}")
            try:
                loader = PyPDFLoader(os.path.join(data_folder, file))
                file_docs = loader.load()
                
                # Smart page limiting
                max_pages = 20  # Tăng lên 20 pages per file
                if len(file_docs) > max_pages:
                    file_docs = file_docs[:max_pages]
                    print(f"   ⚡ Using first {len(file_docs)} pages for speed")
                
                for doc in file_docs:
                    doc.metadata.update({
                        "source_file": file,
                        "page_count": len(file_docs),
                        "file_index": i
                    })
                
                docs.extend(file_docs)
                print(f"   ✅ Success: {len(file_docs)} pages")
                
            except Exception as e:
                print(f"   ❌ Error loading {file}: {e}")
                continue
        
        if not docs:
            print("❌ No documents loaded successfully")
            initialization_status = "❌ Failed to load documents"
            return False
        
        print(f"✅ Total loaded: {len(docs)} pages from {len(limited_files)} files")
        
        # Step 3: Optimized text splitting
        initialization_status = "✂️ Smart text chunking..."
        print("✂️ Creating optimized chunks...")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,        # Balanced for quality vs speed
            chunk_overlap=200,      # Good overlap for context
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
        
        chunks = splitter.split_documents(docs)
        
        # Progressive limiting based on chunk count
        if len(chunks) > 150:
            chunks = chunks[:150]
            print(f"⚡ Limited to 150 chunks for optimal performance")
        elif len(chunks) > 100:
            chunks = chunks[:100]
            print(f"⚡ Limited to 100 chunks for good performance")
        
        print(f"✅ Using {len(chunks)} optimized chunks")
        
        # Step 4: Fast embedding model
        initialization_status = "🔧 Loading embedding model..."
        print("🔧 Loading fast embedding model...")
        
        try:
            embedding = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # Fastest model
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✅ Fast embedding model loaded")
        except Exception as e:
            print(f"❌ Embedding model loading failed: {e}")
            initialization_status = f"❌ Embedding model error: {str(e)[:50]}..."
            return False
        
        # Step 5: Vector database với timeout protection
        initialization_status = "💾 Building vector database..."
        print(f"💾 Building vector database ({len(chunks)} chunks)...")
        
        vector_db, status = create_vector_db_with_timeout(chunks, embedding, timeout_seconds=120)
        
        if status == 'timeout':
            # Emergency mode: Use fewer chunks
            emergency_chunks = chunks[:50]
            print(f"🚨 Emergency mode: Using only {len(emergency_chunks)} chunks")
            
            try:
                vector_db = Chroma.from_documents(
                    documents=emergency_chunks,
                    embedding=embedding,
                    persist_directory=None
                )
                print("✅ Emergency vector database created")
            except Exception as e:
                print(f"❌ Emergency vector DB also failed: {e}")
                initialization_status = f"❌ Vector DB failed: {str(e)[:50]}..."
                return False
                
        elif status != 'success':
            print(f"❌ Vector database creation failed: {status}")
            initialization_status = f"❌ Vector DB error: {status[:50]}..."
            return False
        
        # Step 6: API Key validation
        if GOOGLE_API_KEY == "dummy":
            print("❌ API Key not configured")
            initialization_status = "❌ API Key not configured"
            return False
        
        # Step 7: Setup AI system
        initialization_status = "🤖 Setting up AI system..."
        print("🤖 Setting up Gemini AI...")
        
        try:
            # Create optimized prompt
            prompt = PromptTemplate(
                template="""Bạn là trợ lý y tế AI của Hội Thầy thuốc trẻ Việt Nam.

TÀI LIỆU THAM KHẢO:
{context}

CÂU HỎI: {question}

HƯỚNG DẪN TRẢ LỜI:
- Trả lời bằng tiếng Việt chính xác, chuyên nghiệp
- Dựa chủ yếu vào thông tin từ tài liệu được cung cấp
- Nếu không có thông tin trong tài liệu, nói rõ "Thông tin này chưa có trong tài liệu tham khảo"
- Đưa ra lời khuyên y tế cẩn trọng và khuyến khích tham khảo Thầy thuốc chuyên khoa
- Luôn nhắc nhở tầm quan trọng của việc khám bệnh trực tiếp

TRẢ LỜI:""",
                input_variables=["context", "question"]
            )
            
            # Create LLM
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.2,
                max_output_tokens=6144
            )
            
            # Quick API test
            print("   Testing API connection...")
            test_response = llm.invoke("Test connection")
            print(f"   ✅ API test successful: {test_response.content[:30]}...")
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vector_db.as_retriever(
                    search_kwargs={"k": 5}  # Return top 5 relevant chunks
                ),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            print("✅ QA chain created successfully")
            
        except Exception as llm_error:
            print(f"❌ LLM setup failed: {llm_error}")
            error_msg = str(llm_error).lower()
            
            if "api key" in error_msg or "authentication" in error_msg:
                initialization_status = "❌ API Key authentication failed"
            elif "quota" in error_msg or "limit" in error_msg:
                initialization_status = "❌ API quota exceeded"
            elif "permission" in error_msg:
                initialization_status = "❌ API Key insufficient permissions"
            else:
                initialization_status = f"❌ LLM error: {str(llm_error)[:100]}..."
            
            return False
        
        # Success!
        print("\n" + "=" * 50)
        print("✅ SYSTEM INITIALIZATION COMPLETED!")
        print(f"📊 Final stats:")
        print(f"   • Documents: {len(docs)} pages")
        print(f"   • Chunks: {len(chunks) if 'chunks' in locals() else 'N/A'}")
        print(f"   • Vector DB: {'✅ Ready' if vector_db else '❌ Failed'}")
        print(f"   • AI Model: ✅ Gemini 1.5 Pro")
        print("=" * 50)
        
        initialization_status = "✅ Sẵn sàng tư vấn y tế!"
        system_ready = True
        return True
        
    except Exception as e:
        print(f"\n❌ INITIALIZATION FAILED: {e}")
        initialization_status = f"❌ Error: {str(e)[:100]}..."
        import traceback
        traceback.print_exc()
        return False

def ask_question(query):
    """Xử lý câu hỏi từ người dùng với comprehensive error handling"""
    global initialization_status, system_ready
    
    # Input validation
    if not query or not query.strip():
        return f"❓ Vui lòng nhập câu hỏi.\n\n📊 Trạng thái hệ thống: {initialization_status}"
    
    query = query.strip()
    
    if len(query) > 1000:
        return "📝 Câu hỏi quá dài. Vui lòng rút ngắn dưới 1000 ký tự để đảm bảo chất lượng trả lời tốt nhất."
    
    # System readiness check
    if GOOGLE_API_KEY == "dummy":
        return """🔑 Lỗi API Key - Hệ thống chưa được cấu hình.

📝 Hướng dẫn khắc phục:
1. Truy cập Render Dashboard
2. Vào Settings → Environment
3. Thêm biến GOOGLE_API_KEY với giá trị từ Google AI Studio
4. Redeploy service sau khi cập nhật

💡 Lưu ý: API Key phải bắt đầu bằng 'AIza...'"""
    
    if not system_ready or not qa_chain:
        return f"""🔧 Hệ thống AI chưa sẵn sàng.

📊 Trạng thái hiện tại: {initialization_status}

💡 Thông tin:
• Thời gian ước tính: 1-3 phút tùy dung lượng dữ liệu
• Hệ thống đang load và xử lý tài liệu y tế
• Vui lòng chờ và thử lại sau

🔄 Refresh trang và thử lại sau ít phút..."""
    
    # Process question
    try:
        print(f"🔍 Processing question: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        # Validate qa_chain exists and is callable
        if not hasattr(qa_chain, 'invoke'):
            return "❌ Hệ thống AI chưa được khởi tạo đúng cách. Vui lòng refresh trang và thử lại."
        
        # Execute query
        start_time = time.time()
        result = qa_chain.invoke({"query": query})
        processing_time = time.time() - start_time
        
        print(f"✅ Question processed in {processing_time:.2f}s")
        
        # Extract and format answer
        answer = result.get("result", "Không thể tạo câu trả lời.")
        
        # Add source information
        sources = result.get("source_documents", [])
        if sources:
            source_files = set()
            for doc in sources:
                if "source_file" in doc.metadata:
                    source_files.add(doc.metadata["source_file"])
            
            if source_files:
                answer += f"\n\n📚 **Nguồn tài liệu tham khảo:** {', '.join(sorted(source_files))}"
        
        # Add medical disclaimer
        answer += f"\n\n---\n⚠️ **Lưu ý quan trọng:** Thông tin trên chỉ mang tính chất tham khảo. Hãy tham khảo Thầy thuốc chuyên khoa để được chẩn đoán và điều trị chính xác. Trong trường hợp cấp cứu, hãy gọi 115."
        
        return answer
        
    except Exception as e:
        print(f"❌ Query processing error: {e}")
        error_msg = str(e).lower()
        
        # Specific error handling
        if "quota" in error_msg or "limit" in error_msg:
            return """⚠️ Vượt quá giới hạn API.

📊 Chi tiết:
• Google AI Studio có giới hạn 15 requests/phút cho free tier
• Vui lòng chờ 1-2 phút và thử lại
• Hoặc nâng cấp lên paid plan để có quota cao hơn

⏰ Thử lại sau: 2-3 phút"""
            
        elif "safety" in error_msg:
            return """⚠️ Câu hỏi chứa nội dung được đánh giá là nhạy cảm.

💡 Khuyến nghị:
• Diễn đạt lại câu hỏi một cách rõ ràng và trực tiếp hơn
• Tập trung vào khía cạnh y tế/sức khỏe cụ thể
• Tránh các từ ngữ có thể gây hiểu lầm

🔄 Vui lòng thử đặt câu hỏi khác."""
            
        elif "api" in error_msg or "authentication" in error_msg:
            return """🔑 Lỗi xác thực API Key.

❌ Nguyên nhân có thể:
• API Key không đúng định dạng hoặc đã hết hạn
• Billing account chưa được kích hoạt trong Google Cloud
• Service bị vô hiệu hóa

🔗 Kiểm tra tại: https://console.cloud.google.com/apis/credentials"""
            
        else:
            return f"""❌ Có lỗi xảy ra khi xử lý câu hỏi.

🔍 Chi tiết lỗi: {str(e)[:200]}{'...' if len(str(e)) > 200 else ''}

💡 Các bước khắc phục:
• Thử lại sau vài phút
• Đặt câu hỏi khác hoặc diễn đạt lại
• Kiểm tra kết nối internet
• Liên hệ hỗ trợ nếu lỗi tiếp tục"""

def create_professional_interface():
    """Tạo giao diện chuyên nghiệp cho Hội Thầy thuốc trẻ Việt Nam"""
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        css="""
        .gradio-container { 
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
            font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
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
            transition: all 0.3s ease;
        }
        .stat-item:hover {
            background: #e0f2fe;
            transform: translateY(-2px);
        }
        @media (max-width: 768px) {
            .logo-section { 
                flex-direction: column; 
                gap: 15px; 
            }
            .custom-header { 
                padding: 25px 20px; 
            }
        }
        """,
        title="🏥 Hội Thầy thuốc trẻ Việt Nam - AI Medical Assistant"
    ) as interface:
        
        # PROFESSIONAL HEADER với logo và branding
        gr.HTML("""
        <div class="custom-header">
            <div class="logo-section">
                <div class="logo-circle">
                    <img src="http://thaythuoctre.vn/wp-content/uploads/2020/12/logo-ttt.png" 
                         alt="Logo Hội Thầy thuốc trẻ Việt Nam"
                         onerror="this.style.display='none'; this.parentElement.innerHTML='👨‍⚕️';">
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
                    placeholder="💬 Đặt câu hỏi về: triệu chứng bệnh, thuốc men, chế độ dinh dưỡng, sơ cứu, phòng bệnh, xét nghiệm...",
                    label="🩺 Câu hỏi y tế của bạn",
                    max_lines=6,
                    show_label=True,
                    info="Hãy mô tả chi tiết triệu chứng hoặc vấn đề sức khỏe để nhận được tư vấn chính xác nhất."
                )
                
                with gr.Row():
                    submit_btn = gr.Button(
                        "🔍 Tư vấn với Thầy thuốc AI", 
                        variant="primary", 
                        size="lg",
                        scale=2
                    )
                    clear_btn = gr.Button("🗑️ Xóa", variant="secondary", scale=1)
            
            with gr.Column(scale=1):
                gr.HTML(f"""
                <div class="info-card">
                    <div style="text-align: center; margin-bottom: 20px;">
                        <div style="width: 50px; height: 50px; background: white; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 10px; padding: 4px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                            <img src="http://thaythuoctre.vn/wp-content/uploads/2020/12/logo-ttt.png" 
                                 alt="Logo TTT" 
                                 style="width: 100%; height: 100%; object-fit: contain;"
                                 onerror="this.style.display='none'; this.parentElement.innerHTML='👨‍⚕️';">
                        </div>
                        <h3 style="color: #1e40af; margin: 0; font-size: 18px; font-weight: 700;">
                            Hội Thầy thuốc trẻ Việt Nam
                        </h3>
                    </div>
                    
                    <div style="margin-bottom: 20px;">
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
                        
                        <div style="background: #f1f5f9; padding: 15px; border-radius: 10px; border-left: 4px solid #1d4ed8; margin-bottom: 15px;">
                            <strong style="color: #1e40af;">📊 Trạng thái AI:</strong><br>
                            <span style="color: #059669; font-weight: 600;">
                                {initialization_status}
                            </span>
                        </div>
                        
                        <div style="background: #fef3c7; padding: 12px; border-radius: 8px; border-left: 4px solid #f59e0b;">
                            <strong style="color: #92400e;">🔑 API Status:</strong><br>
                            <span style="color: #78350f; font-weight: 600;">
                                {"✅ Connected" if GOOGLE_API_KEY != "dummy" else "❌ Not configured"}
                            </span>
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
        
        # OUTPUT SECTION
        answer_output = gr.Textbox(
            lines=12,
            label="🩺 Tư vấn từ Thầy thuốc AI",
            show_copy_button=True,
            interactive=False,
            placeholder="Câu trả lời chi tiết từ AI sẽ hiển thị ở đây...",
            info="Bạn có thể copy câu trả lời để lưu lại hoặc chia sẻ với Thầy thuốc."
        )
        
        # EXAMPLES SECTION
        gr.Examples(
            examples=[
                "Triệu chứng của bệnh tiểu đường type 2 là gì?",
                "Cách phòng ngừa bệnh cao huyết áp ở người trẻ?",
                "Thuốc paracetamol có tác dụng phụ gì? Liều dùng như thế nào?",
                "Chế độ ăn uống cho người bệnh tim mạch cần lưu ý gì?",
                "Cách sơ cứu ban đầu khi bị đột quỵ?",
                "Vaccine COVID-19 có an toàn không? Ai nên tiêm?",
                "Triệu chứng viêm gan B như thế nào? Cách phòng ngừa?",
                "Cách chăm sóc trẻ em bị sốt cao tại nhà?",
                "Dấu hiệu nhận biết bệnh trầm cảm ở người trẻ?",
                "Thuốc kháng sinh nên dùng như thế nào cho đúng?",
                "Triệu chứng và cách xử lý khi bị ngộ độc thực phẩm?",
                "Cách chăm sóc da mặt cho người bị mụn trứng cá?"
            ],
            inputs=question_input,
            label="💡 Câu hỏi mẫu - Click để thử ngay",
            examples_per_page=6
        )
        
        # PROFESSIONAL FOOTER
        gr.HTML("""
        <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); padding: 30px; border-radius: 20px; margin-top: 30px; border-top: 4px solid #1d4ed8; text-align: center;">
            <div style="display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 20px; flex-wrap: wrap;">
                <div style="width: 50px; height: 50px; background: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; padding: 4px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <img src="http://thaythuoctre.vn/wp-content/uploads/2020/12/logo-ttt.png" 
                         alt="Logo TTT" 
                         style="width: 100%; height: 100%; object-fit: contain;"
                         onerror="this.style.display='none'; this.parentElement.innerHTML='👨‍⚕️';">
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
                    ⚠️ LƯU Ý QUAN TRỌNG
                </p>
                <p style="color: #64748b; margin: 10px 0 0 0; line-height: 1.6;">
                    Thông tin tư vấn từ AI chỉ mang tính chất <strong>tham khảo</strong> và <strong>không thay thế</strong> 
                    cho việc khám bệnh, chẩn đoán và điều trị trực tiếp từ Thầy thuốc.<br>
                    <strong>Hãy đến cơ sở y tế gần nhất</strong> khi có triệu chứng bất thường hoặc cần hỗ trợ y tế khẩn cấp.<br>
                    <strong>Số điện thoại cấp cứu: 115</strong>
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
                    </a> | 
                    <a href="mailto:info@thaythuoctre.vn" style="color: #1d4ed8; text-decoration: none;">
                        📧 Liên hệ hỗ trợ
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

# Tạo professional interface
print("🎨 Creating professional interface...")
interface = create_professional_interface()

# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 LAUNCHING HEALTHBOT FOR HỘI THẦY THUỐC TRẺ VIỆT NAM")
    print("=" * 60)
    print(f"📡 Server: 0.0.0.0:{port}")
    print(f"🔑 API Key: {'✅ Configured' if GOOGLE_API_KEY != 'dummy' else '❌ Missing'}")
    print(f"🤖 AI Model: Google Gemini 1.5 Pro")
    print(f"⚡ Optimizations: Threading timeout + Smart limiting")
    print("=" * 60)
    
    # Start optimized background initialization
    print("🔥 Starting optimized initialization...")
    init_thread = threading.Thread(target=initialize_system, daemon=True)
    init_thread.start()
    
    # Small delay for thread to start
    time.sleep(0.5)
    
    # Launch interface
    try:
        print("🌟 Launching professional interface...")
        interface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            show_error=True,
            show_api=False,
            quiet=False
        )
        
    except Exception as e:
        print(f"❌ Primary launch failed: {e}")
        print("🔄 Attempting fallback launch...")
        
        try:
            interface.launch(
                server_name="0.0.0.0",
                server_port=port
            )
        except Exception as e2:
            print(f"❌ Fallback launch failed: {e2}")
            print("💔 Unable to start server. Check configuration and try again.")
            sys.exit(1)
