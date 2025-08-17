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

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

if not GOOGLE_API_KEY or GOOGLE_API_KEY == "":
    print("❌ GOOGLE_API_KEY chưa được thiết lập!")
    GOOGLE_API_KEY = "dummy"
else:
    print(f"✅ GOOGLE_API_KEY loaded: {len(GOOGLE_API_KEY)} chars")
    if GOOGLE_API_KEY.startswith("AIza"):
        print("✅ API Key format valid")
    else:
        print("⚠️ API Key format may be invalid")

print("🚀 Khởi động Medical AI cho Hội Thầy thuốc trẻ Việt Nam...")

# Global variables
qa_chain = None
vector_db = None
initialization_status = "⚙️ Đang khởi tạo hệ thống..."
system_ready = False
total_files = 0
total_chunks = 0

def create_vector_db_with_timeout(chunks, embedding, timeout_seconds=120):
    """Tạo vector database với timeout protection"""
    
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
    """Khởi tạo hệ thống"""
    global qa_chain, vector_db, initialization_status, system_ready, total_files, total_chunks
    
    print("\n⚡ STARTING SYSTEM INITIALIZATION")
    print("=" * 50)
    
    try:
        # Step 1: Clean old data
        initialization_status = "🧹 Cleaning old data..."
        chroma_path = "chroma_db"
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)
            print("✅ Old database cleaned")
        
        # Step 2: Load documents
        initialization_status = "📂 Loading documents..."
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
        
        # Process all files but limit pages per file
        total_files = len(pdf_files)
        max_pages_per_file = 15 if total_files > 20 else 25
        
        print(f"📚 Processing {total_files} files, max {max_pages_per_file} pages each")
        initialization_status = f"📄 Loading {total_files} PDF files..."
        
        for i, file in enumerate(pdf_files):
            print(f"📄 Loading ({i+1}/{total_files}): {file}")
            try:
                loader = PyPDFLoader(os.path.join(data_folder, file))
                file_docs = loader.load()
                
                # Limit pages per file
                if len(file_docs) > max_pages_per_file:
                    file_docs = file_docs[:max_pages_per_file]
                    print(f"   ⚡ Using first {len(file_docs)} pages")
                
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
        
        print(f"✅ Total loaded: {len(docs)} pages from {total_files} files")
        
        # Step 3: Create chunks
        initialization_status = "✂️ Creating text chunks..."
        print("✂️ Creating optimized chunks...")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
        
        chunks = splitter.split_documents(docs)
        
        # Limit total chunks for performance
        max_chunks = 300
        if len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]
            print(f"⚡ Limited to {max_chunks} chunks for optimal performance")
        
        total_chunks = len(chunks)
        print(f"✅ Using {total_chunks} optimized chunks")
        
        # Step 4: Load embedding model
        initialization_status = "🔧 Loading embedding model..."
        print("🔧 Loading embedding model...")
        
        try:
            embedding = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✅ Embedding model loaded")
        except Exception as e:
            print(f"❌ Embedding model loading failed: {e}")
            initialization_status = f"❌ Embedding model error: {str(e)[:50]}..."
            return False
        
        # Step 5: Create vector database
        initialization_status = "💾 Building vector database..."
        print(f"💾 Building vector database ({total_chunks} chunks)...")
        
        vector_db, status = create_vector_db_with_timeout(chunks, embedding, timeout_seconds=120)
        
        if status == 'timeout':
            # Emergency mode
            emergency_chunks = chunks[:100]
            print(f"🚨 Emergency mode: Using only {len(emergency_chunks)} chunks")
            
            try:
                vector_db = Chroma.from_documents(
                    documents=emergency_chunks,
                    embedding=embedding,
                    persist_directory=None
                )
                total_chunks = len(emergency_chunks)
                print("✅ Emergency vector database created")
            except Exception as e:
                print(f"❌ Emergency vector DB also failed: {e}")
                initialization_status = f"❌ Vector DB failed: {str(e)[:50]}..."
                return False
                
        elif status != 'success':
            print(f"❌ Vector database creation failed: {status}")
            initialization_status = f"❌ Vector DB error: {status[:50]}..."
            return False
        
        # Step 6: Setup AI system
        if GOOGLE_API_KEY == "dummy":
            print("❌ API Key not configured")
            initialization_status = "❌ API Key not configured"
            return False
        
        initialization_status = "🤖 Setting up AI system..."
        print("🤖 Setting up Gemini AI...")
        
        try:
            prompt = PromptTemplate(
                template="""Bạn là trợ lý y tế AI chuyên nghiệp của Hội Thầy thuốc trẻ Việt Nam.

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
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.2,
                max_output_tokens=6144
            )
            
            # Test API
            print("   Testing API connection...")
            test_response = llm.invoke("Test connection")
            print(f"   ✅ API test successful: {test_response.content[:30]}...")
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vector_db.as_retriever(
                    search_kwargs={"k": 5}
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
            else:
                initialization_status = f"❌ LLM error: {str(llm_error)[:100]}..."
            
            return False
        
        # Success!
        print("\n" + "=" * 50)
        print("✅ SYSTEM INITIALIZATION COMPLETED!")
        print(f"📊 Statistics:")
        print(f"   • Files: {total_files}")
        print(f"   • Documents: {len(docs)} pages")
        print(f"   • Chunks: {total_chunks}")
        print(f"   • Vector DB: ✅ Ready")
        print(f"   • AI Model: ✅ Gemini 1.5 Pro")
        print("=" * 50)
        
        initialization_status = f"✅ Sẵn sàng! ({total_files} files, {total_chunks} chunks)"
        system_ready = True
        return True
        
    except Exception as e:
        print(f"\n❌ INITIALIZATION FAILED: {e}")
        initialization_status = f"❌ Error: {str(e)[:100]}..."
        import traceback
        traceback.print_exc()
        return False

def ask_question(query):
    """Xử lý câu hỏi từ người dùng"""
    global initialization_status, system_ready
    
    if not query or not query.strip():
        return f"❓ Vui lòng nhập câu hỏi.\n\n📊 Trạng thái hệ thống: {initialization_status}"
    
    query = query.strip()
    
    if len(query) > 1000:
        return "📝 Câu hỏi quá dài. Vui lòng rút ngắn dưới 1000 ký tự."
    
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
• Thời gian ước tính: 2-5 phút
• Hệ thống đang load và xử lý tài liệu y tế
• Vui lòng chờ và thử lại sau

🔄 Refresh trang và thử lại sau ít phút..."""
    
    try:
        print(f"🔍 Processing question: {query[:50]}...")
        
        if not hasattr(qa_chain, 'invoke'):
            return "❌ Hệ thống AI chưa được khởi tạo đúng cách. Vui lòng refresh trang và thử lại."
        
        start_time = time.time()
        result = qa_chain.invoke({"query": query})
        processing_time = time.time() - start_time
        
        print(f"✅ Question processed in {processing_time:.2f}s")
        
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
        
        # Add statistics
        answer += f"\n\n📊 **Thống kê:**"
        answer += f"\n• Files trong hệ thống: {total_files}"
        answer += f"\n• Chunks tham khảo: {len(sources)}"
        answer += f"\n• Thời gian xử lý: {processing_time:.1f}s"
        
        # Medical disclaimer
        answer += f"\n\n---\n⚠️ **Lưu ý quan trọng:** Thông tin trên chỉ mang tính chất tham khảo. Hãy tham khảo Thầy thuốc chuyên khoa để được chẩn đoán và điều trị chính xác. Trong trường hợp cấp cứu, hãy gọi 115."
        
        return answer
        
    except Exception as e:
        print(f"❌ Query processing error: {e}")
        error_msg = str(e).lower()
        
        if "quota" in error_msg or "limit" in error_msg:
            return """⚠️ Vượt quá giới hạn API.

📊 Chi tiết:
• Google AI Studio có giới hạn requests/phút cho free tier
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

🔍 Chi tiết lỗi: {str(e)[:200]}

💡 Các bước khắc phục:
• Thử lại sau vài phút
• Đặt câu hỏi khác hoặc diễn đạt lại
• Kiểm tra kết nối internet
• Liên hệ hỗ trợ nếu lỗi tiếp tục"""

def create_beautiful_interface():
    """Tạo giao diện đẹp và hiện đại"""
    
    # Custom CSS for beautiful interface
    custom_css = """
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        min-height: 100vh;
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="0.5"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
        opacity: 0.3;
    }
    
    .header-content {
        position: relative;
        z-index: 1;
        text-align: center;
        color: white;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(45deg, #ffffff, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        letter-spacing: -0.02em;
    }
    
    .sub-title {
        font-size: 1.2rem;
        margin: 0.5rem 0;
        opacity: 0.9;
        font-weight: 500;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1.5rem;
        padding: 1.5rem;
        background: rgba(255,255,255,0.1);
        border-radius: 0.75rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .feature-item {
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        background: rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    
    .feature-item:hover {
        background: rgba(255,255,255,0.2);
        transform: translateY(-2px);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    /* Card Styles */
    .info-card {
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(20px);
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.3);
        margin-bottom: 1rem;
    }
    
    .status-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 4px solid #0ea5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .api-status-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fed7aa 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Input Styles */
    .gradio-textbox {
        border-radius: 0.75rem !important;
        border: 2px solid rgba(59, 130, 246, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .gradio-textbox:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* Button Styles */
    .gradio-button {
        border-radius: 0.75rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        border: none !important;
    }
    
    .gradio-button.primary {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
    }
    
    .gradio-button.primary:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
        margin-top: 1rem;
    }
    
    .stat-item {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        border: 1px solid rgba(148, 163, 184, 0.2);
        transition: all 0.3s ease;
    }
    
    .stat-item:hover {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .stat-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e40af;
        display: block;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.25rem;
    }
    
    /* Footer Styles */
    .footer-section {
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(20px);
        border-radius: 1rem;
        padding: 2rem;
        margin-top: 2rem;
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #fca5a5;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
        position: relative;
    }
    
    .warning-box::before {
        content: '⚠️';
        font-size: 1.5rem;
        position: absolute;
        top: 1rem;
        left: 1rem;
    }
    
    .warning-content {
        margin-left: 2.5rem;
        color: #7f1d1d;
        line-height: 1.6;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 1.8rem;
        }
        
        .feature-grid {
            grid-template-columns: 1fr;
        }
        
        .stats-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* Animation */
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-slide-up {
        animation: slideInUp 0.6s ease-out;
    }
    """
    
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="sky",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter")
        ),
        css=custom_css,
        title="🏥 Hội Thầy thuốc trẻ Việt Nam - AI Medical Assistant"
    ) as interface:
        
        # BEAUTIFUL HEADER
        gr.HTML(f"""
        <div class="main-header animate-slide-up">
            <div class="header-content">
                <h1 class="main-title">
                    🏥 HỘI THẦY THUỐC TRẺ VIỆT NAM
                </h1>
                <p class="sub-title">
                    🤖 Trợ lý Y tế AI - Tư vấn sức khỏe thông minh 24/7
                </p>
                <p style="opacity: 0.8; margin: 0;">
                    Được phát triển bởi các Thầy thuốc trẻ Việt Nam
                </p>
                
                <div class="feature-grid">
                    <div class="feature-item">
                        <span class="feature-icon">🌐</span>
                        <strong>Website chính thức</strong><br>
                        <a href="https://thaythuoctre.vn" target="_blank" style="color: #fbbf24; text-decoration: none; font-weight: 600;">
                            thaythuoctre.vn
                        </a>
                    </div>
                    <div class="feature-item">
                        <span class="feature-icon">🤖</span>
                        <strong>AI Technology</strong><br>
                        <span style="color: #34d399; font-weight: 600;">Google Gemini Pro</span>
                    </div>
                    <div class="feature-item">
                        <span class="feature-icon">📚</span>
                        <strong>Data Coverage</strong><br>
                        <span style="color: #f87171; font-weight: 600;">{total_files} Files Ready</span>
                    </div>
                </div>
            </div>
        </div>
        """)
        
        # MAIN CONTENT AREA
        with gr.Row():
            with gr.Column(scale=2):
                # INPUT SECTION
                gr.HTML("""
                <div class="info-card animate-slide-up">
                    <h3 style="color: #1e40af; margin: 0 0 1rem 0; font-size: 1.5rem; font-weight: 700;">
                        💬 Đặt câu hỏi y tế
                    </h3>
                    <p style="color: #64748b; margin: 0; line-height: 1.5;">
                        Hãy mô tả chi tiết triệu chứng hoặc vấn đề sức khỏe để nhận được tư vấn chính xác nhất từ AI.
                    </p>
                </div>
                """)
                
                question_input = gr.Textbox(
                    lines=5,
                    placeholder="💬 Ví dụ: 'Tôi bị đau đầu kèm sốt nhẹ, có triệu chứng ho khan. Đây có thể là bệnh gì và cần làm gì?'",
                    label="🩺 Câu hỏi y tế của bạn",
                    max_lines=8,
                    show_label=True,
                    elem_classes=["question-input"]
                )
                
                with gr.Row():
                    submit_btn = gr.Button(
                        "🔍 Tư vấn với AI Doctor", 
                        variant="primary", 
                        size="lg",
                        scale=2,
                        elem_classes=["submit-button"]
                    )
                    clear_btn = gr.Button(
                        "🗑️ Xóa", 
                        variant="secondary", 
                        scale=1,
                        elem_classes=["clear-button"]
                    )
            
            with gr.Column(scale=1):
                # STATUS PANEL
                gr.HTML(f"""
                <div class="info-card animate-slide-up">
                    <div style="text-align: center; margin-bottom: 1.5rem;">
                        <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 1rem; font-size: 1.5rem; color: white; box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);">
                            🏥
                        </div>
                        <h3 style="color: #1e40af; margin: 0; font-size: 1.25rem; font-weight: 700;">
                            System Status
                        </h3>
                    </div>
                    
                    <div class="status-card">
                        <strong style="color: #0c4a6e;">📊 Trạng thái hệ thống:</strong><br>
                        <span style="color: #059669; font-weight: 600; font-size: 0.9rem;">
                            {initialization_status}
                        </span>
                    </div>
                    
                    <div class="api-status-card">
                        <strong style="color: #92400e;">🔑 API Status:</strong><br>
                        <span style="color: #78350f; font-weight: 600; font-size: 0.9rem;">
                            {"✅ Connected & Ready" if GOOGLE_API_KEY != "dummy" else "❌ Not configured"}
                        </span>
                    </div>
                    
                    <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #22c55e; margin-top: 1rem;">
                        <strong style="color: #14532d;">💡 Hướng dẫn sử dụng:</strong><br>
                        <ul style="color: #166534; font-size: 0.85rem; margin: 0.5rem 0 0 0; padding-left: 1rem; line-height: 1.4;">
                            <li>Mô tả triệu chứng chi tiết</li>
                            <li>Đề cập thời gian xuất hiện</li>
                            <li>Nêu độ tuổi và giới tính</li>
                            <li>Kể cả tiền sử bệnh (nếu có)</li>
                        </ul>
                    </div>
                    
                    <div class="stats-grid">
                        <div class="stat-item">
                            <span class="stat-number">24/7</span>
                            <span class="stat-label">Hỗ trợ</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-number">AI</span>
                            <span class="stat-label">Thông minh</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-number">VN</span>
                            <span class="stat-label">Tiếng Việt</span>
                        </div>
                    </div>
                </div>
                """)
        
        # OUTPUT SECTION
        gr.HTML("""
        <div class="info-card animate-slide-up" style="margin-top: 2rem;">
            <h3 style="color: #1e40af; margin: 0 0 1rem 0; font-size: 1.5rem; font-weight: 700;">
                🩺 Tư vấn từ AI Doctor
            </h3>
            <p style="color: #64748b; margin: 0; line-height: 1.5;">
                Câu trả lời chi tiết và chuyên nghiệp từ hệ thống AI y tế sẽ hiển thị ở đây.
            </p>
        </div>
        """)
        
        answer_output = gr.Textbox(
            lines=15,
            label="",
            show_copy_button=True,
            interactive=False,
            placeholder="🔄 Đang chờ câu hỏi từ bạn...\n\n💡 Mẹo: Hãy mô tả triệu chứng càng chi tiết càng tốt để nhận được tư vấn chính xác nhất.",
            show_label=False,
            elem_classes=["answer-output"]
        )
        
        # ENHANCED EXAMPLES SECTION
        gr.HTML("""
        <div class="info-card animate-slide-up" style="margin-top: 1.5rem;">
            <h3 style="color: #1e40af; margin: 0 0 1rem 0; font-size: 1.25rem; font-weight: 700;">
                💡 Câu hỏi mẫu - Click để thử ngay
            </h3>
            <p style="color: #64748b; margin: 0; line-height: 1.5; font-size: 0.9rem;">
                Chọn một trong những câu hỏi dưới đây để test khả năng của AI Doctor
            </p>
        </div>
        """)
        
        gr.Examples(
            examples=[
                "Tôi bị đau đầu kèm sốt nhẹ 37.5°C, có triệu chứng ho khan và mệt mỏi. Đây có thể là bệnh gì?",
                "Người tiểu đường type 2 nên ăn gì và tránh gì? Có thể tập thể dục không?",
                "Thuốc paracetamol uống như thế nào cho đúng? Có tác dụng phụ gì không?",
                "Trẻ 3 tuổi bị sốt cao 39°C, co giật. Cần xử lý cấp cứu như thế nào?",
                "Cách nhận biết dấu hiệu đột quỵ? Sơ cứu ban đầu làm gì?",
                "Phụ nữ mang thai nên tiêm vaccine gì? COVID-19 vaccine có an toàn không?",
                "Bị viêm gan B có thể lây nhiễm qua đường nào? Cách phòng ngừa?",
                "Dấu hiệu trầm cảm ở người trẻ? Khi nào cần đi khám bác sĩ?",
                "Nguyên tắc sử dụng kháng sinh: khi nào dùng, dùng bao lâu?",
                "Bị ngộ độc thực phẩm: triệu chứng và cách xử lý tại nhà?",
                "Chăm sóc da mặt bị mụn trứng cá: nguyên nhân và cách điều trị?",
                "Cao huyết áp ở người trẻ: nguyên nhân, triệu chứng và cách kiểm soát?"
            ],
            inputs=question_input,
            label="",
            examples_per_page=6
        )
        
        # BEAUTIFUL FOOTER
        gr.HTML("""
        <div class="footer-section animate-slide-up">
            <div style="text-align: center; margin-bottom: 2rem;">
                <div style="display: inline-flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                    <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; color: white; box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);">
                        🏥
                    </div>
                    <div style="text-align: left;">
                        <h4 style="margin: 0; color: #1e40af; font-size: 1.5rem; font-weight: 700;">
                            Hội Thầy thuốc trẻ Việt Nam
                        </h4>
                        <p style="margin: 0; color: #64748b; font-size: 1rem;">
                            Vietnam Young Physicians' Association
                        </p>
                    </div>
                </div>
            </div>
            
            <!-- System Information -->
            <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); padding: 1.5rem; border-radius: 1rem; margin-bottom: 1.5rem; border: 1px solid rgba(148, 163, 184, 0.2);">
                <h5 style="color: #1e40af; margin: 0 0 1rem 0; font-size: 1.25rem; font-weight: 600; text-align: center;">
                    🔧 Thông tin hệ thống
                </h5>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
                    <div style="background: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid rgba(59, 130, 246, 0.2);">
                        <strong style="color: #1e40af;">📊 Data Processing:</strong><br>
                        <span style="color: #64748b; font-size: 0.9rem; line-height: 1.4;">
                            • Auto-detect all PDF files<br>
                            • Smart page sampling<br>
                            • Optimized chunking<br>
                            • Vector database ready
                        </span>
                    </div>
                    <div style="background: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid rgba(34, 197, 94, 0.2);">
                        <strong style="color: #059669;">🤖 AI Features:</strong><br>
                        <span style="color: #64748b; font-size: 0.9rem; line-height: 1.4;">
                            • Google Gemini 1.5 Pro<br>
                            • Vietnamese language optimized<br>
                            • Medical knowledge base<br>
                            • Source document tracking
                        </span>
                    </div>
                    <div style="background: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid rgba(245, 158, 11, 0.2);">
                        <strong style="color: #d97706;">⚡ Performance:</strong><br>
                        <span style="color: #64748b; font-size: 0.9rem; line-height: 1.4;">
                            • Real-time responses<br>
                            • Timeout protection<br>
                            • Emergency fallback<br>
                            • Memory optimization
                        </span>
                    </div>
                </div>
            </div>
            
            <!-- Medical Disclaimer -->
            <div class="warning-box">
                <div class="warning-content">
                    <h5 style="color: #7f1d1d; margin: 0 0 0.5rem 0; font-size: 1.1rem; font-weight: 600;">
                        LƯU Ý Y KHOA QUAN TRỌNG
                    </h5>
                    <p style="margin: 0; line-height: 1.6;">
                        Thông tin tư vấn từ AI chỉ mang tính chất <strong>tham khảo</strong> và <strong>không thay thế</strong> 
                        cho việc khám bệnh, chẩn đoán và điều trị trực tiếp từ Thầy thuốc chuyên khoa.
                    </p>
                    <p style="margin: 0.5rem 0 0 0; line-height: 1.6;">
                        <strong>🏥 Hãy đến cơ sở y tế gần nhất</strong> khi có triệu chứng bất thường hoặc cần hỗ trợ y tế khẩn cấp.<br>
                        <strong>📞 Số điện thoại cấp cứu: 115</strong>
                    </p>
                </div>
            </div>
            
            <!-- Footer Links -->
            <div style="border-top: 2px solid rgba(148, 163, 184, 0.2); padding-top: 1.5rem; text-align: center;">
                <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin-bottom: 1rem;">
                    <a href="https://thaythuoctre.vn" target="_blank" style="color: #3b82f6; text-decoration: none; font-weight: 600; display: flex; align-items: center; gap: 0.5rem;">
                        🌐 Website chính thức
                    </a>
                    <a href="mailto:info@thaythuoctre.vn" style="color: #3b82f6; text-decoration: none; font-weight: 600; display: flex; align-items: center; gap: 0.5rem;">
                        📧 Liên hệ hỗ trợ
                    </a>
                </div>
                <div style="color: #94a3b8; font-size: 0.9rem; line-height: 1.5;">
                    <p style="margin: 0;">
                        🔒 Bảo mật dữ liệu | 🚀 Powered by Google Gemini AI | 🧠 Smart Medical Assistant | 🇻🇳 Made in Vietnam
                    </p>
                    <p style="margin: 0.5rem 0 0 0;">
                        © 2024 Hội Thầy thuốc trẻ Việt Nam. AI Medical Assistant v4.0
                    </p>
                </div>
            </div>
        </div>
        """)
        
        # EVENT HANDLERS
        submit_btn.click(
            fn=ask_question, 
            inputs=question_input, 
            outputs=answer_output,
            show_progress=True
        )
        question_input.submit(
            fn=ask_question, 
            inputs=question_input, 
            outputs=answer_output,
            show_progress=True
        )
        clear_btn.click(
            fn=lambda: ("", ""), 
            outputs=[question_input, answer_output]
        )
    
    return interface

# Create beautiful interface
print("🎨 Creating beautiful modern interface...")
interface = create_beautiful_interface()

# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 LAUNCHING BEAUTIFUL MEDICAL AI")
    print("=" * 60)
    print(f"📡 Server: 0.0.0.0:{port}")
    print(f"🔑 API Key: {'✅ Configured' if GOOGLE_API_KEY != 'dummy' else '❌ Missing'}")
    print(f"🎨 Interface: Beautiful Modern Design")
    print(f"🤖 AI Model: Google Gemini 1.5 Pro")
    print(f"⚡ Features: Responsive + Animated + Professional")
    print("=" * 60)
    
    # Start initialization
    print("🔥 Starting system initialization...")
    init_thread = threading.Thread(target=initialize_system, daemon=True)
    init_thread.start()
    
    time.sleep(0.5)
    
    # Launch interface
    try:
        print("🌟 Launching beautiful interface...")
        interface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            show_error=True,
            show_api=False,
            quiet=False,
            favicon_path=None,
            app_kwargs={"docs_url": None, "redoc_url": None}
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
