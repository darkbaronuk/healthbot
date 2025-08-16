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

# Setup port cho Render
port = int(os.environ.get("PORT", 7860))
print(f"🔍 ENV PORT: {os.environ.get('PORT', 'Not set')}")
print(f"🔍 Using port: {port}")
print(f"💾 Optimized for Render Standard Plan (2GB RAM)")

# DEBUG: Kiểm tra environment trước khi load dotenv
print("\n🔍 DEBUG - BEFORE dotenv:")
print(f"   GOOGLE_API_KEY in os.environ: {'GOOGLE_API_KEY' in os.environ}")
raw_key = os.environ.get("GOOGLE_API_KEY", "NOT_FOUND")
print(f"   Raw environment value: '{raw_key}' (length: {len(raw_key)})")

# Load environment variables
load_dotenv()

# DEBUG: Kiểm tra sau khi load dotenv
print("🔍 DEBUG - AFTER dotenv:")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
print(f"   getenv result: '{GOOGLE_API_KEY}' (type: {type(GOOGLE_API_KEY)}, length: {len(GOOGLE_API_KEY)})")

# FIXED LOGIC: Kiểm tra API Key đúng cách
if not GOOGLE_API_KEY or GOOGLE_API_KEY == "" or GOOGLE_API_KEY.lower() == "none":
    print("❌ GOOGLE_API_KEY chưa được thiết lập hoặc rỗng!")
    print(f"   Actual value: '{GOOGLE_API_KEY}'")
    print("   ⚠️ Vui lòng kiểm tra Environment Variables trong Render Dashboard")
    print("   📝 Cần thiết lập: GOOGLE_API_KEY=AIza...")
    GOOGLE_API_KEY = "dummy"
else:
    print(f"✅ GOOGLE_API_KEY loaded successfully:")
    print(f"   Length: {len(GOOGLE_API_KEY)} characters")
    print(f"   Preview: {GOOGLE_API_KEY[:10]}...{GOOGLE_API_KEY[-4:]}")
    
    # Validate API Key format
    if not GOOGLE_API_KEY.startswith("AIza"):
        print(f"⚠️ WARNING: API Key không bắt đầu bằng 'AIza': {GOOGLE_API_KEY[:10]}...")
        print("   → Có thể không phải là Google API Key hợp lệ")
    else:
        print("✅ API Key format validation passed")

print(f"🔑 Final API Key status: {'✅ Valid' if GOOGLE_API_KEY != 'dummy' else '❌ Invalid/Missing'}")
print("=" * 60)
print("🚀 Khởi động Medical Chatbot cho Hội Thầy thuốc trẻ Việt Nam...")

# Global variables
qa_chain = None
vector_db = None
initialization_status = "⚙️ Đang khởi tạo hệ thống..."
system_ready = False

def test_api_key():
    """Test API Key trước khi sử dụng"""
    if GOOGLE_API_KEY == "dummy":
        return False, "API Key chưa được cấu hình"
    
    try:
        print("🔍 Testing Google API Key...")
        test_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1
        )
        
        # Test với câu đơn giản
        response = test_llm.invoke("Hello")
        print(f"✅ API Key test successful: {response.content[:50]}...")
        return True, "API Key hoạt động bình thường"
        
    except Exception as e:
        error_msg = str(e).lower()
        print(f"❌ API Key test failed: {e}")
        
        if "api key" in error_msg or "invalid" in error_msg:
            return False, "API Key không hợp lệ hoặc sai format"
        elif "quota" in error_msg or "limit" in error_msg:
            return False, "API Key đã vượt quá giới hạn quota"
        elif "authentication" in error_msg:
            return False, "Lỗi xác thực API Key"
        elif "permission" in error_msg:
            return False, "API Key không có quyền truy cập"
        else:
            return False, f"Lỗi không xác định: {str(e)[:100]}"

def initialize_system():
    """Khởi tạo hệ thống AI tối ưu cho Standard Plan (2GB RAM)"""
    global qa_chain, vector_db, initialization_status, system_ready
    
    print("\n🔄 STARTING SYSTEM INITIALIZATION FOR STANDARD PLAN (2GB RAM)")
    print("=" * 60)
    
    # Verify API Key trước tiên
    initialization_status = "🔑 Đang kiểm tra API Key..."
    api_valid, api_message = test_api_key()
    
    if not api_valid:
        print(f"❌ API Key validation failed: {api_message}")
        initialization_status = f"❌ API Key lỗi: {api_message}"
        return False
    
    print(f"✅ API Key validation passed: {api_message}")
    
    try:
        # Clean old ChromaDB
        initialization_status = "🧹 Dọn dẹp database cũ..."
        chroma_path = "chroma_db"
        if os.path.exists(chroma_path):
            print("🧹 Cleaning old ChromaDB...")
            shutil.rmtree(chroma_path)
            print("✅ Old database cleaned")
        
        # Load documents
        initialization_status = "📂 Đang quét thư mục dữ liệu..."
        docs = []
        data_folder = "data"
        
        if not os.path.exists(data_folder):
            print(f"❌ Thư mục {data_folder} không tồn tại")
            initialization_status = "❌ Thư mục data không tồn tại"
            return False
        
        print(f"📂 Scanning folder: {data_folder}")
        pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]
        
        if not pdf_files:
            print(f"❌ Không tìm thấy file PDF trong {data_folder}")
            initialization_status = "❌ Không tìm thấy file PDF"
            return False
        
        print(f"📚 Found {len(pdf_files)} PDF files")
        initialization_status = f"📄 Đang tải {len(pdf_files)} file PDF..."
        
        # Load PDF files
        for i, file in enumerate(pdf_files):
            print(f"📄 Loading ({i+1}/{len(pdf_files)}): {file}")
            try:
                loader = PyPDFLoader(os.path.join(data_folder, file))
                file_docs = loader.load()
                
                # Add metadata
                for doc in file_docs:
                    doc.metadata.update({
                        "source_file": file,
                        "plan": "standard_2gb",
                        "loaded_at": time.time()
                    })
                
                docs.extend(file_docs)
                print(f"   ✅ Success: {len(file_docs)} pages")
                
            except Exception as e:
                print(f"   ❌ Error loading {file}: {e}")
                continue
        
        if not docs:
            print("❌ Không có tài liệu nào được tải thành công")
            initialization_status = "❌ Không thể tải tài liệu"
            return False
        
        print(f"✅ Total loaded: {len(docs)} pages from {len(pdf_files)} files")
        
        # Text splitting với cấu hình tối ưu cho Standard Plan
        initialization_status = "✂️ Đang chia nhỏ tài liệu..."
        print("✂️ Splitting documents with optimized settings...")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,        # Optimal cho 2GB RAM
            chunk_overlap=200,      # Good context overlap
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        chunks = splitter.split_documents(docs)
        print(f"✅ Created {len(chunks)} chunks (avg: {sum(len(c.page_content) for c in chunks)//len(chunks)} chars)")
        
        # Create embeddings
        initialization_status = "🔧 Đang tạo embeddings..."
        print("🔧 Creating embeddings with multilingual model...")
        
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ Embedding model loaded")
        
        # Create vector database
        initialization_status = "💾 Đang tạo vector database..."
        print("💾 Creating vector database...")
        
        try:
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embedding,
                persist_directory=None  # Use memory cho performance tốt hơn trên Standard Plan
            )
            print("✅ Vector database created successfully")
            
        except Exception as e:
            print(f"❌ ChromaDB creation failed: {e}")
            initialization_status = f"❌ Lỗi tạo vector database: {str(e)[:50]}..."
            return False
        
        # Setup QA chain
        initialization_status = "🤖 Đang thiết lập Gemini AI..."
        print("🤖 Setting up Gemini AI...")
        
        # Double-check API key
        current_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not current_api_key or current_api_key == "dummy":
            print("❌ API Key bị mất trong quá trình khởi tạo")
            initialization_status = "❌ API Key bị mất"
            return False
        
        try:
            # Create prompt template
            prompt = PromptTemplate(
                template="""Bạn là trợ lý y tế AI chuyên nghiệp của Hội Thầy thuốc trẻ Việt Nam.

TÀI LIỆU THAM KHẢO:
{context}

CÂU HỎI: {question}

HƯỚNG DẪN TRẢ LỜI:
- Trả lời bằng tiếng Việt chính xác, chuyên nghiệp
- Dựa chủ yếu vào thông tin từ tài liệu được cung cấp
- Nếu không có thông tin cụ thể trong tài liệu, hãy nói rõ "Thông tin này chưa có trong tài liệu tham khảo"
- Đưa ra lời khuyên y tế cẩn trọng và khuyến khích tham khảo Thầy thuốc chuyên khoa khi cần
- Luôn đề cập đến việc tham khảo ý kiến Thầy thuốc cho chẩn đoán và điều trị chính xác

TRẢ LỜI:""",
                input_variables=["context", "question"]
            )
            
            # Create LLM
            print(f"   Creating LLM with API key: {current_api_key[:10]}...{current_api_key[-4:]}")
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                google_api_key=current_api_key,
                temperature=0.2,  # Slightly lower for more consistent medical advice
                max_output_tokens=8192
            )
            
            # Test LLM
            print("   Testing LLM connection...")
            test_response = llm.invoke("Test connection")
            print(f"   ✅ LLM test successful: {test_response.content[:30]}...")
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vector_db.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": 5,         # Return top 5 relevant chunks
                        "fetch_k": 20   # Search in top 20 candidates
                    }
                ),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            print("✅ QA chain created successfully")
            
        except Exception as llm_error:
            print(f"❌ LLM/QA chain creation failed: {llm_error}")
            error_msg = str(llm_error).lower()
            
            if "api key" in error_msg or "authentication" in error_msg:
                initialization_status = "❌ Lỗi xác thực API Key"
            elif "quota" in error_msg or "limit" in error_msg:
                initialization_status = "❌ Vượt quá giới hạn API quota"
            elif "permission" in error_msg:
                initialization_status = "❌ API Key không có quyền truy cập"
            else:
                initialization_status = f"❌ Lỗi tạo AI model: {str(llm_error)[:100]}..."
            
            return False
        
        # Success!
        print("\n" + "=" * 60)
        print("✅ HỆ THỐNG ĐÃ SẴN SÀNG!")
        print(f"📊 Thống kê:")
        print(f"   • {len(docs)} trang tài liệu")
        print(f"   • {len(chunks)} chunks")
        print(f"   • Vector database: In-memory")
        print(f"   • AI Model: Gemini 1.5 Pro")
        print(f"   • Plan: Standard (2GB RAM)")
        print("=" * 60)
        
        initialization_status = "✅ Sẵn sàng tư vấn y tế!"
        system_ready = True
        return True
        
    except Exception as e:
        print(f"\n❌ SYSTEM INITIALIZATION FAILED: {e}")
        initialization_status = f"❌ Lỗi khởi tạo: {str(e)[:100]}..."
        import traceback
        traceback.print_exc()
        return False

def ask_question(query):
    """Xử lý câu hỏi từ người dùng với error handling toàn diện"""
    global initialization_status, system_ready
    
    # Basic validation
    if not query or not query.strip():
        return f"❓ Vui lòng nhập câu hỏi.\n\n📊 Trạng thái hệ thống: {initialization_status}"
    
    query = query.strip()
    
    if len(query) > 1000:
        return "📝 Câu hỏi quá dài. Vui lòng rút ngắn dưới 1000 ký tự để đảm bảo chất lượng trả lời tốt nhất."
    
    # Check API Key
    current_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not current_api_key or current_api_key == "dummy":
        return """🔑 Lỗi API Key - Hệ thống chưa được cấu hình đúng.

🔍 Thông tin debug:
• GOOGLE_API_KEY không được tìm thấy hoặc có giá trị rỗng
• Vui lòng kiểm tra Environment Variables trong Render Dashboard

📝 Hướng dẫn kiểm tra:
1. Truy cập Render Dashboard
2. Chọn service healthbot
3. Vào Settings → Environment
4. Verify biến GOOGLE_API_KEY tồn tại và có giá trị bắt đầu bằng 'AIza'
5. Redeploy service sau khi cập nhật

💡 Lưu ý: API Key phải là Google AI Studio API Key hợp lệ."""
    
    # Check system readiness
    if not system_ready or not qa_chain:
        return f"""🔧 Hệ thống AI chưa sẵn sàng.

📊 Trạng thái hiện tại: {initialization_status}

🔑 API Key: {"✅ Đã cấu hình" if current_api_key != "dummy" else "❌ Chưa cấu hình"}

💡 Thông tin hệ thống:
• Cấu hình: Standard Plan (2GB RAM)
• Đang xử lý: Tải tài liệu PDF và tạo vector database
• Thời gian ước tính: 1-3 phút (tùy thuộc vào dung lượng dữ liệu)

🔄 Vui lòng chờ hệ thống khởi tạo hoàn tất và thử lại..."""
    
    # Process question
    try:
        print(f"🔍 Processing question: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        # Validate qa_chain
        if not hasattr(qa_chain, 'invoke'):
            return "❌ Hệ thống AI chưa được khởi tạo đúng cách. Vui lòng thử lại sau."
        
        # Query the system
        start_time = time.time()
        result = qa_chain.invoke({"query": query})
        processing_time = time.time() - start_time
        
        print(f"✅ Question processed in {processing_time:.2f}s")
        
        # Extract answer
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
        
        # Add disclaimer
        answer += f"\n\n---\n⚠️ **Lưu ý:** Thông tin trên chỉ mang tính chất tham khảo. Hãy tham khảo Thầy thuốc chuyên khoa để được chẩn đoán và điều trị chính xác."
        
        return answer
        
    except Exception as e:
        print(f"❌ Question processing error: {e}")
        error_msg = str(e).lower()
        
        # Specific error handling
        if "api key" in error_msg or "authentication" in error_msg:
            return """🔑 Lỗi xác thực API Key.

❌ Nguyên nhân có thể:
• API Key không đúng định dạng
• API Key đã bị vô hiệu hóa
• Billing account chưa được kích hoạt
• Service chưa được enable trong Google Cloud Console

🔗 Kiểm tra tại: https://console.cloud.google.com/apis/credentials
💡 Đảm bảo Google AI Studio API được kích hoạt và có quota khả dụng."""
            
        elif "quota" in error_msg or "limit" in error_msg:
            return """⚠️ Đã vượt quá giới hạn API.

📊 Chi tiết:
• Google AI Studio có giới hạn 15 requests/phút cho free tier
• Vui lòng chờ ít phút và thử lại
• Hoặc nâng cấp lên paid plan để có quota cao hơn

⏰ Thử lại sau: 1-2 phút"""
            
        elif "safety" in error_msg:
            return """⚠️ Câu hỏi có thể chứa nội dung nhạy cảm.

💡 Vui lòng:
• Diễn đạt lại câu hỏi một cách rõ ràng hơn
• Tránh các từ ngữ có thể gây hiểu lầm
• Tập trung vào khía cạnh y tế/sức khỏe

🔄 Thử đặt câu hỏi khác hoặc diễn đạt lại."""
            
        else:
            return f"""❌ Có lỗi xảy ra khi xử lý câu hỏi.

🔍 Chi tiết lỗi: {str(e)[:200]}{'...' if len(str(e)) > 200 else ''}

💡 Khuyến nghị:
• Thử lại sau ít phút
• Đặt câu hỏi khác
• Kiểm tra kết nối internet
• Liên hệ hỗ trợ nếu lỗi tiếp tục xảy ra"""

def create_thaythuoctre_interface():
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
        
        # HEADER với logo thật và branding chuyên nghiệp
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
                # Input section
                question_input = gr.Textbox(
                    lines=4,
                    placeholder="💬 Đặt câu hỏi về: triệu chứng bệnh, thuốc men, chế độ dinh dưỡng, sơ cứu, phòng bệnh, xét nghiệm...",
                    label="🩺 Câu hỏi y tế của bạn",
                    max_lines=6,
                    show_label=True,
                    info="Hãy mô tả chi tiết triệu chứng hoặc vấn đề sức khỏe để nhận được tư vấn chính xác nhất."
                )
                
                # Buttons
                with gr.Row():
                    submit_btn = gr.Button(
                        "🔍 Tư vấn với Thầy thuốc AI", 
                        variant="primary", 
                        size="lg",
                        scale=2
                    )
                    clear_btn = gr.Button("🗑️ Xóa", variant="secondary", scale=1)
            
            with gr.Column(scale=1):
                # Thông tin tổ chức và trạng thái
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
                            <span id="ai-status" style="color: #059669; font-weight: 600;">
                                {initialization_status}
                            </span>
                        </div>
                        
                        <div style="background: #e0f2fe; padding: 12px; border-radius: 8px; border-left: 4px solid #0891b2; margin-bottom: 15px;">
                            <strong style="color: #0891b2;">🚀 Cấu hình:</strong><br>
                            <span style="color: #0f766e; font-weight: 600;">Standard Plan (2GB RAM)</span>
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
                "Thuốc kháng sinh nên dùng như thế nào cho đúng?"
            ],
            inputs=question_input,
            label="💡 Câu hỏi mẫu - Click để thử ngay",
            examples_per_page=5
        )
        
        # FOOTER
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

# Tạo interface
print("🎨 Creating professional interface...")
interface = create_thaythuoctre_interface()

# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 LAUNCHING HEALTHBOT FOR HỘI THẦY THUỐC TRẺ VIỆT NAM")
    print("=" * 60)
    print(f"📡 Server: 0.0.0.0:{port}")
    print(f"💾 Plan: Standard (2GB RAM)")
    print(f"🔑 API Key: {'✅ Configured' if GOOGLE_API_KEY != 'dummy' else '❌ Missing'}")
    print(f"🤖 AI Model: Google Gemini 1.5 Pro")
    print("=" * 60)
    
    # Start background initialization
    print("🔥 Starting background initialization...")
    init_thread = threading.Thread(target=initialize_system, daemon=True)
    init_thread.start()
    
    # Small delay to let thread start
    time.sleep(0.5)
    
    # Launch interface
    try:
        print("🌟 Launching Gradio interface...")
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
