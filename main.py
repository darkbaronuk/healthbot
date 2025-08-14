import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# 1. Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")  # Đổi từ OPENAI_API_KEY
if not api_key:
    print("❌ GOOGLE_API_KEY chưa được thiết lập!")
    print("🔑 Hướng dẫn lấy key: https://makersuite.google.com/app/apikey")
    exit()

print("✅ Google API Key đã được tải")

# 2. Tự động nạp toàn bộ PDF từ thư mục /data
pdf_folder = "data"
all_docs = []
print(f"📂 Đang quét file PDF trong thư mục: {pdf_folder}")

if not os.path.exists(pdf_folder):
    print(f"❌ Thư mục {pdf_folder} không tồn tại!")
    print(f"📁 Tạo thư mục {pdf_folder} và thêm file PDF vào đó")
    os.makedirs(pdf_folder)
    exit()

pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
if not pdf_files:
    print(f"❌ Không có file PDF nào trong thư mục {pdf_folder}")
    print("📄 Vui lòng thêm file PDF vào thư mục data/")
    exit()

for filename in pdf_files:
    file_path = os.path.join(pdf_folder, filename)
    print(f"📄 Đang nạp: {filename}")
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = filename
        all_docs.extend(docs)
        print(f"   ✅ Thành công: {len(docs)} trang")
    except Exception as e:
        print(f"   ❌ Lỗi: {e}")

print(f"✅ Tổng cộng đã nạp {len(all_docs)} trang từ {len(pdf_files)} file PDF.")

if not all_docs:
    print("❌ Không có tài liệu nào được tải thành công!")
    exit()

# 3. Cắt nhỏ tài liệu
print("✂️ Đang chia nhỏ tài liệu...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(all_docs)
print(f"✅ Đã chia thành {len(chunks)} đoạn văn bản.")

# 4. Tạo vector database với HuggingFace embeddings (miễn phí)
print("🔧 Đang tải mô hình embedding (lần đầu có thể mất vài phút)...")
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("💾 Đang tạo vector database...")
# Kiểm tra xem đã có database chưa
if os.path.exists("chroma_db") and os.listdir("chroma_db"):
    print("📂 Tìm thấy database cũ, đang tải...")
    vector_db = Chroma(
        embedding_function=embedding, 
        persist_directory="chroma_db"
    )
    print("✅ Đã tải database có sẵn")
else:
    print("🆕 Tạo database mới...")
    vector_db = Chroma(
        embedding_function=embedding, 
        persist_directory="chroma_db"
    )
    
    # Thêm từng batch để tránh lỗi memory
    batch_size = 50
    for i in tqdm(range(0, len(chunks), batch_size), desc="Đang vector hóa"):
        batch = chunks[i:i + batch_size]
        vector_db.add_documents(batch)
    
    vector_db.persist()
    print("✅ Đã lưu vector database vào thư mục chroma_db/")

# 5. Tạo prompt tiếng Việt chuyên ngành y tế cho Gemini
custom_prompt = PromptTemplate(
    template="""
Bạn là trợ lý y tế chuyên nghiệp của Bộ Y tế Việt Nam. Hãy trả lời dựa trên các tài liệu chính thức.

TÀI LIỆU THAM KHẢO:
{context}

CÂU HỎI: {question}

HƯỚNG DẪN TRẢ LỜI:
- Trả lời bằng tiếng Việt chính xác, chuyên nghiệp
- Dựa chủ yếu vào thông tin trong tài liệu
- Nếu không có thông tin cụ thể, hãy nói rõ "Thông tin này chưa có trong tài liệu"
- Luôn khuyến khích tham khảo ý kiến bác sĩ chuyên khoa khi cần thiết
- Đưa ra thông tin y tế cẩn trọng và có trách nhiệm

TRẢ LỜI:""",
    input_variables=["context", "question"]
)

# 6. Tạo mô hình hỏi đáp với Google Gemini
print("🤖 Đang khởi tạo Google Gemini...")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # Model tốt nhất hiện tại
    google_api_key=api_key,
    temperature=0.3,  # Giảm tính ngẫu nhiên cho câu trả lời y tế
    max_output_tokens=8192,
    safety_settings={
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE", 
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE"
    }
)

retriever = vector_db.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 5}  # Lấy 5 đoạn liên quan nhất
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

print("✅ Hệ thống đã sẵn sàng!")

# 7. Giao diện hỏi đáp với error handling
def ask_question(query):
    """Xử lý câu hỏi với error handling"""
    try:
        result = qa_chain({"query": query})
        answer = result["result"]
        
        # Thêm thông tin nguồn
        sources = result.get("source_documents", [])
        if sources:
            source_files = set()
            for doc in sources:
                source_file = doc.metadata.get("source_file", "Unknown")
                source_files.add(source_file)
            
            if source_files:
                answer += f"\n\n📚 Nguồn tài liệu: {', '.join(source_files)}"
        
        return answer
        
    except Exception as e:
        error_msg = str(e).lower()
        if "quota" in error_msg or "limit" in error_msg:
            return "⚠️ Đã vượt quá giới hạn API (15 requests/phút). Vui lòng chờ và thử lại."
        elif "safety" in error_msg:
            return "⚠️ Câu hỏi có thể chứa nội dung nhạy cảm. Vui lòng diễn đạt lại."
        else:
            return f"❌ Lỗi: {str(e)}\nVui lòng thử lại hoặc đặt câu hỏi khác."

# 8. Vòng lặp hỏi đáp
print("\n" + "="*60)
print("🏥 TRỢ LÝ Y TẾ AI - POWERED BY GOOGLE GEMINI")
print("="*60)
print("💡 Mẹo sử dụng:")
print("   - Đặt câu hỏi cụ thể về y tế")
print("   - Có thể hỏi về triệu chứng, điều trị, phòng ngừa")
print("   - Gõ 'exit', 'quit' hoặc 'thoát' để kết thúc")
print("   - Giới hạn: 15 câu hỏi/phút")
print("\n⚠️  LƯU Ý: Thông tin chỉ mang tính tham khảo, không thay thế tư vấn y tế chuyên nghiệp!")
print("="*60)

question_count = 0
while True:
    try:
        query = input(f"\n🧑 Câu hỏi #{question_count + 1}: ").strip()
        
        if query.lower() in ["exit", "quit", "thoát", "q"]:
            print("👋 Cảm ơn bạn đã sử dụng Trợ lý Y tế AI!")
            break
        
        if not query:
            print("❌ Vui lòng nhập câu hỏi.")
            continue
            
        if len(query) > 1000:
            print("❌ Câu hỏi quá dài. Vui lòng rút ngắn dưới 1000 ký tự.")
            continue
        
        print(f"\n🔍 Đang tìm kiếm thông tin liên quan...")
        answer = ask_question(query)
        
        print("\n🩺 TRẢ LỜI:")
        print("-" * 50)
        print(answer)
        print("-" * 50)
        
        question_count += 1
        
        # Cảnh báo giới hạn API
        if question_count % 10 == 0:
            print(f"\n💡 Bạn đã hỏi {question_count} câu. Nhớ giới hạn 15 câu/phút của Gemini API.")
        
    except KeyboardInterrupt:
        print("\n\n👋 Đã dừng chương trình. Cảm ơn bạn!")
        break
    except Exception as e:
        print(f"\n❌ Lỗi không mong muốn: {e}")
        print("🔄 Vui lòng thử lại...")

print("\n🔚 Chương trình kết thúc.")
