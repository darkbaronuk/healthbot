import os
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Load API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY chưa được thiết lập!")
    exit()

# 2. Tự động nạp toàn bộ PDF từ thư mục /data
pdf_folder = "data"
all_docs = []

print(f"📂 Đang quét file PDF trong thư mục: {pdf_folder}")
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_folder, filename)
        print(f"📄 Nạp: {filename}")
        loader = UnstructuredPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = filename
        all_docs.extend(docs)

print(f"✅ Đã nạp {len(all_docs)} trang tài liệu từ {len(os.listdir(pdf_folder))} file.")

# 3. Cắt nhỏ tài liệu
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(all_docs)
print(f"✂️ Đã chia thành {len(chunks)} đoạn văn bản.")

# 4. Tạo vector database (Chroma)
embedding = OpenAIEmbeddings()
from tqdm import tqdm  # hiển thị tiến trình

vector_db = Chroma(embedding_function=embedding, persist_directory="chroma_db")

# Thêm từng batch nhỏ (vd: mỗi batch 50 đoạn)
batch_size = 50
for i in tqdm(range(0, len(chunks), batch_size), desc="Đang tạo vector"):
    batch = chunks[i:i + batch_size]
    vector_db.add_documents(batch)

vector_db.persist()
print("✅ Đã lưu vector hóa vào thư mục chroma_db/")

# 5. Tạo prompt tiếng Việt chuyên ngành y tế
custom_prompt = PromptTemplate(
    template="""
Bạn là trợ lý y tế thông minh. Hãy trả lời dựa trên hướng dẫn chính thức của Bộ Y tế Việt Nam.

Dưới đây là các tài liệu tham khảo:

{context}

Câu hỏi: {question}

Trả lời bằng tiếng Việt, ngắn gọn, chính xác theo chuyên môn y tế. Nếu không chắc chắn, hãy nói rõ chưa có đủ thông tin.""",
    input_variables=["context", "question"]
)

# 6. Tạo mô hình hỏi đáp
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

# 7. Giao diện hỏi đáp cơ bản
print("\n🚑 Chatbot Y tế sẵn sàng! Gõ 'exit' để thoát.\n")
while True:
    query = input("🧑 Bạn hỏi: ")
    if query.strip().lower() in ["exit", "quit", "thoát"]:
        break
    result = qa_chain({"query": query})
    print("\n🤖 Trợ lý y tế trả lời:\n", result["result"])
    print("-" * 60)

