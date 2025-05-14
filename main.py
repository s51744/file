import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate

# 載入 .env（若需要）
load_dotenv()

# Streamlit 頁面設定
st.set_page_config(page_title="宜蘭大學行事曆查詢系統", page_icon="📅", layout="wide")
st.title("📚 國立宜蘭大學行事曆查詢系統")
st.markdown("自然語言提問，系統將自動查找行事曆內容並回答問題。")

# 初始化狀態
if "retriever_chain" not in st.session_state:
    st.session_state.retriever_chain = None
    st.session_state.pdf_loaded = False
    st.session_state.chat_history = []


# 初始化 RAG 系統
def initialize_rag_system():
    with st.spinner("正在初始化系統..."):
        pdf_path = "280557416.pdf"
        faiss_index_path = "faiss_index_local"

        if not os.path.exists(pdf_path):
            st.error(f"找不到 PDF 檔案：{pdf_path}")
            return False

        # 1. 載入 PDF
        loader = PyPDFLoader(file_path=pdf_path)
        documents = loader.load()

        # 2. 分割文本
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # 3. 使用本地 Embeddings（sentence-transformers）
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 4. 建立向量資料庫（FAISS）
        if os.path.exists(faiss_index_path):
            vectorstore = FAISS.load_local(
                faiss_index_path, embeddings, allow_dangerous_deserialization=True
            )
        else:
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(faiss_index_path)

        # 5. 自訂提示模板
        template = """你是一個問答助手，專門回答關於國立宜蘭大學行事曆的問題。
請根據以下檢索到的文件內容回答問題。
如果文件中沒有足夠的信息回答問題，請猜測用戶的需求進行回答，不要拒絕用戶的問題。

<檢索到的文件>
{context}
</檢索到的文件>

用戶問題: {input}
你的回答:"""
        prompt = ChatPromptTemplate.from_template(template)

        # 6. 使用本地模型（透過 Ollama）
        llm = ChatOllama(model="llama3.2", temperature=0)

        # 7. 創建文檔處理鏈與檢索鏈
        doc_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )
        st.session_state.retriever_chain = create_retrieval_chain(retriever, doc_chain)
        st.session_state.pdf_loaded = True
        return True


# 側邊欄
with st.sidebar:
    st.header("系統控制")
    if st.button("初始化系統"):
        if initialize_rag_system():
            st.success("初始化成功！可以開始提問。")
    st.divider()
    st.subheader("範例問題")
    for q in [
        "五月有哪些活動？",
        "什麼時候開學？",
        "中秋節放假嗎？",
        "期中考是什麼時候？",
    ]:
        if st.button(q):
            st.session_state.user_question = q
    st.divider()
    st.caption("📍 使用 Ollama 本地模型 + 本地向量資料庫")

# 主聊天區域
st.header("💬 提問")
if not st.session_state.pdf_loaded:
    st.warning("請先點擊『初始化系統』")

# 顯示對話歷史
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

# 提問區
user_question = st.chat_input(
    "請輸入您的問題...", disabled=not st.session_state.pdf_loaded
)
if user_question:
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            chain = st.session_state.retriever_chain
            if chain:
                res = chain.invoke({"input": user_question})
                answer = res.get("answer", "抱歉，我無法回答這個問題。")
                st.write(answer)
                st.session_state.chat_history.append((user_question, answer))
            else:
                st.error("系統未初始化")

st.divider()
st.caption("© 2025 宜蘭大學行事曆查詢系統 | 本地 RAG 問答")
