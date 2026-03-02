# src/web_ui.py
import streamlit as st
from pathlib import Path

from config import settings
from data_processor import VideoProcessor
from rag_engine import VideoRAG

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.retrievers import ContextualCompressionRetriever

from langchain_community.vectorstores import FAISS
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker

# ==========================================
# 1. Global UI Configuration
# ==========================================
st.set_page_config(page_title="Video-RAG System", page_icon="🎬", layout="wide")

st.markdown("""
    <style>
    .stButton>button { border-radius: 8px; font-weight: bold; }
    .stTextInput>div>div>input { border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Sidebar Settings & Parameters
# ==========================================
with st.sidebar:
    st.header("⚙️ Engine Configuration")
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=60) 
    st.markdown("**LLM:** `Qwen2.5:7b` (Local)")
    st.markdown("**Embedding:** `BGE-m3`")
    
    st.divider()
    st.subheader("RAG Parameters")
    # Dynamic parameter for top_k retrieval control
    top_k = st.slider("Target Chunks (Top-K)", min_value=1, max_value=10, value=5)
    
    st.divider()
    st.caption("Powered by LangChain & Streamlit")

# ==========================================
# 3. Core Engine Initialization
# ==========================================
@st.cache_resource
def load_engines():
    """Cache the initialization of NLP models to prevent reloading on UI refresh."""
    return VideoProcessor(), VideoRAG()

processor, rag_engine = load_engines()

st.title("🎬 Video-RAG Multi-Modal QA System")
col1, col2 = st.columns([1, 1.2]) 

# ==========================================
# 4. Left Panel: Knowledge Base Management
# ==========================================
with col1:
    st.subheader("📺 Knowledge Base Status")
    
    # Load the pre-compiled global vector store containing all video chunks
    index_name = "multi_video_kb"
    index_path = str(settings.INDEX_DIR / f"{index_name}_faiss")
    
    if Path(index_path).exists():
        st.success(f"✅ Global FAISS Index detected: `{index_name}`")
        if st.button("🚀 Initialize Global RAG Engine", use_container_width=True, type="primary"):
            with st.spinner("🧠 Loading FAISS index and BGE Embedding model into memory..."):
                vectorstore = FAISS.load_local(
                    index_path, 
                    rag_engine.embeddings, 
                    allow_dangerous_deserialization=True
                )
                st.session_state['vectorstore'] = vectorstore
                st.success("🎉 Engine Ready. You can now query across all processed videos.")
    else:
        st.error(f"❌ Global Index `{index_name}_faiss` not found. Please run `python main.py` first.")

# ==========================================
# 5. Right Panel: Interactive QA & Retrieval
# ==========================================
with col2:
    st.subheader("💬 Interactive Assistant")
    
    # Initialize chat history state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Render previous chat history with source traceability
    for msg in st.session_state.messages:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("🔍 View Retrieved Sources"):
                    for src in msg["sources"]:
                        st.caption(f"📺 **Source: `{src.get('source', 'Unknown')}.mp4`** | ⏱️ [{src['start']}s - {src['end']}s]: {src['text']}")
            
    # Process new user query
    if query := st.chat_input("Ask questions about the videos (e.g., summaries, specific details)..."):
        if 'vectorstore' not in st.session_state:
            st.warning("⚠️ Please initialize the RAG engine from the left panel first.")
        else:
            # Display user query
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(query)
            
            # Extract recent chat history for context-aware processing
            history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:-1]]) 
            
            with st.chat_message("assistant", avatar="🤖"):
                
                # ------------------------------------------------
                # Phase 1: Contextual Query Rewriting
                # ------------------------------------------------
                with st.spinner("Analyzing intent and rewriting query..."):
                    rewrite_prompt = (
                        "Task: You are an expert search query generator. Based on the provided Chat History, "
                        "convert the user's latest ambiguous query into 1-2 specific, standalone search keywords.\n"
                        "Rules:\n"
                        "1. Remove all pronouns (e.g., 'it', 'then', 'next').\n"
                        "2. Resolve contextual references (e.g., if history mentions 'gloves' and user asks 'what next', rewrite to specific actions like 'stop bleeding').\n"
                        "3. Output ONLY the keywords. No explanations.\n\n"
                        f"Chat History:\n{history_text}\n"
                        f"Latest Query: {query}"
                    )
                    
                    if history_text.strip():
                        standalone_query = rag_engine.llm.invoke(rewrite_prompt).content.strip()
                    else:
                        standalone_query = query 
                        
                    if standalone_query != query:
                        st.caption(f"*(System Rewritten Query: '{standalone_query}')*")

                # ------------------------------------------------
                # Phase 2: Retrieval, Reranking & Generation
                # ------------------------------------------------
                with st.spinner("Retrieving cross-video knowledge and generating response..."):
                    
                    system_prompt = (
                        "You are a professional video content analyst. Engage in natural conversation based on the provided Context.\n"
                        "Guidelines:\n"
                        "1. Use conversational language rather than just listing timestamps.\n"
                        "2. STRICT METADATA COMPLIANCE: When referencing time, you MUST use the exact 'start' and 'end' values from the Metadata. Do not estimate or invent timestamps.\n"
                        "3. Address the user's specific context before expanding on the details.\n\n"
                        "Context:\n{context}"
                    )
                    
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])

                    # Step A: Broad Retrieval (Recall)
                    base_retriever = st.session_state['vectorstore'].as_retriever(search_kwargs={"k": top_k * 4})
                    
                    # Step B: Semantic Reranking (Precision)
                    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
                    compressor = CrossEncoderReranker(model=model, top_n=top_k)
                    rerank_retriever = ContextualCompressionRetriever(
                        base_compressor=compressor, 
                        base_retriever=base_retriever
                    )

                    # Step C: LLM Generation Chain
                    qa_chain = create_stuff_documents_chain(rag_engine.llm, prompt)
                    rag_chain = create_retrieval_chain(rerank_retriever, qa_chain) 
                    
                    response = rag_chain.invoke({"input": standalone_query})
                    answer = response["answer"]
                    docs = response["context"]
                    
                    # Display Answer
                    st.markdown(answer)
                    
                    # Step D: Metadata Extraction & Traceability Rendering
                    source_list = [
                        {
                            "start": d.metadata.get("start", 0.0), 
                            "end": d.metadata.get("end", 0.0), 
                            "source": d.metadata.get("source_video", "Unknown"),
                            "text": d.page_content
                        } for d in docs
                    ]
                    
                    with st.expander("🔍 View Retrieved Sources (Cross-Video)"):
                        for src in source_list:
                            st.caption(f"📺 **Source: `{src['source']}.mp4`** | ⏱️ [{src['start']}s - {src['end']}s]: {src['text']}")
                    
                    # Update state
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": source_list
                    })