import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from config import settings

class VideoRAG:
    def __init__(self):
        print(f"[Engine] 初始化 BGE 向量模型 (CPU模式)...")
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=settings.EMBED_MODEL_NAME,
            model_kwargs={'device': settings.DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        print(f"[Engine] 连接本地大模型: {settings.LLM_MODEL_NAME}...")
        self.llm = ChatOllama(
            model=settings.LLM_MODEL_NAME, 
            temperature=0.1
        )

    def build_vectorstore(self, chunks: list[dict], index_name: str):
            # 规范化命名：将 video_stem 改为 index_name，以兼容多视频合并建库的场景
            index_path = str(settings.INDEX_DIR / f"{index_name}_faiss")
            
            # 缓存读取逻辑
            if os.path.exists(index_path):
                print(f"[Engine] 检测到本地缓存，加载已有向量库: {index_name}")
                return FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
            
            print(f"[Engine] 正在构建新的 FAISS 向量库: {index_name}...")
            
            # 构造文档对象并完整映射元数据 (Metadata)
            docs = []
            for c in chunks:
                # 使用 dict.get 提供默认值，增强代码鲁棒性
                meta = {
                    "start": c.get("start", 0.0),
                    "end": c.get("end", 0.0),
                    "source_video": c.get("source_video", "unknown")  # 关键修复：保留多文档溯源字段
                }
                docs.append(Document(page_content=c.get("text", ""), metadata=meta))
                
            vectorstore = FAISS.from_documents(docs, self.embeddings)
            vectorstore.save_local(index_path)
            return vectorstore

    def chat_loop(self, vectorstore, top_k=5): 
        """带有动态 Top-K 调节和严格元数据约束的进阶版问答循环"""
        
        # 【修改】升级 System Prompt：断绝模型“脑补”时间戳的念头
        system_prompt = (
            "你是一个极其严谨的视频内容分析助手。\n"
            "【核心指令】：请严格根据以下提供的视频字幕片段（Context）回答问题。\n"
            "【时间戳要求】：你必须直接引用片段中元数据（Metadata）里的 'start' 和 'end' 字段。\n"
            "【严禁】：绝对禁止自行估算或修改时间。如果 Context 显示是 14.3s，你就必须写 14.3s，不能写 14s 或 15s。\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
       
        # 【第一阶段：海选】召回范围设为滑块值的 4 倍，保证 Reranker 有足够的筛选空间
        # 比如 UI 拉到 10，这里就搜 40 个片段
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k * 4})
        
        # 【第二阶段：精选】
        # 【修改】显式设置 top_n 为 UI 传入的 top_k。
        # 之前因为没设这个，导致它默认只吐出 5 个片段给大模型。
        compressor = FlashrankRerank(model="ms-marco-MultiBERT-L-12", top_n=top_k)
        
        rerank_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=base_retriever
        )
        # --------------------------------------------

        # 2. 构建 LCEL 链
        qa_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(rerank_retriever, qa_chain)

        print(f"\n🚀 [Rerank 引擎启动] 召回深度: {top_k * 4}, 最终精选: {top_k}")
        
        # ... 后续循环逻辑保持不变 ...
            
        while True:
            query = input("Q: ").strip()
            if query.lower() in {"q", "quit", "exit"}: break
                
            print("正在深度检索并精排中，请稍候...")
            response = rag_chain.invoke({"input": query})
            
            print("\n🤖 大模型回答：")
            print(response["answer"])
                
            # 可选：打印出被精选出来的参考片段，方便调试
            print("\n[🔍 重排后的参考来源]：")
            for doc in response["context"]:
                print(f"- {doc.metadata['start']:.1f}s ~ {doc.metadata['end']:.1f}s: {doc.page_content[:50]}...")
            print("-" * 50 + "\n")
