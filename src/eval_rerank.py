# eval_rerank.py
import json
from pathlib import Path
from config import settings
from rag_engine import VideoRAG

from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker

def check_hit_rank(retrieved_docs, gt_video, gt_start, gt_end, tolerance=5.0):
    for i, doc in enumerate(retrieved_docs):
        doc_start = doc.metadata.get("start", 0)
        doc_end = doc.metadata.get("end", 0)
        doc_source = doc.metadata.get("source_video", "") # 校验防伪标签
        
        # 只有视频来源一致，且时间戳吻合，才算真正命中！
        if doc_source == gt_video and (doc_start <= gt_end + tolerance) and (doc_end >= gt_start - tolerance):
            return i + 1 # 返回排名（第一名就是 1，第二名就是 2）
    return 0 # 没命中

def run_rerank_eval():
    rag_engine = VideoRAG()
    eval_file = Path("data/eval/eval_dataset.json")
    
    with open(eval_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    index_name = "multi_video_kb" 
    index_path = str(settings.INDEX_DIR / f"{index_name}_faiss")
    
    print(f"[Engine] 正在加载混合知识库: {index_name}")
    vectorstore = FAISS.load_local(index_path, rag_engine.embeddings, allow_dangerous_deserialization=True)

    print("\n" + "="*55)
    print("🔥 终极对决：多视频混合库下的纯向量 vs BGE重排")
    print("="*55)

    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    wide_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    
    print("[Engine] 加载 BAAI/bge-reranker-base 模型...")
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=5)
    rerank_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=wide_retriever
    )

    pipelines = {"Base (纯FAISS向量)": base_retriever, "BGE Reranker (重排)": rerank_retriever}
    results = {}

    for name, retriever in pipelines.items():
        print(f"\n[正在测试] {name} ...")
        hits_top1 = 0
        hits_top5 = 0
        mrr_sum = 0.0
        
        for item in dataset:
            docs = retriever.invoke(item["query"])
            rank = check_hit_rank(docs, item["gt_video"], item["gt_start"], item["gt_end"])
            
            if rank > 0:
                hits_top5 += 1
                mrr_sum += 1.0 / rank  # 计算 MRR 核心公式
                if rank == 1:
                    hits_top1 += 1
                
        total = len(dataset)
        results[name] = {
            "Recall@1": (hits_top1 / total) * 100,
            "Recall@5": (hits_top5 / total) * 100,
            "MRR": mrr_sum / total
        }

    print("\n🏆 多文档混合场景 - Rerank 性能提升表：")
    print("-" * 65)
    print(f"{'检索架构':<22} | {'Recall@1':<8} | {'Recall@5':<8} | {'MRR':<5}")
    print("-" * 65)
    for k, v in results.items():
        print(f"{k:<22} | {v['Recall@1']:>7.1f}% | {v['Recall@5']:>7.1f}% | {v['MRR']:.3f}")
    print("-" * 65)

if __name__ == "__main__":
    run_rerank_eval()