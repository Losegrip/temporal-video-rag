# eval.py
import json
from pathlib import Path
from config import settings
from data_processor import VideoProcessor
from rag_engine import VideoRAG

def check_hit_rank(retrieved_docs, gt_video, gt_start, gt_end, tolerance=5.0):
    """【核心指标函数】返回命中的真实排名(Rank)，用于计算 MRR"""
    for i, doc in enumerate(retrieved_docs):
        doc_start = doc.metadata.get("start", 0)
        doc_end = doc.metadata.get("end", 0)
        doc_source = doc.metadata.get("source_video", "") # 严苛的防伪溯源
        
        # 必须来源一致，且时间戳吻合，才算真正命中
        if doc_source == gt_video and (doc_start <= gt_end + tolerance) and (doc_end >= gt_start - tolerance):
            return i + 1 # 返回名次 (1到5)
    return 0 # 未命中

def run_chunk_eval():
    processor = VideoProcessor()
    rag_engine = VideoRAG()
    
    eval_file = Path("data/eval/eval_dataset.json")
    with open(eval_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 待检验的三大策略
    strategies = ["single", "merge", "sliding_window"]
    results = {}

    # 扫描所有视频，准备制造“混合干扰池”
    video_files = list(settings.VIDEO_DIR.glob("*.mp4"))
    if not video_files:
        print("❌ 找不到任何视频文件！请检查 data/videos 目录。")
        return

    print("\n" + "="*65)
    print(f"🚀 开始执行 Chunk 策略大比拼 (混合 {len(video_files)} 个视频制造干扰)")
    print("="*65)

    for strategy in strategies:
        print(f"\n[正在构建与测试] 当前切片策略: {strategy.upper()} ...")
        all_chunks = []
        
        # 1. 动态生成当前策略下的多视频混合库
        for video_path in video_files:
            chunks = processor.process_video(video_path, strategy=strategy)
            for chunk in chunks:
                chunk["source_video"] = video_path.stem # 注入防伪标签
            all_chunks.extend(chunks)
            
        # 2. 建立独立命名的临时向量库，防止互相覆盖
        index_name = f"eval_multi_video_{strategy}"
        vectorstore = rag_engine.build_vectorstore(all_chunks, index_name)
        
        # 3. 纯向量检索测试 (不加 Rerank，直接看切片本身的抗干扰能力)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        hits_top1 = 0
        hits_top5 = 0
        mrr_sum = 0.0
        
        # 4. 跑题库计算指标
        for item in dataset:
            docs = retriever.invoke(item["query"])
            rank = check_hit_rank(docs, item["gt_video"], item["gt_start"], item["gt_end"])
            
            if rank > 0:
                hits_top5 += 1
                mrr_sum += 1.0 / rank
                if rank == 1:
                    hits_top1 += 1
                
        total = len(dataset)
        results[strategy] = {
            "Recall@1": (hits_top1 / total) * 100,
            "Recall@5": (hits_top5 / total) * 100,
            "MRR": mrr_sum / total
        }

    print("\n🏆 多视频混合场景 - Chunk 策略实验结论：")
    print("-" * 65)
    print(f"{'切片策略 (Chunking)':<20} | {'Recall@1':<8} | {'Recall@5':<8} | {'MRR':<5}")
    print("-" * 65)
    for k, v in results.items():
        print(f"{k:<20} | {v['Recall@1']:>7.1f}% | {v['Recall@5']:>7.1f}% | {v['MRR']:.3f}")
    print("-" * 65)

if __name__ == "__main__":
    run_chunk_eval()