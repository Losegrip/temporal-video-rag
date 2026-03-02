import json
from pathlib import Path
from config import settings
from data_processor import VideoProcessor
from rag_engine import VideoRAG

def main():
    print("\n=== 1. 系统初始化 ===")
    processor = VideoProcessor()
    rag_engine = VideoRAG()
    all_chunks = []

    # 🌟 核心改动：直接扫描你左侧目录中已有的 videos 文件夹下的所有 mp4！
    video_files = list(settings.VIDEO_DIR.glob("*.mp4"))
    
    if not video_files:
        print(f"❌ 在 {settings.VIDEO_DIR} 目录下找不到任何 mp4 视频，请检查！")
        return

    print(f"\n=== 2. 开始批量解析 {len(video_files)} 个视频 ===")
    for video_path in video_files:
        print(f"\n>> 正在处理: {video_path.name}")
        
        # 调用 data_processor 里的方法进行切片
        chunks = processor.process_video(video_path, strategy="sliding_window")
        
        # 🌟 注入来源防伪标签 (重要：用于多视频区分)
        for chunk in chunks:
            chunk["source_video"] = video_path.stem
            
        all_chunks.extend(chunks)
        print(f"   - {video_path.name} 提取了 {len(chunks)} 个切片。")

    print(f"\n=== 3. 统一知识库挂载 (总计合并 {len(all_chunks)} 个切片) ===")
    # 建立统一的多视频向量库
    vectorstore = rag_engine.build_vectorstore(all_chunks, "multi_video_kb")
    
    # 【可选】如果你需要把总数据保存为 JSON 供 web_ui 读取，加上这两句：
    # output_path = settings.DATA_DIR / "total_rag_kb.json"
    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    # print(f"✅ 总 JSON 已保存至: {output_path.name}")

    print("\n=== 4. 多文档问答引擎就绪 ===")
    # 启动命令行问答测试 (如果你用 web_ui 问答，这里可以注释掉)
    rag_engine.chat_loop(vectorstore)

if __name__ == "__main__":
    main()