import os
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from tqdm import tqdm
import torch
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import faiss


# ---------- 配置 ----------
DATA_DIR = Path("data")
VIDEO_DIR = DATA_DIR / "videos"
TMP_DIR = DATA_DIR / "tmp"
INDEX_DIR = DATA_DIR / "index"

TMP_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_SUFFIX = ".wav"

WHISPER_MODEL_NAME = "model/faster-whisper-small"    
EMBED_MODEL_NAME = "model/bge-m3"    # 或者 m3e


# ---------- 工具函数 ----------

def run_cmd(cmd: List[str]):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def extract_audio_ffmpeg(video_path: Path, audio_path: Path):
    """
    用 ffmpeg 从视频中抽取音频，保存为 wav
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",          # 不要视频
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path)
    ]
    run_cmd(cmd)


# ---------- 1. Whisper 转录 ----------

def transcribe_audio(audio_path: Path, json_path: Path) -> List[Dict[str, Any]]:
    """
    用 faster-whisper 转录音频，返回每个片段（含时间戳和文本）
    """
    print(f"[Whisper] loading model: {WHISPER_MODEL_NAME}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(
        WHISPER_MODEL_NAME,
        device=device,
        compute_type="float16" if device == "cuda" else "int8"
    )

    print(f"[Whisper] transcribing {audio_path}")
    segments, info = model.transcribe(str(audio_path), beam_size=5)

    seg_list = []
    for seg in segments:
        seg_list.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip()
        })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "language": info.language,
            "segments": seg_list
        }, f, ensure_ascii=False, indent=2)

    print(f"[Whisper] saved segments -> {json_path}")
    return seg_list


def load_transcript(json_path: Path) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["segments"]


# ---------- 2. 文本切块 ----------

def build_chunks(segments: List[Dict[str, Any]],
                 max_chars: int = 200,
                 max_time_span: float = 30.0) -> List[Dict[str, Any]]:
    """
    简单按照“字符数 + 时间跨度”做滑动聚合
    """
    chunks = []
    buf_text = []
    start_time = None
    last_time = None

    for seg in segments:
        text = seg["text"]
        s, e = seg["start"], seg["end"]

        if start_time is None:
            start_time = s
        last_time = e

        buf_text.append(text)
        joined = " ".join(buf_text)

        too_long = len(joined) > max_chars
        too_slow = (e - start_time) > max_time_span

        if too_long or too_slow:
            chunks.append({
                "start": start_time,
                "end": e,
                "text": joined
            })
            buf_text = []
            start_time = None

    # 剩余缓冲
    if buf_text:
        chunks.append({
            "start": start_time,
            "end": last_time,
            "text": " ".join(buf_text)
        })

    return chunks


# ---------- 3. Embedding + 向量库 ----------

def build_index(chunks: List[Dict[str, Any]],
                index_path: Path,
                meta_path: Path):
    print(f"[Embed] loading model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    texts = [c["text"] for c in chunks]
    print(f"[Embed] encoding {len(texts)} chunks...")
    emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    emb = emb.astype("float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)   # 内积，记得先 L2 normalize
    faiss.normalize_L2(emb)
    index.add(emb)

    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"[Index] saved index -> {index_path}")
    print(f"[Index] saved meta  -> {meta_path}")


def load_index(index_path: Path, meta_path: Path):
    index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks


def search(query: str,
           index,
           chunks: List[Dict[str, Any]],
           top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
    model = SentenceTransformer(EMBED_MODEL_NAME)
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    scores = D[0]
    idxs = I[0]

    results = []
    for score, idx in zip(scores, idxs):
        results.append((float(score), chunks[idx]))
    return results


# ---------- 主流程 ----------

def build_video_index(video_path: Path):
    video_stem = video_path.stem
    audio_path = TMP_DIR / f"{video_stem}{AUDIO_SUFFIX}"
    json_path = TMP_DIR / f"{video_stem}_whisper.json"
    index_path = INDEX_DIR / f"{video_stem}.faiss"
    meta_path = INDEX_DIR / f"{video_stem}_meta.json"

    if not audio_path.exists():
        extract_audio_ffmpeg(video_path, audio_path)

    if not json_path.exists():
        segments = transcribe_audio(audio_path, json_path)
    else:
        segments = load_transcript(json_path)

    chunks = build_chunks(segments)
    print(f"[Chunk] got {len(chunks)} chunks.")
    build_index(chunks, index_path, meta_path)


def qa_loop(video_stem: str):
    index_path = INDEX_DIR / f"{video_stem}.faiss"
    meta_path = INDEX_DIR / f"{video_stem}_meta.json"

    index, chunks = load_index(index_path, meta_path)

    print("\n进入问答模式，输入问题，回车检索；输入 q 退出。\n")
    while True:
        q = input("Q: ").strip()
        if q.lower() in {"q", "quit", "exit"}:
            break

        results = search(q, index, chunks, top_k=5)
        print("---- 检索结果 ----")
        for i, (score, ch) in enumerate(results, 1):
            print(f"[{i}] score={score:.3f} time={ch['start']:.1f}~{ch['end']:.1f}s")
            print(ch["text"])
            print()
        print("------------------\n")


if __name__ == "__main__":
    import argparse
    import torch  # 用于判断是否有 cuda

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="视频文件路径（放在 data/videos 也行）")
    parser.add_argument("--build_only", action="store_true", help="只构建索引，不进入问答环节")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        # 尝试从 data/videos 下找
        candidate = VIDEO_DIR / args.video
        if candidate.exists():
            video_path = candidate
        else:
            raise FileNotFoundError(f"找不到视频：{args.video}")

    build_video_index(video_path)
    if not args.build_only:
        qa_loop(video_path.stem)
