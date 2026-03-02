import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any
from faster_whisper import WhisperModel
from config import settings

class VideoProcessor:
    def __init__(self):
        print(f"[Processor] 加载 Whisper: {settings.WHISPER_MODEL_NAME} (CPU模式)")
        self.whisper_model = WhisperModel(
            settings.WHISPER_MODEL_NAME, 
            device=settings.DEVICE, 
            compute_type="int8"
        )

    def extract_audio(self, video_path: Path, audio_path: Path):
        if audio_path.exists():
            print(f"[Processor] 音频已存在，跳过提取: {audio_path.name}")
            return
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_path)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # 隐藏烦人的 ffmpeg 刷屏日志

    def transcribe(self, audio_path: Path, json_path: Path) -> List[Dict[str, Any]]:
        if json_path.exists():
            print(f"[Processor] 转录结果已存在，直接读取: {json_path.name}")
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)["segments"]
                
        print(f"[Processor] 正在用 Whisper 转录音频...")
        segments, info = self.whisper_model.transcribe(str(audio_path), beam_size=5)
        seg_list = [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()} for s in segments]
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"language": info.language, "segments": seg_list}, f, ensure_ascii=False, indent=2)
        return seg_list

    def build_chunks(self, segments: List[Dict[str, Any]], strategy: str = "merge", max_chars: int = 200, max_time_span: float = 30.0) -> List[Dict[str, Any]]:
        chunks = []
        
        # 策略 1：Whisper 原始单句切分 (粒度极细，易丢失上下文)
        if strategy == "single":
            return segments
            
        # 策略 2：滑窗重叠切分 (工业界最常用，每 3 句话合并，步长为 2，保证上下文不被硬生生切断)
        elif strategy == "sliding_window":
            window_size = 3
            step = 2
            for i in range(0, len(segments), step):
                window = segments[i:i+window_size]
                if not window: break
                start_time = window[0]["start"]
                end_time = window[-1]["end"]
                text = " ".join([s["text"] for s in window])
                chunks.append({"start": start_time, "end": end_time, "text": text})
            return chunks
            
        # 策略 3：长度/时间合并切分
        else: # "merge"
            buf_text, start_time, last_time = [], None, None
            for seg in segments:
                s, e, text = seg["start"], seg["end"], seg["text"]
                if start_time is None: start_time = s
                last_time = e
                buf_text.append(text)
                joined = " ".join(buf_text)

                if len(joined) > max_chars or (e - start_time) > max_time_span:
                    chunks.append({"start": start_time, "end": e, "text": joined})
                    buf_text, start_time = [], None

            if buf_text:
                chunks.append({"start": start_time, "end": last_time, "text": " ".join(buf_text)})
            return chunks

    # 同时修改 process_video，接收 strategy 参数
    def process_video(self, video_path: Path, strategy: str = "merge") -> List[Dict[str, Any]]:
        video_stem = video_path.stem
        audio_path = settings.TMP_DIR / f"{video_stem}.wav"
        json_path = settings.TMP_DIR / f"{video_stem}_whisper.json"
        
        self.extract_audio(video_path, audio_path)
        segments = self.transcribe(audio_path, json_path)
        
        # 传入策略
        chunks = self.build_chunks(segments, strategy=strategy)
        print(f"[Processor] 使用 {strategy} 策略，共生成 {len(chunks)} 个字幕检索块。")
        return chunks
