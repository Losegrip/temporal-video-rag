import os
from pathlib import Path

class Settings:
    # 路径配置 (自动寻找项目根目录)
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    VIDEO_DIR = DATA_DIR / "videos"
    TMP_DIR = DATA_DIR / "tmp"
    INDEX_DIR = DATA_DIR / "index"

    # 模型配置
    WHISPER_MODEL_NAME = "model/faster-whisper-small"
    EMBED_MODEL_NAME = "model/bge-m3"
    LLM_MODEL_NAME = "qwen2.5:7b"

    # 硬件配置: 强制前置处理走 CPU，保护 8G 显存留给 Ollama
    DEVICE = "cpu" 

    def __init__(self):
        # 确保目录存在
        self.TMP_DIR.mkdir(parents=True, exist_ok=True)
        self.INDEX_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()