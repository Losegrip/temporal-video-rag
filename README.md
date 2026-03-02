# Video-RAG: 跨视频多模态检索与语义分析系统

本项目实现了一个基于 **RAG (Retrieval-Augmented Generation)** 架构的本地化视频内容分析系统。通过集成 ASR (语音识别)、向量检索与交叉编码器重排技术，系统能够对多源视频进行自动化索引，并提供具备精确时间戳溯源的问答能力。

## 🛠 技术架构

系统流水线由数据预处理、多维索引构建、两阶段检索链路以及生成式问答四个核心环节组成。

- **音频转录与预处理**：利用 `faster-whisper` 模型提取视频音频轨并转录文本，生成的 JSON 缓存包含每个 Segments 的精确起止时间。

- **语义分块 (Chunking)**：针对视频流式文本特性，实现了单句切分 (Single)、定长合并 (Merge) 以及滑动窗口 (Sliding Window) 三种策略，以解决长文本检索中的上下文缺失问题。

- **向量化存储**：使用 `BGE-m3` 模型将文本块映射为 1024 维密集向量，并利用 `FAISS` 进行索引构建与持久化。

- **多阶段检索链路**：系统采用 Bi-Encoder 进行初筛召回，随后使用 Cross-Encoder (`bge-reranker-base`) 对结果进行深度语义评分与排序重排。

## 📊 实验数据与指标评估

通过对比实验，系统验证了不同算法策略在多视频混合干扰场景下的性能表现。

### 1. 切片策略对比 (Recall & MRR)

对比在不启用重排的情况下，不同 Chunking 粒度对检索精度的影响。

| **策略类型 (Strategy)**       | **Recall@1** | **Recall@5** | **MRR** |
| ------------------------- | ------------ | ------------ | ------- |
| **Single (原始切片)**         | 55.0%        | 90.0%        | 0.685   |
| **Merge (定长合并)**          | 85.0%        | 100.0%       | 0.904   |
| **Sliding Window (滑动窗口)** | 75.0%        | 95.0%        | 0.842   |

> **实验跑图记录：** > ![Chunking Strategy Results](assets/test1.png)

### 2. Rerank 性能提升验证

在多视频混合的知识库中，测试两阶段检索链路相对于基础向量检索的增益。

| **检索架构 (Architecture)**  | **Recall@1** | **Recall@5** | **MRR**   |
| ------------------------ | ------------ | ------------ | --------- |
| **Base (纯向量检索)**         | 75.0%        | 95.0%        | 0.842     |
| **BGE Reranker (召回+重排)** | **90.0%**    | **100.0%**   | **0.938** |

> **实验跑图记录：** > ![Rerank Performance Improvement](assets/test2.png)

- **指标解释**：MRR (Mean Reciprocal Rank) 反映了正确答案在返回列表中的平均名次倒数，体现了系统的排序能力。

## 📂 目录结构

Plaintext

```
├── assets               #[Local Directory Structure](assets/image)
├── data/
│   ├── videos/          # 原始视频文件目录 (.mp4)
│   ├── eval/            # 自动化评测集与 Ground Truth
│   ├── index/           # FAISS 向量索引持久化文件
│   └── tmp/             # Whisper 转录 JSON 缓存
├── src/
│   ├── config.py         # 环境路径与模型参数配置
│   ├── data_processor.py # ASR 逻辑与切片算法实现
│   ├── rag_engine.py     # 向量检索与重排核心引擎
│   ├── eval.py           # 切片策略量化评测脚本
│   ├── eval_rerank.py    # 检索链路性能评估脚本
│   └── web_ui.py         # 基于 Streamlit 的交互前端
├── requirements.txt      # 依赖库清单
└── README.md
```

## 🚀 部署指引

### 1. 基础环境

- **Python**: 3.9+

- **FFmpeg**: 需自行安装二进制包并配置系统环境变量。

- **Models**: 系统首次运行将自动从 HuggingFace Hub 拉取所需的 `BGE` 与 `Whisper` 模型权重。

### 2. 执行步骤

Bash

```
# 安装依赖
pip install -r requirements.txt

# 自动化构建混合索引
python src/main.py

# 启动可视化分析系统
streamlit run src/web_ui.py
```

## 🚀 运行效果预览 (Demo)

### 跨视频语义溯源 系统不仅能给出回答，还能从混合知识库中准确识别视频来源（如 `blood.mp4`, `history.mp4`）并定位到具体秒数。 >

> ![UI Screenshot Placeholder](assets/ui1.png) 

> ![UI Screenshot Placeholder](assets/ui2.png)

## 📌 免责声明

本仓库仅包含核心源代码、评测基准数据及环境配置文件。由于体积与版权限制，仓库内**不包含** FFmpeg 工具包、模型权重文件及原始视频媒体。
