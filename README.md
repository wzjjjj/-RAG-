# RAG - 任务型对话 Agent 检索增强生成系统

这是一个基于 RAG (Retrieval-Augmented Generation) 技术的任务型对话 Agent 系统，专注于处理文档（如 PDF）、语义切分、混合检索和重排序，以及自动生成 QA 对用于模型训练。

## ✨ 主要特性

*   **文档解析**: 支持 PDF 文档解析，提取文本和图像 (`src/parser`)。
*   **语义切分**: 提供基于语义相似度的文本切分服务，使用 `SentenceTransformer` 和聚类算法 (`src/server/semantic_chunk.py`)。
*   **混合检索**: 支持多种检索方式，包括关键字检索 (BM25, TF-IDF) 和向量检索 (FAISS, Milvus) (`src/retriever`)。
*   **重排序 (Rerank)**: 集成先进的重排序模型 (BGE-M3, Qwen3) 以提高检索准确性 (`src/reranker`)。
*   **数据生成**: 利用 LLM 自动从文档中生成 QA 对、泛化问题和提取关键词，用于构建训练和测试数据集 (`src/gen_qa`)。

## 🛠️ 环境依赖

*   Python 3.8+ (推荐 Python 3.10+)
*   MongoDB (用于存储文档和元数据)
*   CUDA 环境 (用于模型推理加速)

安装 Python 依赖:

```bash
pip install -r requirements.txt
```

## 📂 项目结构

```
RAG/
├── data/                   # 数据目录 (通常在 .gitignore 中)
├── src/                    # 源代码目录
│   ├── client/             # 客户端工具 (LLM 客户端, Mongo 配置等)
│   ├── fields/             # 数据字段定义
│   ├── gen_qa/             # QA 数据生成脚本
│   ├── parser/             # 文档解析器 (PDF, 图片处理)
│   ├── reranker/           # 重排序模型实现 (BGE-M3, Qwen3)
│   ├── retriever/          # 检索器实现 (BM25, FAISS, Milvus)
│   ├── server/             # 后端服务 (如语义切分 API)
│   └── utils.py            # 通用工具函数
├── requirements.txt        # 项目依赖
└── README.md               # 项目说明文档
```

## 🚀 快速开始

### 1. 启动语义切分服务

该服务提供基于语义的文本切分 API，默认运行在 6000 端口。

```bash
python src/server/semantic_chunk.py
```

### 2. 文档解析与处理

解析 PDF 文档并进行切分（需配置 `constant.py` 中的路径）：

```bash
python src/parser/pdf_parse.py
```

### 3. 生成 QA 数据集

从处理好的文档中自动生成问答对：

```bash
python src/gen_qa/run.py
```

## 🧩 模块说明

### Retriever (检索器)
位于 `src/retriever`，提供了统一的 `BaseRetriever` 接口。
*   **BM25Retriever**: 基于 BM25 算法的关键字检索。
*   **FAISSRetriever**: 基于 FAISS 的本地向量检索。
*   **MilvusRetriever**: 基于 Milvus 向量数据库的检索。

### Reranker (重排序)
位于 `src/reranker`，提供了 `RerankerBase` 接口。
*   **BGEM3Reranker**: 使用 BGE-M3 模型进行重排序。
*   **Qwen3Reranker**: 使用 Qwen3 模型进行重排序。

## 🤝 贡献

欢迎提交 Pull Request 或 Issue 来改进本项目。

## 📄 许可证

[MIT License](LICENSE)
