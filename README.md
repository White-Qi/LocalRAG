## 简介

本项目针对于当前大模型部署难、个性化和专业化程度不够高的问题，设计了一套本地知识库的RAG系统，有效帮助个人快捷在本地部署大模型，并且利用当前主流方式，创建一个个人知识问答系统。

**亮点**

1. 文档处理：LangChain的RecursiveCharacterTextSplitter
2. 向量化：Sentence-Transformers / Ollama Embedding API
3. 向量检索：Faiss高效相似度搜索
4. 精确排序：CrossEncoder重排序
5. 答案生成：Ollama本地大模型

## 安装部署

### 环境要求

- Python 3.8+
- 4GB+ 内存
- 磁盘空间：至少5GB（用于存储模型和向量数据库）
- Ollama（用于本地大模型推理）

### 1. 克隆项目

```bash
git clone https://github.com/White-Qi/LocalRAG.git
cd LocalRAG
```

### 2. 安装Python依赖

建议使用虚拟环境：

```bash
# 创建虚拟环境
python -m venv venv
# 或者使用conda


# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 安装Ollama

#### macOS

```bash
# 使用Homebrew安装
brew install ollama

# 或直接下载安装包
# 访问 https://ollama.ai 下载安装
```

#### Linux

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Windows

访问 https://ollama.ai 下载Windows安装包

### 4. 启动Ollama服务

```bash
# 启动Ollama服务（默认端口11434）
ollama serve
```

### 5. 下载所需模型

打开新终端窗口，下载项目所需的模型：

```bash
# 下载嵌入模型（用于文本向量化）
ollama pull dengcao/Qwen3-Embedding-0.6B:Q8_0

# 下载问答生成模型
ollama pull qwen2.5:7b

# 下载重排序模型（通过sentence-transformers自动下载）
# 首次运行时会自动下载 BAAI/bge-reranker-base
```

### 6. 准备文档

将您的文档放入 `documents/` 目录：

```bash
# 在 documents 目录下放置文档文件
# 系统会自动扫描该目录及子目录下的所有支持格式的文件
```

**支持的文件格式**:

- `.txt` - 纯文本文件
- `.md` - Markdown 文件
- `.pdf` - PDF 文档
- `.doc`, `.docx` - Word 文档

**注意**:

- 系统会**自动递归扫描** `documents` 目录及其所有子目录
- 无需在配置文件中手动指定文件路径
- 可以创建子目录来组织不同类别的文档
- 如需手动指定特定文件，可以编辑 `src/config.py` 中的 `FILE_PATHS` 变量

### 7. 配置系统(可选，第一次使用建议跳过)

编辑 `src/config.py` 文件，根据需要调整配置：

- `OLLAMA_BASE_URL`: Ollama服务地址（默认：http://localhost:11434）
- `OLLAMA_EMBEDDING_MODEL`: 嵌入模型名称
- `OLLAMA_QA_MODEL`: 问答生成模型名称
- `USE_LOCAL_EMBEDDING_MODEL`: 是否使用本地sentence-transformers模型
- `USE_RERANKER`: 是否启用重排序（提高检索精度）

### 8. 运行系统

```bash
# 首次运行，建立向量索引
python main.py --reindex

# 后续运行（使用已有索引）
python main.py
```

### 常见问题

#### 1. Ollama连接失败

- 确保Ollama服务已启动：`ollama serve`
- 检查端口是否被占用
- 确认 `config.py` 中的 `OLLAMA_BASE_URL` 配置正确

#### 2. 内存不足

- 可以切换到更小的模型，如：`ollama pull qwen2.5:1.5b`
- 在 `config.py` 中修改 `OLLAMA_QA_MODEL = "qwen2.5:1.5b"`

#### 3. 首次运行缓慢

- 首次运行时会下载重排序模型（约500MB），请耐心等待
- 向量索引构建需要一定时间，取决于文档数量

#### 4. GPU加速（可选）

如果有NVIDIA GPU，可以启用GPU加速

### 使用说明

运行后，系统会进入交互式问答模式：

```
请输入您的问题 (输入 'quit' 或 Ctrl+C 退出): 你的问题
```

输入问题后，系统会：

1. 在向量数据库中检索相关文档片段
2. 使用重排序模型优化检索结果
3. 调用大模型生成答案

输入 `quit` 或按 `Ctrl+C` 退出系统。
