import os

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_DIR = os.path.join(ROOT_DIR, "documents")

# 向量数据库存储路径
INDEX_DIR = os.path.join(ROOT_DIR, "vector_db")
INDEX_FILE = os.path.join(INDEX_DIR, "faiss_index.bin")

# 文档路径
FILE_PATHS = [
    # 在此添加更多的文档路径
    # os.path.join(DATA_DIR, "***.txt"),
]

# 文本分割参数
# 增加chunk_size可以保留更多上下文，提高精确查询的准确性
CHUNK_SIZE = 800
# 增加overlap确保重要信息不会被切断
CHUNK_OVERLAP = 150

# 是否使用本地模型进行向量化
# 如果设置为False，将使用Ollama API
# 如果设置为True，请确保已安装sentence-transformers和torch，并有可用的GPU（推荐）
USE_LOCAL_EMBEDDING_MODEL = False

# 本地向量化模型配置 (当 USE_LOCAL_EMBEDDING_MODEL = True 时有效)
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 输出384维

# Ollama API 向量化模型配置 (当 USE_LOCAL_EMBEDDING_MODEL = False 时有效)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "dengcao/Qwen3-Embedding-0.6B:Q8_0"
OLLAMA_EMBEDDING_DIMENSION = 1024  # Qwen3-Embedding 实际输出1024维

# 问答生成模型配置 (Ollama)
# OLLAMA_QA_MODEL = "deepseek-r1:1.5b"  # 太小，容易产生幻觉
OLLAMA_QA_MODEL = "qwen2.5:7b" 

# 检索参数
RETRIEVER_TOP_K = 10

# 重排序模型配置
# 是否启用重排序
USE_RERANKER = True
# Cross-Encoder模型，用于对检索结果进行重排序，以提高精度
RERANKER_MODEL = 'BAAI/bge-reranker-base'
# 重排序后返回的文档数量
RERANKER_TOP_K = 10  # 增加返回数量，给LLM更多上下文
# 初始检索时应获取更多文档以供重排序
INITIAL_RETRIEVAL_TOP_K = 30  # 增加初始检索数量以提高召回率
