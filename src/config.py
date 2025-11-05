import os
import glob

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_DIR = os.path.join(ROOT_DIR, "documents")

# 向量数据库存储路径
INDEX_DIR = os.path.join(ROOT_DIR, "vector_db")
INDEX_FILE = os.path.join(INDEX_DIR, "faiss_index.bin")

# 支持的文档文件格式
SUPPORTED_EXTENSIONS = ['.txt', '.md', '.pdf', '.doc', '.docx']

def get_all_document_files(data_dir=DATA_DIR, extensions=None, recursive=True):
    """
    自动扫描数据目录中的所有文档文件
    
    Args:
        data_dir (str): 文档目录路径
        extensions (list): 支持的文件扩展名列表，默认为 SUPPORTED_EXTENSIONS
        recursive (bool): 是否递归扫描子目录，默认为 True
    
    Returns:
        list: 找到的所有文档文件路径列表
    """
    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS
    
    file_paths = []
    
    # 确保目录存在
    if not os.path.exists(data_dir):
        print(f"警告: 文档目录不存在: {data_dir}")
        return file_paths
    
    # 根据是否递归扫描，选择不同的匹配模式
    if recursive:
        # 递归扫描所有子目录
        for ext in extensions:
            pattern = os.path.join(data_dir, '**', f'*{ext}')
            file_paths.extend(glob.glob(pattern, recursive=True))
    else:
        # 只扫描当前目录
        for ext in extensions:
            pattern = os.path.join(data_dir, f'*{ext}')
            file_paths.extend(glob.glob(pattern))
    
    # 去重并排序
    file_paths = sorted(list(set(file_paths)))
    
    return file_paths

# 文档路径 - 自动扫描 documents 目录
FILE_PATHS = get_all_document_files()

# 如果需要手动指定特定文件，可以取消下面的注释并添加路径
# FILE_PATHS = [
#     os.path.join(DATA_DIR, "your_file.txt"),
# ]

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
