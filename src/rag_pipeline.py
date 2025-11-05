import os
from . import config
from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from .answer_generator import AnswerGenerator


class RAGPipeline:
    """
    整合RAG所有流程的核心管线。
    """
    def __init__(self):
        print("正在初始化RAG核心管线...")
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.answer_generator = AnswerGenerator()
        
        # 确保索引目录存在
        if not os.path.exists(config.INDEX_DIR):
            os.makedirs(config.INDEX_DIR)
        print("RAG核心管线初始化完成。")

    def setup(self, force_reindex=False):
        """
        设置或加载向量索引。

        Args:
            force_reindex (bool): 是否强制重新索引。
        """
        # 显示找到的文档文件
        if config.FILE_PATHS:
            print(f"\n自动发现 {len(config.FILE_PATHS)} 个文档文件:")
            for i, file_path in enumerate(config.FILE_PATHS, 1):
                file_name = os.path.basename(file_path)
                print(f"  {i}. {file_name}")
        else:
            print(f"\n警告: 在 {config.DATA_DIR} 目录中未找到任何文档文件。")
            print(f"支持的文件格式: {', '.join(config.SUPPORTED_EXTENSIONS)}")
            print("请将文档文件放入 documents 目录后重新运行。")
            return
        
        # 1. 加载和分割文档
        documents = self.doc_processor.load_and_split(config.FILE_PATHS)
        if not documents:
            print("未能加载或分割任何文档，流程中止。")
            return
            
        # 2. 构建或加载索引
        self.vector_store.build_index(documents, force_reindex)

    def ask(self, query: str) -> str:
        """
        接收问题，执行完整的RAG流程，并返回答案。

        Args:
            query (str): 用户输入的问题。

        Returns:
            str: 生成的答案。
        """
        print(f"\n接收到新问题: {query}")
        # 1. 检索相关文档
        retrieved_context = self.vector_store.retrieve(query)
        
        if not retrieved_context:
            return "未能从知识库中检索到相关信息来回答此问题。"
            
        context_str = "\n---\n".join(retrieved_context)

        # 2. 基于上下文生成答案
        answer = self.answer_generator.generate(query, context_str)
        return answer
