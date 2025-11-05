import faiss
import numpy as np
import requests
from . import config
import os

# 动态导入 sentence-transformers 库，增强代码灵活性
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import torch
    TRANSFORMERS_AVAILABLE = True
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    SentenceTransformer, CrossEncoder, torch = None, None, None
    GPU_AVAILABLE = False


class VectorStore:
    """
    统一管理向量化、索引构建、检索和重排序。
    """
    def __init__(self):
        self.embedding_model = None
        self.reranker = None
        self.index = None
        self.documents = []  # 存储与索引向量对应的文档
        self._load_models()

        if config.USE_LOCAL_EMBEDDING_MODEL:
            self.dimension = config.LOCAL_EMBEDDING_DIMENSION
        else:
            self.dimension = config.OLLAMA_EMBEDDING_DIMENSION

    def _load_models(self):
        """根据配置加载向量化模型和重排序模型。"""
        if not TRANSFORMERS_AVAILABLE:
            if config.USE_LOCAL_EMBEDDING_MODEL or config.USE_RERANKER:
                print("警告: sentence-transformers 未安装，无法使用本地模型或重排序器。")
                print("请运行 'pip install sentence-transformers torch' 进行安装。")
            return

        # 加载本地向量化模型
        if config.USE_LOCAL_EMBEDDING_MODEL:
            print(f"正在加载本地向量化模型: {config.LOCAL_EMBEDDING_MODEL}...")
            try:
                self.embedding_model = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
                if GPU_AVAILABLE:
                    self.embedding_model.to('cuda')
                    print("检测到GPU，向量化模型将运行在GPU上。")
                else:
                    print("未检测到GPU，向量化模型将运行在CPU上。")
            except Exception as e:
                print(f"加载本地向量化模型失败: {e}")

        # 加载重排序模型
        if config.USE_RERANKER:
            print(f"正在加载重排序模型: {config.RERANKER_MODEL}...")
            try:
                self.reranker = CrossEncoder(config.RERANKER_MODEL)
                if GPU_AVAILABLE:
                    # CrossEncoder对多GPU支持不佳，通常在单个GPU上表现良好
                    print("重排序模型加载完成。")
                else:
                    print("重排序模型将运行在CPU上。")
            except Exception as e:
                print(f"加载重排序模型失败: {e}")

    def build_index(self, documents: list, force_reindex=False):
        """
        构建或加载向量索引。

        Args:
            documents (list): 待索引的LangChain Document对象列表。
            force_reindex (bool): 是否强制重新构建索引。
        """
        self.documents = documents
        if not force_reindex and os.path.exists(config.INDEX_FILE):
            print(f"正在从 {config.INDEX_FILE} 加载现有索引...")
            self.index = faiss.read_index(config.INDEX_FILE)
            print("索引加载完成。")
            return

        print("正在为文档创建向量嵌入...")
        texts = [doc.page_content for doc in self.documents]
        vectors = self._embed_texts(texts)
        
        if not vectors:
            print("错误: 未能生成任何向量，无法构建索引。")
            return

        print("正在构建Faiss索引...")
        vectors_np = np.array(vectors).astype('float32')
        
        # 检查并更新实际的向量维度
        actual_dimension = vectors_np.shape[1]
        print(f"配置的维度: {self.dimension}, 实际向量维度: {actual_dimension}")
        
        if actual_dimension != self.dimension:
            print(f"警告: 实际向量维度 ({actual_dimension}) 与配置维度 ({self.dimension}) 不匹配！")
            print(f"自动调整为实际维度: {actual_dimension}")
            self.dimension = actual_dimension
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(vectors_np)
        
        print("正在保存索引...")
        faiss.write_index(self.index, config.INDEX_FILE)
        print(f"索引已保存至 {config.INDEX_FILE}")

    def retrieve(self, query: str) -> list:
        """
        接收查询，执行检索和重排序，返回最相关的文档内容。

        Args:
            query (str): 用户的查询字符串。

        Returns:
            list: 包含最相关文档内容的字符串列表。
        """
        if self.index is None:
            print("错误: 索引未构建或加载。")
            return []
        
        # 验证查询字符串
        query = query.strip()
        if not query:
            print("错误: 查询字符串为空。")
            return []

        print("正在为查询创建向量嵌入...")
        query_embeddings = self._embed_texts([query])
        if not query_embeddings:
            print("错误: 查询向量化失败。")
            return []
        query_vector = query_embeddings[0]
        query_vector_np = np.array([query_vector]).astype('float32')
        
        # 确定初步检索的数量
        k = config.INITIAL_RETRIEVAL_TOP_K if self.reranker else config.RETRIEVER_TOP_K
        
        print(f"正在执行向量检索 (Top {k})...")
        distances, indices = self.index.search(query_vector_np, k)
        
        retrieved_docs_info = []
        for i, idx in enumerate(indices[0]):
            if idx != -1: # Faiss在结果不足k时可能返回-1
                retrieved_docs_info.append({
                    "document": self.documents[idx],
                    "distance": distances[0][i]
                })

        if not self.reranker:
            return [info["document"].page_content for info in retrieved_docs_info]

        print(f"正在对 {len(retrieved_docs_info)} 个文档进行重排序...")
        pairs = [[query, info["document"].page_content] for info in retrieved_docs_info]
        scores = self.reranker.predict(pairs)
        
        for info, score in zip(retrieved_docs_info, scores):
            info['rerank_score'] = score
            
        reranked_docs = sorted(retrieved_docs_info, key=lambda x: x['rerank_score'], reverse=True)
        
        final_docs = reranked_docs[:config.RERANKER_TOP_K]
        print(f"重排序完成，选出 Top {len(final_docs)} 个文档。")
        
        return [info["document"].page_content for info in final_docs]

    def _embed_texts(self, texts: list) -> list:
        """根据配置选择向量化方式。"""
        if config.USE_LOCAL_EMBEDDING_MODEL and self.embedding_model:
            return self._embed_with_local_model(texts)
        else:
            return self._embed_with_ollama_api(texts)

    def _embed_with_local_model(self, texts: list) -> list:
        """使用本地模型进行向量化。"""
        print(f"正在使用本地模型向量化 {len(texts)} 个文本...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True).tolist()
        return embeddings

    def _embed_with_ollama_api(self, texts: list) -> list:
        """使用Ollama API进行向量化。"""
        print(f"正在使用Ollama API向量化 {len(texts)} 个文本...")
        embeddings = []
        for i, text in enumerate(texts):
            print(f"\r  - 正在处理文本 {i+1}/{len(texts)}...", end="")
            # 确保文本不为空
            if not text or not text.strip():
                print(f"\n警告: 文本 {i+1} 为空，跳过向量化。")
                continue
            try:
                payload = {"model": config.OLLAMA_EMBEDDING_MODEL, "prompt": text}
                response = requests.post(f"{config.OLLAMA_BASE_URL}/api/embeddings", json=payload, timeout=30)
                response.raise_for_status()
                embedding = response.json().get("embedding")
                if embedding:
                    embeddings.append(embedding)
                else:
                    print(f"\n警告: 文本 {i+1} 未返回有效的embedding。")
            except requests.RequestException as e:
                print(f"\n调用Ollama嵌入API时出错: {e}")
                print(f"文本内容: {text[:100]}...")  # 打印前100个字符用于调试
        print("\n向量化完成。")
        if len(embeddings) != len(texts):
            print(f"警告: 输入了 {len(texts)} 个文本，但只成功向量化了 {len(embeddings)} 个。")
        return embeddings
