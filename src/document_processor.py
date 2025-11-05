from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from . import config

class DocumentProcessor:
    """
    统一处理文档加载和文本分割。
    """
    def __init__(self, chunk_size=None, chunk_overlap=None):
        """
        初始化文档处理器。

        Args:
            chunk_size (int, optional): 每个文本块的最大长度。默认为配置文件中的值。
            chunk_overlap (int, optional): 文本块之间的重叠长度。默认为配置文件中的值。
        """
        chunk_size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
        chunk_overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def load_and_split(self, file_paths: list) -> list:
        """
        从文件路径列表加载文档，并将其分割成文本块。

        Args:
            file_paths (list): 包含文档文件路径的列表。

        Returns:
            list: 分割后的 LangChain Document 对象列表。
        """
        all_split_documents = []
        print(f"正在从 {len(file_paths)} 个文件加载和分割文档...")
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # --- 诊断日志 ---
                content_length = len(content)
                print(f"  - 正在处理文件: {file_path}, 读取到 {content_length} 个字符。")
                # ---------------

                # 如果文件内容为空，则跳过
                if content_length == 0:
                    print(f"  - 警告: 文件为空，已跳过。")
                    continue

                # 将文件内容包装成Document对象
                document = Document(page_content=content, metadata={"source": file_path})
                
                # 分割文档
                split_docs = self.text_splitter.split_documents([document])
                all_split_documents.extend(split_docs)
                print(f"  - 成功分割成 {len(split_docs)} 个块。")

            except FileNotFoundError:
                print(f"  - 错误: 文件未找到: {file_path}")
            except Exception as e:
                print(f"  - 加载或处理文件时出错: {file_path}, 错误: {e}")

        total_chunks = len(all_split_documents)
        print(f"文档处理完成，总共生成 {total_chunks} 个文本块。")
        return all_split_documents
