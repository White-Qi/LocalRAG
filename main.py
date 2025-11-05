"""
RAG系统入口文件
整合各模块功能，提供完整的问答流程
"""
import argparse
from src.rag_pipeline import RAGPipeline


def main():
    """主函数，运行RAG系统"""
    parser = argparse.ArgumentParser(description="一个基于知识库的问答系统")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="强制重新构建向量索引"
    )
    args = parser.parse_args()

    # 初始化核心管线
    pipeline = RAGPipeline()

    # 准备索引（加载或构建）
    pipeline.setup(force_reindex=args.reindex)

    # 进入交互式问答循环
    print("\n系统准备就绪，可以开始提问了。")
    try:
        while True:
            query = input("\n请输入您的问题 (输入 'quit' 或 Ctrl+C 退出): ").strip()
            if not query:
                print("问题不能为空，请重新输入。")
                continue
            if query.lower() == 'quit':
                break
            
            # 获取答案
            answer = pipeline.ask(query)
            print(f"\n答案:\n{answer}")

    except (KeyboardInterrupt, EOFError):
        print("\n程序已退出。")


if __name__ == "__main__":
    main()