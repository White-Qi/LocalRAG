import requests
from . import config

PROMPT_TEMPLATE = """
你是一个严谨的问答助手。请根据以下提供的上下文信息，用中文准确地回答用户的问题。

重要规则：
1. 答案必须**基于**提供的上下文。
2. 如果问题要求具体内容（如"第几句"、"原文"等），请**直接引用**上下文中的原文。
3. 保持客观，合理的陈述上下文中已有的信息。

上下文：
---
{context}
---

问题：{query}

答案：
"""

class AnswerGenerator:
    """
    调用大语言模型生成答案。
    """
    def __init__(self):
        self.model_name = config.OLLAMA_QA_MODEL
        self.base_url = config.OLLAMA_BASE_URL

    def generate(self, query: str, context: str) -> str:
        """
        根据查询和上下文生成答案。

        Args:
            query (str): 用户查询。
            context (str): 从向量数据库检索到的上下文信息。

        Returns:
            str: 生成的答案。
        """
        print("正在生成最终答案...")
        prompt = PROMPT_TEMPLATE.format(context=context, query=query)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            answer = result.get("response", "未能生成答案。")
            print("答案生成完成。")
            return answer
        except requests.RequestException as e:
            print(f"调用Ollama生成API时出错: {e}")
            return "调用问答模型时遇到网络错误，请检查Ollama服务是否正在运行。"
