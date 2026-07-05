from vectorizer.app.vectordb.vectordb import VectorDB
from customer_support_chat.app.core.settings import get_settings
from langchain_core.tools import tool
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

settings = get_settings()
faq_vectordb = VectorDB(table_name="faq", collection_name="faq_collection")

@tool
def search_faq(
    query: str,
    limit: int = 2,
) -> List[Dict]:
    """根据自然语言查询搜索 FAQ 条目。"""
    search_results = faq_vectordb.search(query, limit=limit)

    faq_entries = []
    for result in search_results:
        payload = result.payload
        content = payload.get("content", "")
        
        # 如果内容符合编号问答格式，尝试解析其中的问答
        question = "常规 FAQ 信息"
        answer = content
        category = "常见问题"
        
        # 查找带编号的问题格式（例如 “1. 我可以……”）
        import re
        question_match = re.search(r'^\d+\. (.+?)(?=\n|$)', content, re.MULTILINE)
        if question_match:
            question = question_match.group(1).strip()
            # 提取答案（问题之后的全部内容）
            answer_start = content.find(question) + len(question)
            answer = content[answer_start:].strip()
        elif content.startswith('##'):
            # 处理章节标题
            lines = content.split('\n', 1)
            question = lines[0].replace('##', '').strip()
            answer = lines[1] if len(lines) > 1 else "详情请查看该章节内容。"
        
        faq_entries.append({
            "question": question,
            "answer": answer,
            "category": category,
            "chunk": content,
            "similarity": result.score,
        })
    return faq_entries

@tool
def lookup_policy(query: str) -> str:
    """查询公司政策，确认某些操作是否被允许。
    在改签航班或执行其他写入类操作前，应先调用此工具。"""
    faq_results = search_faq.invoke({"query": query, "limit": 2})
    if not faq_results:
        return "抱歉，我没有找到相关的政策信息。请联系客服获取帮助。"
    
    policy_info = "\n\n".join([f"问：{entry['question']}\n答：{entry['answer']}" for entry in faq_results])
    return f"以下是相关的政策信息：\n\n{policy_info}"
