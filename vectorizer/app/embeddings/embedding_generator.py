from langchain_openai import OpenAIEmbeddings
from vectorizer.app.core.settings import get_settings
from vectorizer.app.core.logger import logger
from typing import Union, List

settings = get_settings()

# 使用嵌入专用配置来初始化 OpenAI embeddings
if settings.EMBEDDING_BASE_URL:
    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.EMBEDDING_API_KEY,
        openai_api_base=settings.EMBEDDING_BASE_URL
    )
else:
    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.EMBEDDING_API_KEY
    )

def generate_embedding(content: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    """
    根据配置使用 API 或本地模型生成嵌入向量
    """
    # 检查是否启用了本地嵌入
    if settings.USE_LOCAL_EMBEDDINGS:
        logger.info("使用本地嵌入")
        try:
            from .local_embedding_generator import generate_local_embedding
            return generate_local_embedding(content)
        except Exception as e:
            logger.error(f"本地嵌入生成失败: {str(e)}")
            logger.info("回退到 API 嵌入...")
    
    # 使用 API 嵌入
    logger.info("使用 API 嵌入")
    try:
        if isinstance(content, str):
            return embeddings.embed_query(content)
        elif isinstance(content, list):
            return embeddings.embed_documents(content)
        else:
            raise ValueError("content 必须是字符串或字符串列表")
    except Exception as e:
        logger.error(f"API 嵌入生成失败: {str(e)}")
        
        # 如果 API 失败且未启用本地嵌入，则尝试使用本地嵌入兜底
        if not settings.USE_LOCAL_EMBEDDINGS:
            logger.info("API 调用失败，尝试使用本地嵌入作为兜底...")
            try:
                from .local_embedding_generator import generate_local_embedding
                return generate_local_embedding(content)
            except Exception as local_e:
                logger.error(f"本地嵌入兜底也失败了: {str(local_e)}")
                raise Exception(f"API 和本地嵌入都失败了。API 错误: {str(e)}，本地错误: {str(local_e)}")
        else:
            raise
