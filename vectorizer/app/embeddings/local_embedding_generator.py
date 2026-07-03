"""
使用 sentence-transformers 的本地嵌入生成器
无需 API 调用，可离线运行
"""

import sys
import os
from typing import Union, List
import numpy as np

# 将项目根目录加入 Python 路径，便于导入模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_root)

try:
    from vectorizer.app.core.settings import get_settings
    from vectorizer.app.core.logger import logger
    settings = get_settings()
except ImportError:
    # 直接执行脚本时的兜底逻辑
    print("以独立模式运行，使用默认配置")
    
    class MockSettings:
        LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    settings = MockSettings()
    
    # 简单日志兜底
    class SimpleLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
    
    logger = SimpleLogger()

# 全局模型实例
_model = None

def get_local_model():
    """获取或初始化本地嵌入模型"""
    global _model
    
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            model_name = settings.LOCAL_EMBEDDING_MODEL
            logger.info(f"正在加载本地嵌入模型: {model_name}")
            
            # 尝试加载模型
            _model = SentenceTransformer(model_name)
            logger.info(f"本地模型加载成功: {model_name}")
            logger.info(f"模型嵌入维度: {_model.get_sentence_embedding_dimension()}")
            
        except ImportError:
            logger.error("未安装 sentence-transformers，请执行: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"加载本地模型 {model_name} 失败: {str(e)}")
            logger.info("尝试加载兜底模型: all-MiniLM-L6-v2")
            try:
                _model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("兜底模型加载成功: all-MiniLM-L6-v2")
            except Exception as fallback_error:
                logger.error(f"加载兜底模型失败: {str(fallback_error)}")
                raise
    
    return _model

def generate_local_embedding(content: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    """
    使用本地 sentence-transformers 模型生成嵌入向量
    
    Args:
        content: 需要生成嵌入的字符串或字符串列表
        
    Returns:
        以浮点数列表形式返回的嵌入向量
    """
    try:
        model = get_local_model()
        
        if isinstance(content, str):
            # 单个字符串
            embedding = model.encode(content)
            return embedding.tolist()
            
        elif isinstance(content, list):
            # 字符串列表
            embeddings = model.encode(content)
            return [emb.tolist() for emb in embeddings]
            
        else:
            raise ValueError("content 必须是字符串或字符串列表")
            
    except Exception as e:
        logger.error(f"生成本地嵌入时出错: {str(e)}")
        raise

def test_local_embeddings():
    """测试本地嵌入生成功能"""
    try:
        logger.info("开始测试本地嵌入生成...")
        
        test_texts = [
            "你好，世界",
            "这是一条测试语句",
            "本地嵌入功能正常工作！"
        ]
        
        # 测试单个字符串
        single_embedding = generate_local_embedding(test_texts[0])
        logger.info(f"单条嵌入长度: {len(single_embedding)}")
        logger.info(f"前 5 个值: {single_embedding[:5]}")
        
        # 测试多个字符串
        batch_embeddings = generate_local_embedding(test_texts)
        logger.info(f"批量嵌入形状: {len(batch_embeddings)} x {len(batch_embeddings[0])}")
        
        logger.info("本地嵌入测试成功！")
        return True
        
    except Exception as e:
        logger.error(f"本地嵌入测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    # 直接运行脚本时执行测试
    test_local_embeddings()
