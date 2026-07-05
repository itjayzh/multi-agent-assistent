# customer_support_chat/app/services/tools/blog.py

import httpx
import urllib.parse
from langchain_core.tools import tool
from customer_support_chat.app.core.settings import get_settings
from typing import List, Dict

settings = get_settings()

@tool
def search_blog_posts(keyword: str, limit: int = 5) -> List[Dict]:
    """根据关键词搜索博客文章。
    
    Args:
        keyword: 用于搜索博客文章的关键词。
        limit: 最多返回的文章数量（默认 5）。
        
    Returns:
        包含标题、摘要和链接的博客文章字典列表。
    """
    if not settings.BLOG_SEARCH_API_URL:
        raise ValueError("博客搜索 API URL 未配置。")
    
    # 对关键词进行 URL 编码
    encoded_keyword = urllib.parse.quote(keyword)
    url = f"{settings.BLOG_SEARCH_API_URL}?search={encoded_keyword}"
    
    # 如果博客接口需要认证（如 WooCommerce REST API），可在这里添加
    # 当前默认它是公开的 WordPress REST API 端点
    auth = None
    if settings.WOOCOMMERCE_CONSUMER_KEY and settings.WOOCOMMERCE_CONSUMER_SECRET:
        auth = httpx.BasicAuth(settings.WOOCOMMERCE_CONSUMER_KEY, settings.WOOCOMMERCE_CONSUMER_SECRET)
    
    with httpx.Client() as client:
        try:
            response = client.get(
                url,
                auth=auth
            )
            response.raise_for_status()
            posts = response.json()
            
            # 提取关键信息
            simplified_posts = []
            for post in posts[:limit]:
                simplified_posts.append({
                    "id": post.get("id"),
                    "title": post.get("title", {}).get("rendered", "无标题"),
                    "excerpt": post.get("excerpt", {}).get("rendered", "")[:200] + "...",
                    "link": post.get("link"),
                    "date": post.get("date"),
                })
            
            return simplified_posts
        except httpx.HTTPStatusError as e:
            raise Exception(f"搜索博客文章时发生 HTTP 错误: {e}")
        except Exception as e:
            raise Exception(f"搜索博客文章时发生错误: {e}")
