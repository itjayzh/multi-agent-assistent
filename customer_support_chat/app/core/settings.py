from os import environ
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[3]
load_dotenv(project_root / ".env")
load_dotenv(project_root / ".env.local", override=True)


class Config:
    OPENAI_API_KEY: str = environ.get("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = environ.get("OPENAI_BASE_URL", "")

    # 对话模型配置
    OPENAI_MODEL: str = environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
    GUARDRAIL_MODEL: str = environ.get("GUARDRAIL_MODEL", environ.get("OPENAI_MODEL", "gpt-3.5-turbo"))
    MAX_TOKENS: int = int(environ.get("MAX_TOKENS", "1000"))  # 限制 token 以控制成本
    ENABLE_APPROVALS: bool = environ.get("ENABLE_APPROVALS", "true").lower() == "true"
    FEISHU_APP_ID: str = environ.get("FEISHU_APP_ID", "")
    FEISHU_APP_SECRET: str = environ.get("FEISHU_APP_SECRET", "")
    FEISHU_APPROVAL_CODE: str = environ.get("FEISHU_APPROVAL_CODE", "")
    FEISHU_APPLICANT_USER_ID: str = environ.get("FEISHU_APPLICANT_USER_ID", "")
    FEISHU_APPROVER_USER_ID: str = environ.get("FEISHU_APPROVER_USER_ID", "")
    FEISHU_FORM_FIELD_ID: str = environ.get("FEISHU_FORM_FIELD_ID", "")
    FEISHU_VERIFICATION_TOKEN: str = environ.get("FEISHU_VERIFICATION_TOKEN", "")
    FEISHU_ENABLED: bool = environ.get("FEISHU_ENABLED", "false").lower() == "true"
    FEISHU_API_BASE_URL: str = environ.get("FEISHU_API_BASE_URL", "https://open.feishu.cn")
    FEISHU_SDK_LOG_LEVEL: str = environ.get("FEISHU_SDK_LOG_LEVEL", "WARNING")

    # 向量模型配置
    EMBEDDING_API_KEY: str = environ.get("EMBEDDING_API_KEY", environ.get("OPENAI_API_KEY", ""))
    EMBEDDING_BASE_URL: str = environ.get("EMBEDDING_BASE_URL", environ.get("OPENAI_BASE_URL", ""))
    EMBEDDING_MODEL: str = environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

    DATA_PATH: str = "DATA_PATH"
    LOG_LEVEL: str = environ.get("LOG_LEVEL", "DEBUG")
    DATABASE_HOST: str = environ.get("DATABASE_HOST", "localhost")
    DATABASE_PORT: int = int(environ.get("DATABASE_PORT", "5432"))
    DATABASE_NAME: str = environ.get("DATABASE_NAME", "multi_agent")
    DATABASE_USER: str = environ.get("DATABASE_USER", "postgres")
    DATABASE_PASSWORD: str = environ.get("DATABASE_PASSWORD", "")
    QDRANT_URL: str = environ.get("QDRANT_URL", "http://localhost:6333")
    QDRANT_KEY: str = environ.get("QDRANT_KEY", "")
    RECREATE_COLLECTIONS: bool = environ.get("RECREATE_COLLECTIONS", "False")
    LIMIT_ROWS: int = environ.get("LIMIT_ROWS", "100")

    # WooCommerce API 配置
    # WOOCOMMERCE_API_URL 应为 WordPress 站点基础 URL（例如 "https://yourstore.com"）
    # 系统会自动追加 "/wp-json/wc/v3" 以生成完整接口地址
    WOOCOMMERCE_CONSUMER_KEY: str = environ.get("WOOCOMMERCE_CONSUMER_KEY", "")
    WOOCOMMERCE_CONSUMER_SECRET: str = environ.get("WOOCOMMERCE_CONSUMER_SECRET", "")
    WOOCOMMERCE_API_URL: str = environ.get("WOOCOMMERCE_API_URL", "")

    # 表单提交 API 配置
    FORM_SUBMISSION_API_URL: str = environ.get("FORM_SUBMISSION_API_URL", "")

    # 博客搜索 API 配置
    BLOG_SEARCH_API_URL: str = environ.get("BLOG_SEARCH_API_URL", "")


def get_settings():
    return Config()
