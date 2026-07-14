from psycopg import Connection, connect
from psycopg.rows import dict_row

from customer_support_chat.app.core.settings import get_settings


def get_connection(*, rows_as_dict: bool = False) -> Connection:
    """创建 PostgreSQL 数据库连接。"""
    settings = get_settings()
    row_factory = dict_row if rows_as_dict else None
    return connect(
        host=settings.DATABASE_HOST,
        port=settings.DATABASE_PORT,
        dbname=settings.DATABASE_NAME,
        user=settings.DATABASE_USER,
        password=settings.DATABASE_PASSWORD,
        row_factory=row_factory,
    )
