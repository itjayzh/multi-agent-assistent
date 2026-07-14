from typing import Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from customer_support_chat.app.services.order_service import (
    get_order_detail,
    query_user_orders,
    retry_supplier_order,
)


def get_user_id(config: RunnableConfig) -> int:
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        raise ValueError("未配置登录用户 ID。")
    return int(user_id)


@tool
def fetch_user_orders(
    status: Optional[str] = None,
    order_type: Optional[str] = None,
    *,
    config: RunnableConfig,
) -> List[Dict]:
    """从 PostgreSQL 查询当前用户的订单和最新状态。"""
    result = query_user_orders(
        get_user_id(config),
        status=status,
        order_type=order_type,
        page=1,
        page_size=50,
    )
    return result["items"]


@tool
def fetch_order_detail(order_id: int, *, config: RunnableConfig) -> Dict:
    """从 PostgreSQL 查询当前用户指定订单的业务明细和状态历史。"""
    return get_order_detail(get_user_id(config), order_id)


@tool
async def retry_order_booking(order_id: int, *, config: RunnableConfig) -> str:
    """重试当前用户失败的正式订单供应商下单。"""
    passenger_id = config.get("configurable", {}).get("passenger_id")
    if not passenger_id:
        raise ValueError("未配置乘机人 ID。")
    return retry_supplier_order(order_id, passenger_id)
