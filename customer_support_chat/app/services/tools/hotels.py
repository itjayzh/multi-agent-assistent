from datetime import date, datetime
from typing import Dict, List, Union

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from customer_support_chat.app.core.database import get_connection
from customer_support_chat.app.services.order_service import (
    cancel_order,
    create_hotel_order,
    update_hotel_order,
)
from vectorizer.app.vectordb.vectordb import VectorDB


hotels_vectordb = VectorDB(table_name="hotels", collection_name="hotels_collection")


def get_passenger_id(config: RunnableConfig) -> str:
    passenger_id = config.get("configurable", {}).get("passenger_id")
    if not passenger_id:
        raise ValueError("未配置乘机人 ID。")
    return passenger_id


@tool
def search_hotels(query: str, limit: int = 2) -> List[Dict]:
    """用向量库召回酒店，并从 PostgreSQL 返回当前价格和状态。"""
    search_results = hotels_vectordb.search(query, limit=limit)
    hotels = []
    for result in search_results:
        payload = result.payload
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT h.id, h.name, h.location, h.price_tier,
                           h.checkin_date, h.checkout_date, h.booked,
                           p.base_amount_minor, p.currency
                    FROM hotels h
                    JOIN products p
                      ON p.product_type = 'hotel'
                     AND p.external_product_id = %s
                     AND p.active = TRUE
                    WHERE h.id = %s
                    """,
                    (f"legacy-hotel-{payload['id']}", payload["id"]),
                )
                hotel = cursor.fetchone()
        if hotel is None:
            continue
        hotels.append({
            "id": hotel[0],
            "name": hotel[1],
            "location": hotel[2],
            "price_tier": hotel[3],
            "checkin_date": hotel[4],
            "checkout_date": hotel[5],
            "booked": hotel[6],
            "amount_minor": hotel[7],
            "currency": hotel[8],
            "chunk": payload["content"],
            "similarity": result.score,
        })
    return hotels


@tool
async def book_hotel(hotel_id: int, *, config: RunnableConfig) -> str:
    """根据酒店产品 ID 创建当前用户的正式订单。"""
    return create_hotel_order(hotel_id, get_passenger_id(config))


@tool
async def update_hotel(
    order_id: int,
    checkin_date: Union[datetime, date],
    checkout_date: Union[datetime, date],
    *,
    config: RunnableConfig,
) -> str:
    """根据正式订单 ID 修改当前用户的酒店入住日期。"""
    return update_hotel_order(
        order_id,
        get_passenger_id(config),
        checkin_date,
        checkout_date,
    )


@tool
async def cancel_hotel(order_id: int, *, config: RunnableConfig) -> str:
    """根据正式订单 ID 取消当前用户的酒店订单。"""
    return cancel_order(order_id, get_passenger_id(config))
