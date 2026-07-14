from datetime import date, datetime
from typing import Dict, List

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from customer_support_chat.app.core.database import get_connection
from customer_support_chat.app.services.order_service import (
    cancel_order,
    create_trip_order,
    update_trip_order,
)
from vectorizer.app.vectordb.vectordb import VectorDB


excursions_vectordb = VectorDB(
    table_name="trip_recommendations",
    collection_name="excursions_collection",
)


def get_passenger_id(config: RunnableConfig) -> str:
    passenger_id = config.get("configurable", {}).get("passenger_id")
    if not passenger_id:
        raise ValueError("未配置乘机人 ID。")
    return passenger_id


@tool
def search_trip_recommendations(query: str, limit: int = 2) -> List[Dict]:
    """用向量库召回行程，并从 PostgreSQL 返回当前价格和状态。"""
    search_results = excursions_vectordb.search(query, limit=limit)
    recommendations = []
    for result in search_results:
        payload = result.payload
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT t.id, t.name, t.location, t.keywords, t.details,
                           t.booked,
                           p.base_amount_minor, p.currency
                    FROM trip_recommendations t
                    JOIN products p
                      ON p.product_type = 'trip'
                     AND p.external_product_id = %s
                     AND p.active = TRUE
                    WHERE t.id = %s
                    """,
                    (f"legacy-trip-{payload['id']}", payload["id"]),
                )
                recommendation = cursor.fetchone()
        if recommendation is None:
            continue
        recommendations.append({
            "id": recommendation[0],
            "name": recommendation[1],
            "location": recommendation[2],
            "keywords": recommendation[3],
            "details": recommendation[4],
            "booked": recommendation[5],
            "amount_minor": recommendation[6],
            "currency": recommendation[7],
            "chunk": payload["content"],
            "similarity": result.score,
        })
    return recommendations


@tool
async def book_excursion(
    recommendation_id: int,
    *,
    config: RunnableConfig,
) -> str:
    """根据行程产品 ID 创建当前用户的正式订单。"""
    return create_trip_order(recommendation_id, get_passenger_id(config))


@tool
async def update_excursion(
    order_id: int,
    visit_date: date | datetime,
    participant_count: int = 1,
    *,
    config: RunnableConfig,
) -> str:
    """根据正式订单 ID 修改当前用户的行程日期和人数。"""
    return update_trip_order(
        order_id,
        get_passenger_id(config),
        visit_date,
        participant_count,
    )


@tool
async def cancel_excursion(order_id: int, *, config: RunnableConfig) -> str:
    """根据正式订单 ID 取消当前用户的行程订单。"""
    return cancel_order(order_id, get_passenger_id(config))
