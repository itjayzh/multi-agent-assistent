from datetime import date, datetime
from typing import Dict, List, Union

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from customer_support_chat.app.core.database import get_connection
from customer_support_chat.app.services.order_service import (
    cancel_order,
    create_car_order,
    update_car_order,
)
from vectorizer.app.vectordb.vectordb import VectorDB


cars_vectordb = VectorDB(table_name="car_rentals", collection_name="car_rentals_collection")


def get_passenger_id(config: RunnableConfig) -> str:
    passenger_id = config.get("configurable", {}).get("passenger_id")
    if not passenger_id:
        raise ValueError("未配置乘机人 ID。")
    return passenger_id


@tool
def search_car_rentals(query: str, limit: int = 2) -> List[Dict]:
    """用向量库召回车辆，并从 PostgreSQL 返回当前价格和状态。"""
    search_results = cars_vectordb.search(query, limit=limit)
    rentals = []
    for result in search_results:
        payload = result.payload
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT c.id, c.name, c.location, c.price_tier,
                           c.start_date, c.end_date, c.booked,
                           p.base_amount_minor, p.currency
                    FROM car_rentals c
                    JOIN products p
                      ON p.product_type = 'car'
                     AND p.external_product_id = %s
                     AND p.active = TRUE
                    WHERE c.id = %s
                    """,
                    (f"legacy-car-{payload['id']}", payload["id"]),
                )
                rental = cursor.fetchone()
        if rental is None:
            continue
        rentals.append({
            "id": rental[0],
            "name": rental[1],
            "location": rental[2],
            "price_tier": rental[3],
            "start_date": rental[4],
            "end_date": rental[5],
            "booked": rental[6],
            "amount_minor": rental[7],
            "currency": rental[8],
            "chunk": payload["content"],
            "similarity": result.score,
        })
    return rentals


@tool
async def book_car_rental(rental_id: int, *, config: RunnableConfig) -> str:
    """根据车辆产品 ID 创建当前用户的正式订单。"""
    return create_car_order(rental_id, get_passenger_id(config))


@tool
async def update_car_rental(
    order_id: int,
    start_date: Union[datetime, date],
    end_date: Union[datetime, date],
    *,
    config: RunnableConfig,
) -> str:
    """根据正式订单 ID 修改当前用户的租车日期。"""
    return update_car_order(
        order_id,
        get_passenger_id(config),
        start_date,
        end_date,
    )


@tool
async def cancel_car_rental(order_id: int, *, config: RunnableConfig) -> str:
    """根据正式订单 ID 取消当前用户的租车订单。"""
    return cancel_order(order_id, get_passenger_id(config))
