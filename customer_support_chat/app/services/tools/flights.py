from vectorizer.app.vectordb.vectordb import VectorDB
from customer_support_chat.app.core.database import get_connection
from customer_support_chat.app.services.order_service import (
    cancel_order,
    create_flight_order,
    update_flight_order,
)
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from typing import Optional, List, Dict
from datetime import datetime, timezone

flights_vectordb = VectorDB(table_name="flights", collection_name="flights_collection")


def normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@tool
def fetch_user_flight_information(*, config: RunnableConfig) -> List[Dict]:
    """获取用户的全部机票信息，以及对应的航班信息和座位分配。"""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("未配置乘客 ID。")

    conn = get_connection()
    cursor = conn.cursor()

    query = """
    SELECT 
        t.ticket_no, t.book_ref,
        f.flight_id, f.flight_no, f.departure_airport, f.arrival_airport, f.scheduled_departure, f.scheduled_arrival,
        bp.seat_no, tf.fare_conditions
    FROM 
        tickets t
        JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
        JOIN flights f ON tf.flight_id = f.flight_id
        LEFT JOIN boarding_passes bp ON bp.ticket_no = t.ticket_no AND bp.flight_id = f.flight_id
    WHERE 
        t.passenger_id = %s

    UNION ALL

    SELECT
        fop.ticket_no, fo.booking_reference AS book_ref,
        split_part(p.external_product_id, '-', 3)::BIGINT AS flight_id,
        fos.flight_no, fos.departure_airport, fos.arrival_airport,
        fos.departure_at AS scheduled_departure,
        fos.arrival_at AS scheduled_arrival,
        fop.seat_no, fos.fare_conditions
    FROM orders o
        JOIN users u ON u.id = o.user_id
        JOIN flight_orders fo ON fo.order_id = o.id
        JOIN flight_order_segments fos ON fos.order_id = o.id
        JOIN products p ON p.id = fos.flight_product_id
        JOIN flight_order_passengers fop ON fop.order_id = o.id
    WHERE u.passenger_id = %s
      AND o.status = 'confirmed'
    """
    cursor.execute(query, (passenger_id, passenger_id))
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()

    return results

@tool
def search_flights(
    query: str,
    limit: int = 5,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> List[Dict]:
    """根据自然语言和可选起止时间搜索航班。"""
    search_results = flights_vectordb.search(query, limit=max(limit, 20))

    flights = []
    for result in search_results:
        payload = result.payload
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT f.flight_id, f.flight_no, f.departure_airport,
                           f.arrival_airport, f.scheduled_departure,
                           f.scheduled_arrival, f.status, f.aircraft_code,
                           f.actual_departure, f.actual_arrival,
                           p.base_amount_minor, p.currency
                    FROM flights f
                    JOIN products p
                      ON p.product_type = 'flight'
                     AND p.external_product_id = %s
                     AND p.active = TRUE
                    WHERE f.flight_id = %s
                    """,
                    (f"legacy-flight-{payload['flight_id']}", payload["flight_id"]),
                )
                flight = cursor.fetchone()
        if flight is None:
            continue

        scheduled_departure = normalize_datetime(flight[4])
        if start_time and scheduled_departure < normalize_datetime(start_time):
            continue
        if end_time and scheduled_departure > normalize_datetime(end_time):
            continue

        flights.append({
            "flight_id": flight[0],
            "flight_no": flight[1],
            "departure_airport": flight[2],
            "arrival_airport": flight[3],
            "scheduled_departure": flight[4],
            "scheduled_arrival": flight[5],
            "status": flight[6],
            "aircraft_code": flight[7],
            "actual_departure": flight[8],
            "actual_arrival": flight[9],
            "amount_minor": flight[10],
            "currency": flight[11],
            "chunk": payload["content"],
            "similarity": result.score,
        })
        if len(flights) >= limit:
            break
    return flights


@tool
async def book_flight(flight_id: int, *, config: RunnableConfig) -> str:
    """根据航班 ID 创建当前登录用户的航班订单。"""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id")
    if not passenger_id:
        raise ValueError("未配置乘客 ID。")
    return create_flight_order(flight_id, passenger_id)

@tool
async def update_ticket_to_new_flight(
    order_id: int, new_flight_id: int, *, config: RunnableConfig
) -> str:
    """根据正式订单 ID 将当前用户的航班订单改签到新航班。"""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("未配置乘客 ID。")

    return update_flight_order(order_id, passenger_id, new_flight_id)

@tool
async def cancel_ticket(order_id: int, *, config: RunnableConfig) -> str:
    """根据正式订单 ID 取消当前用户的航班订单。"""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("未配置乘客 ID。")

    return cancel_order(order_id, passenger_id)
