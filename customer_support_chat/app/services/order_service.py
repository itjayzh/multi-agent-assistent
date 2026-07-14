import uuid
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

from psycopg.types.json import Jsonb

from customer_support_chat.app.core.database import get_connection
from customer_support_chat.app.services.supplier_gateway import get_supplier_gateway


def get_user_by_passenger_id(cursor, passenger_id: str) -> Dict[str, Any]:
    cursor.execute(
        "SELECT id, username, passenger_id FROM users WHERE passenger_id = %s",
        (passenger_id,),
    )
    user = cursor.fetchone()
    if user is None:
        raise ValueError("当前乘机人 ID 尚未绑定登录用户。")
    return {"id": user[0], "username": user[1], "passenger_id": user[2]}


def get_product(cursor, product_type: str, legacy_id: int) -> Dict[str, Any]:
    cursor.execute(
        """
        SELECT p.id, p.supplier_id, p.name, p.base_amount_minor, p.currency,
               p.external_product_id, s.code
        FROM products p
        JOIN suppliers s ON s.id = p.supplier_id
        WHERE p.product_type = %s
          AND p.external_product_id = %s
          AND p.active = TRUE
        """,
        (product_type, f"legacy-{product_type}-{legacy_id}"),
    )
    product = cursor.fetchone()
    if product is None:
        raise ValueError(f"未找到可下单的 {product_type} 产品 {legacy_id}。")
    if product[3] <= 0:
        raise ValueError(f"{product_type} 产品 {legacy_id} 尚未配置价格。")
    return {
        "id": product[0],
        "supplier_id": product[1],
        "name": product[2],
        "unit_amount_minor": product[3],
        "currency": product[4],
        "external_product_id": product[5],
        "supplier_code": product[6],
    }


def normalize_date(value: Optional[Union[datetime, date]], fallback: date) -> date:
    if value is None:
        return fallback
    if isinstance(value, datetime):
        return value.date()
    return value


def resolve_service_dates(
    start_value: Optional[Union[datetime, date]],
    end_value: Optional[Union[datetime, date]],
    stored_start: Optional[Union[datetime, date]],
    stored_end: Optional[Union[datetime, date]],
) -> tuple[date, date]:
    tomorrow = date.today() + timedelta(days=1)
    stored_start_date = normalize_date(stored_start, tomorrow)
    default_start = stored_start_date if stored_start_date >= date.today() else tomorrow
    start_date = normalize_date(start_value, default_start)
    stored_end_date = normalize_date(stored_end, start_date + timedelta(days=1))
    default_end = stored_end_date if stored_end_date > start_date else start_date + timedelta(days=1)
    return start_date, normalize_date(end_value, default_end)


def to_utc_datetime(value: date) -> datetime:
    return datetime.combine(value, time.min, tzinfo=timezone.utc)


def build_order_identifiers(order_type: str, user_id: int, key_parts: List[Any]) -> Dict[str, str]:
    stable_key = ":".join(str(part) for part in key_parts)
    return {
        "order_no": f"ORD-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:10].upper()}",
        "idempotency_key": f"{order_type}:{user_id}:{stable_key}",
    }


def create_common_order(
    cursor,
    *,
    user_id: int,
    product: Dict[str, Any],
    order_type: str,
    total_amount_minor: int,
    service_start_at: datetime,
    service_end_at: datetime,
    key_parts: List[Any],
) -> Dict[str, Any]:
    identifiers = build_order_identifiers(order_type, user_id, key_parts)
    cursor.execute(
        """
        SELECT id, order_no, supplier_confirmation_no, status
        FROM orders
        WHERE user_id = %s AND idempotency_key = %s
        """,
        (user_id, identifiers["idempotency_key"]),
    )
    existing = cursor.fetchone()
    if existing is not None:
        return {
            "id": existing[0],
            "order_no": existing[1],
            "confirmation_no": existing[2],
            "created": False,
            "status": existing[3],
        }

    cursor.execute(
        """
        INSERT INTO orders (
            order_no, user_id, supplier_id, order_type,
            total_amount_minor, currency, status,
            idempotency_key, service_start_at, service_end_at
        ) VALUES (%s, %s, %s, %s, %s, %s, 'processing', %s, %s, %s)
        RETURNING id
        """,
        (
            identifiers["order_no"],
            user_id,
            product["supplier_id"],
            order_type,
            total_amount_minor,
            product["currency"],
            identifiers["idempotency_key"],
            service_start_at,
            service_end_at,
        ),
    )
    order_id = cursor.fetchone()[0]
    cursor.execute(
        """
        INSERT INTO order_status_history (order_id, from_status, to_status, reason, operator_type)
        VALUES (%s, 'pending', 'processing', '用户审批通过，等待供应商确认', 'user')
        """,
        (order_id,),
    )
    return {
        "id": order_id,
        "order_no": identifiers["order_no"],
        "confirmation_no": None,
        "created": True,
        "status": "processing",
    }


class SupplierBookingError(RuntimeError):
    pass


def finalize_supplier_booking(
    order: Dict[str, Any],
    product: Dict[str, Any],
    request_payload: Dict[str, Any],
) -> None:
    """在本地订单提交后调用供应商，并记录成功或失败终态。"""
    if not order["created"]:
        return
    payload = {
        "order_no": order["order_no"],
        "order_type": request_payload["order_type"],
        "product_id": product["id"],
        "external_product_id": product["external_product_id"],
        **request_payload,
    }
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO supplier_booking_attempts (
                    order_id, operation, request_payload
                ) VALUES (%s, 'book', %s)
                RETURNING id
                """,
                (order["id"], Jsonb(payload)),
            )
            attempt_id = cursor.fetchone()[0]

    try:
        gateway = get_supplier_gateway(product["supplier_code"])
        result = gateway.book(payload)
    except Exception as error:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE supplier_booking_attempts
                    SET status = 'failed', error_message = %s,
                        completed_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """,
                    (str(error), attempt_id),
                )
                cursor.execute(
                    "UPDATE orders SET status = 'failed' WHERE id = %s",
                    (order["id"],),
                )
                add_status_history(
                    cursor,
                    order["id"],
                    "processing",
                    "failed",
                    f"供应商下单失败：{error}",
                )
        order["status"] = "failed"
        raise SupplierBookingError(str(error)) from error

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE supplier_booking_attempts
                SET status = 'succeeded', response_payload = %s,
                    completed_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (Jsonb(result.payload), attempt_id),
            )
            cursor.execute(
                """
                UPDATE orders
                SET status = 'confirmed', supplier_confirmation_no = %s,
                    confirmed_at = CURRENT_TIMESTAMP
                WHERE id = %s AND status = 'processing'
                """,
                (result.confirmation_no, order["id"]),
            )
            add_status_history(
                cursor,
                order["id"],
                "processing",
                "confirmed",
                "供应商确认下单成功",
            )
    order["status"] = "confirmed"
    order["confirmation_no"] = result.confirmation_no


def execute_supplier_operation(
    order_id: int,
    operation: str,
    request_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """执行供应商改期或取消；失败时保留本地订单原状态。"""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT o.order_no, o.order_type, o.supplier_confirmation_no, s.code
                FROM orders o
                JOIN suppliers s ON s.id = o.supplier_id
                WHERE o.id = %s
                """,
                (order_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"订单 {order_id} 不存在。")
            payload = {
                "order_id": order_id,
                "order_no": row[0],
                "order_type": row[1],
                "supplier_confirmation_no": row[2],
                **request_payload,
            }
            cursor.execute(
                """
                INSERT INTO supplier_booking_attempts (
                    order_id, operation, request_payload
                ) VALUES (%s, %s, %s)
                RETURNING id
                """,
                (order_id, operation, Jsonb(payload)),
            )
            attempt_id = cursor.fetchone()[0]
            supplier_code = row[3]

    try:
        gateway = get_supplier_gateway(supplier_code)
        response = getattr(gateway, operation)(payload)
    except Exception as error:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE supplier_booking_attempts
                    SET status = 'failed', error_message = %s,
                        completed_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """,
                    (str(error), attempt_id),
                )
        raise SupplierBookingError(str(error)) from error

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE supplier_booking_attempts
                SET status = 'succeeded', response_payload = %s,
                    completed_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (Jsonb(response), attempt_id),
            )
    return response


def retry_supplier_order(order_id: int, passenger_id: str) -> str:
    """重试失败的供应商下单，保留原订单号和幂等键。"""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            user = get_user_by_passenger_id(cursor, passenger_id)
            order = get_order_for_update(cursor, order_id, user["id"])
            if order["status"] != "failed":
                raise ValueError(f"订单 {order['order_no']} 不是失败状态，无需重试。")
            cursor.execute(
                """
                SELECT request_payload
                FROM supplier_booking_attempts
                WHERE order_id = %s AND operation = 'book'
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (order_id,),
            )
            attempt = cursor.fetchone()
            if attempt is None:
                raise ValueError("未找到可重试的供应商下单记录。")
            request_payload = attempt[0]
            cursor.execute(
                """
                SELECT p.id, p.external_product_id, s.code
                FROM products p
                JOIN suppliers s ON s.id = p.supplier_id
                WHERE p.id = %s
                """,
                (request_payload["product_id"],),
            )
            product_row = cursor.fetchone()
            if product_row is None:
                raise ValueError("重试所需产品已不存在。")
            cursor.execute(
                "UPDATE orders SET status = 'processing' WHERE id = %s",
                (order_id,),
            )
            add_status_history(cursor, order_id, "failed", "processing", "用户审批通过后重试供应商下单")

    retry_order = {
        "id": order_id,
        "order_no": order["order_no"],
        "created": True,
        "status": "processing",
        "confirmation_no": None,
    }
    product = {
        "id": product_row[0],
        "external_product_id": product_row[1],
        "supplier_code": product_row[2],
    }
    retry_payload = dict(request_payload)
    retry_payload.pop("order_no", None)
    retry_payload.pop("product_id", None)
    retry_payload.pop("external_product_id", None)
    finalize_supplier_booking(retry_order, product, retry_payload)
    return (
        f"订单 {order['order_no']} 重试成功，"
        f"供应商确认号 {retry_order['confirmation_no']}。"
    )


def create_flight_order(flight_id: int, passenger_id: str) -> str:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            user = get_user_by_passenger_id(cursor, passenger_id)
            product = get_product(cursor, "flight", flight_id)
            cursor.execute(
                """
                SELECT flight_no, departure_airport, arrival_airport,
                       scheduled_departure, scheduled_arrival
                FROM flights WHERE flight_id = %s
                """,
                (flight_id,),
            )
            flight = cursor.fetchone()
            if flight is None:
                raise ValueError(f"未找到航班 {flight_id}。")

            order = create_common_order(
                cursor,
                user_id=user["id"],
                product=product,
                order_type="flight",
                total_amount_minor=product["unit_amount_minor"],
                service_start_at=flight[3],
                service_end_at=flight[4],
                key_parts=[flight_id],
            )
            if order["created"]:
                ticket_no = f"TKT-{order['order_no']}"
                cursor.execute(
                    """
                    INSERT INTO flight_orders (
                        order_id, contact_name, booking_reference, ticket_status
                    ) VALUES (%s, %s, %s, 'issued')
                    """,
                    (order["id"], user["username"], order["confirmation_no"]),
                )
                cursor.execute(
                    """
                    INSERT INTO flight_order_segments (
                        order_id, segment_no, flight_product_id, flight_no,
                        departure_airport, arrival_airport, departure_at,
                        arrival_at, cabin_class, fare_conditions, amount_minor
                    ) VALUES (%s, 1, %s, %s, %s, %s, %s, %s, 'Economy', 'Economy', %s)
                    """,
                    (
                        order["id"], product["id"], flight[0], flight[1],
                        flight[2], flight[3], flight[4], product["unit_amount_minor"],
                    ),
                )
                cursor.execute(
                    """
                    INSERT INTO flight_order_passengers (
                        order_id, passenger_ref, passenger_name, ticket_no
                    ) VALUES (%s, %s, %s, %s)
                    """,
                    (order["id"], passenger_id, user["username"], ticket_no),
                )

    finalize_supplier_booking(order, product, {
        "order_type": "flight",
        "flight_id": flight_id,
        "passenger_id": passenger_id,
    })
    prefix = "订单已存在" if not order["created"] else "航班订单创建成功"
    return f"{prefix}：订单号 {order['order_no']}，确认号 {order['confirmation_no']}。"


def create_hotel_order(
    hotel_id: int,
    passenger_id: str,
    checkin_date: Optional[Union[datetime, date]] = None,
    checkout_date: Optional[Union[datetime, date]] = None,
) -> str:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            user = get_user_by_passenger_id(cursor, passenger_id)
            product = get_product(cursor, "hotel", hotel_id)
            cursor.execute(
                "SELECT name, checkin_date, checkout_date, booked FROM hotels WHERE id = %s FOR UPDATE",
                (hotel_id,),
            )
            hotel = cursor.fetchone()
            if hotel is None:
                raise ValueError(f"未找到酒店 {hotel_id}。")
            start_date, end_date = resolve_service_dates(
                checkin_date,
                checkout_date,
                hotel[1],
                hotel[2],
            )
            if end_date <= start_date:
                raise ValueError("酒店退房日期必须晚于入住日期。")
            nights = (end_date - start_date).days
            total_amount = product["unit_amount_minor"] * nights
            order = create_common_order(
                cursor,
                user_id=user["id"], product=product, order_type="hotel",
                total_amount_minor=total_amount,
                service_start_at=to_utc_datetime(start_date),
                service_end_at=to_utc_datetime(end_date),
                key_parts=[hotel_id, start_date, end_date],
            )
            if order["created"] and hotel[3]:
                raise ValueError("该酒店产品已被预订。")
            if order["created"]:
                cursor.execute(
                    """
                    INSERT INTO hotel_orders (
                        order_id, hotel_product_id, hotel_name, room_type,
                        checkin_date, checkout_date, guest_names,
                        unit_amount_minor, total_amount_minor
                    ) VALUES (%s, %s, %s, '标准房', %s, %s, %s, %s, %s)
                    """,
                    (
                        order["id"], product["id"], hotel[0], start_date, end_date,
                        Jsonb([user["username"]]), product["unit_amount_minor"], total_amount,
                    ),
                )
                cursor.execute(
                    "UPDATE hotels SET booked = TRUE, owner_passenger_id = %s, checkin_date = %s, checkout_date = %s WHERE id = %s",
                    (passenger_id, start_date, end_date, hotel_id),
                )

    finalize_supplier_booking(order, product, {
        "order_type": "hotel",
        "hotel_id": hotel_id,
        "checkin_date": start_date.isoformat(),
        "checkout_date": end_date.isoformat(),
    })
    prefix = "订单已存在" if not order["created"] else "酒店订单创建成功"
    return f"{prefix}：订单号 {order['order_no']}，确认号 {order['confirmation_no']}。"


def create_car_order(
    rental_id: int,
    passenger_id: str,
    start_date: Optional[Union[datetime, date]] = None,
    end_date: Optional[Union[datetime, date]] = None,
) -> str:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            user = get_user_by_passenger_id(cursor, passenger_id)
            product = get_product(cursor, "car", rental_id)
            cursor.execute(
                "SELECT name, location, start_date, end_date, booked FROM car_rentals WHERE id = %s FOR UPDATE",
                (rental_id,),
            )
            rental = cursor.fetchone()
            if rental is None:
                raise ValueError(f"未找到租车产品 {rental_id}。")
            pickup_date, return_date = resolve_service_dates(
                start_date,
                end_date,
                rental[2],
                rental[3],
            )
            if return_date <= pickup_date:
                raise ValueError("还车日期必须晚于取车日期。")
            days = (return_date - pickup_date).days
            total_amount = product["unit_amount_minor"] * days
            order = create_common_order(
                cursor,
                user_id=user["id"], product=product, order_type="car",
                total_amount_minor=total_amount,
                service_start_at=to_utc_datetime(pickup_date),
                service_end_at=to_utc_datetime(return_date),
                key_parts=[rental_id, pickup_date, return_date],
            )
            if order["created"] and rental[4]:
                raise ValueError("该租车产品已被预订。")
            if order["created"]:
                cursor.execute(
                    """
                    INSERT INTO car_orders (
                        order_id, car_product_id, product_name, pickup_location,
                        return_location, pickup_at, return_at, driver_name,
                        unit_amount_minor, total_amount_minor
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        order["id"], product["id"], rental[0], rental[1], rental[1],
                        to_utc_datetime(pickup_date), to_utc_datetime(return_date),
                        user["username"], product["unit_amount_minor"], total_amount,
                    ),
                )
                cursor.execute(
                    "UPDATE car_rentals SET booked = TRUE, owner_passenger_id = %s, start_date = %s, end_date = %s WHERE id = %s",
                    (passenger_id, pickup_date, return_date, rental_id),
                )

    finalize_supplier_booking(order, product, {
        "order_type": "car",
        "rental_id": rental_id,
        "start_date": pickup_date.isoformat(),
        "end_date": return_date.isoformat(),
    })
    prefix = "订单已存在" if not order["created"] else "租车订单创建成功"
    return f"{prefix}：订单号 {order['order_no']}，确认号 {order['confirmation_no']}。"


def create_trip_order(recommendation_id: int, passenger_id: str) -> str:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            user = get_user_by_passenger_id(cursor, passenger_id)
            product = get_product(cursor, "trip", recommendation_id)
            cursor.execute(
                "SELECT name, booked FROM trip_recommendations WHERE id = %s FOR UPDATE",
                (recommendation_id,),
            )
            trip = cursor.fetchone()
            if trip is None:
                raise ValueError(f"未找到行程产品 {recommendation_id}。")
            visit_date = date.today() + timedelta(days=1)
            order = create_common_order(
                cursor,
                user_id=user["id"], product=product, order_type="trip",
                total_amount_minor=product["unit_amount_minor"],
                service_start_at=to_utc_datetime(visit_date),
                service_end_at=to_utc_datetime(visit_date + timedelta(days=1)),
                key_parts=[recommendation_id, visit_date],
            )
            if order["created"] and trip[1]:
                raise ValueError("该行程产品已被预订。")
            if order["created"]:
                cursor.execute(
                    """
                    INSERT INTO trip_orders (
                        order_id, trip_product_id, product_name, visit_date,
                        unit_amount_minor, total_amount_minor
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        order["id"], product["id"], trip[0], visit_date,
                        product["unit_amount_minor"], product["unit_amount_minor"],
                    ),
                )
                cursor.execute(
                    "UPDATE trip_recommendations SET booked = TRUE WHERE id = %s",
                    (recommendation_id,),
                )

    finalize_supplier_booking(order, product, {
        "order_type": "trip",
        "recommendation_id": recommendation_id,
        "visit_date": visit_date.isoformat(),
    })
    prefix = "订单已存在" if not order["created"] else "行程订单创建成功"
    return f"{prefix}：订单号 {order['order_no']}，确认号 {order['confirmation_no']}。"


def serialize_value(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def list_user_orders(user_id: int) -> List[Dict[str, Any]]:
    with get_connection(rows_as_dict=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, order_no, order_type, total_amount_minor, currency,
                       status, supplier_confirmation_no, service_start_at,
                       service_end_at, created_at
                FROM orders
                WHERE user_id = %s
                ORDER BY created_at DESC
                """,
                (user_id,),
            )
            orders = cursor.fetchall()

    return [
        {key: serialize_value(value) for key, value in order.items()}
        for order in orders
    ]


def get_order_for_update(cursor, order_id: int, user_id: int) -> Dict[str, Any]:
    cursor.execute(
        """
        SELECT id, order_no, order_type, status, total_amount_minor,
               currency, service_start_at, service_end_at
        FROM orders
        WHERE id = %s AND user_id = %s
        FOR UPDATE
        """,
        (order_id, user_id),
    )
    row = cursor.fetchone()
    if row is None:
        raise ValueError(f"未找到属于当前用户的订单 {order_id}。")
    return {
        "id": row[0],
        "order_no": row[1],
        "order_type": row[2],
        "status": row[3],
        "total_amount_minor": row[4],
        "currency": row[5],
        "service_start_at": row[6],
        "service_end_at": row[7],
    }


def require_confirmed_order(order: Dict[str, Any], order_type: str) -> None:
    if order["order_type"] != order_type:
        raise ValueError(f"订单 {order['order_no']} 不是 {order_type} 类型。")
    if order["status"] == "cancelled":
        raise ValueError(f"订单 {order['order_no']} 已取消，无法继续修改。")
    if order["status"] != "confirmed":
        raise ValueError(f"订单 {order['order_no']} 当前状态不允许修改。")


def add_status_history(
    cursor,
    order_id: int,
    from_status: str,
    to_status: str,
    reason: str,
) -> None:
    cursor.execute(
        """
        INSERT INTO order_status_history (
            order_id, from_status, to_status, reason, operator_type
        ) VALUES (%s, %s, %s, %s, 'user')
        """,
        (order_id, from_status, to_status, reason),
    )


def get_legacy_product_id(cursor, product_id: int) -> Optional[int]:
    cursor.execute("SELECT external_product_id FROM products WHERE id = %s", (product_id,))
    row = cursor.fetchone()
    if row is None or not row[0]:
        return None
    suffix = row[0].rsplit("-", maxsplit=1)[-1]
    return int(suffix) if suffix.isdigit() else None


def cancel_order(order_id: int, passenger_id: str) -> str:
    """取消正式订单并同步业务明细和兼容产品状态。"""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            user = get_user_by_passenger_id(cursor, passenger_id)
            current_order = get_order_for_update(cursor, order_id, user["id"])
            if current_order["status"] == "cancelled":
                return f"订单 {current_order['order_no']} 已经取消。"
            if current_order["status"] not in {"pending", "processing", "confirmed"}:
                raise ValueError(f"订单 {current_order['order_no']} 当前状态不允许取消。")
    execute_supplier_operation(order_id, "cancel", {})

    with get_connection() as conn:
        with conn.cursor() as cursor:
            user = get_user_by_passenger_id(cursor, passenger_id)
            order = get_order_for_update(cursor, order_id, user["id"])
            if order["status"] == "cancelled":
                return f"订单 {order['order_no']} 已经取消。"
            if order["status"] not in {"pending", "processing", "confirmed"}:
                raise ValueError(f"订单 {order['order_no']} 当前状态不允许取消。")

            if order["order_type"] == "flight":
                cursor.execute(
                    "UPDATE flight_orders SET ticket_status = 'cancelled' WHERE order_id = %s",
                    (order_id,),
                )
            elif order["order_type"] == "hotel":
                cursor.execute(
                    "SELECT hotel_product_id FROM hotel_orders WHERE order_id = %s",
                    (order_id,),
                )
                detail = cursor.fetchone()
                legacy_id = get_legacy_product_id(cursor, detail[0]) if detail else None
                if legacy_id is not None:
                    cursor.execute(
                        """
                        UPDATE hotels
                        SET booked = FALSE, owner_passenger_id = NULL
                        WHERE id = %s AND owner_passenger_id = %s
                        """,
                        (legacy_id, passenger_id),
                    )
            elif order["order_type"] == "car":
                cursor.execute(
                    "SELECT car_product_id FROM car_orders WHERE order_id = %s",
                    (order_id,),
                )
                detail = cursor.fetchone()
                legacy_id = get_legacy_product_id(cursor, detail[0]) if detail else None
                if legacy_id is not None:
                    cursor.execute(
                        """
                        UPDATE car_rentals
                        SET booked = FALSE, owner_passenger_id = NULL
                        WHERE id = %s AND owner_passenger_id = %s
                        """,
                        (legacy_id, passenger_id),
                    )
            elif order["order_type"] == "trip":
                cursor.execute(
                    "SELECT trip_product_id FROM trip_orders WHERE order_id = %s",
                    (order_id,),
                )
                detail = cursor.fetchone()
                legacy_id = get_legacy_product_id(cursor, detail[0]) if detail else None
                if legacy_id is not None:
                    cursor.execute(
                        "UPDATE trip_recommendations SET booked = FALSE WHERE id = %s",
                        (legacy_id,),
                    )

            cursor.execute(
                """
                UPDATE orders
                SET status = 'cancelled', cancelled_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (order_id,),
            )
            add_status_history(
                cursor,
                order_id,
                order["status"],
                "cancelled",
                "用户审批通过后取消订单",
            )
    return f"订单 {order['order_no']} 已成功取消。"


def update_hotel_order(
    order_id: int,
    passenger_id: str,
    checkin_date: Union[datetime, date],
    checkout_date: Union[datetime, date],
) -> str:
    start_date = normalize_date(checkin_date, date.today())
    end_date = normalize_date(checkout_date, start_date + timedelta(days=1))
    if end_date <= start_date:
        raise ValueError("酒店退房日期必须晚于入住日期。")

    with get_connection() as conn:
        with conn.cursor() as cursor:
            user = get_user_by_passenger_id(cursor, passenger_id)
            current_order = get_order_for_update(cursor, order_id, user["id"])
            require_confirmed_order(current_order, "hotel")
    execute_supplier_operation(order_id, "update", {
        "checkin_date": start_date.isoformat(),
        "checkout_date": end_date.isoformat(),
    })

    with get_connection() as conn:
        with conn.cursor() as cursor:
            user = get_user_by_passenger_id(cursor, passenger_id)
            order = get_order_for_update(cursor, order_id, user["id"])
            require_confirmed_order(order, "hotel")
            cursor.execute(
                """
                SELECT hotel_product_id, unit_amount_minor
                FROM hotel_orders WHERE order_id = %s
                """,
                (order_id,),
            )
            detail = cursor.fetchone()
            if detail is None:
                raise ValueError("酒店订单明细不存在。")
            total_amount = detail[1] * (end_date - start_date).days
            cursor.execute(
                """
                UPDATE hotel_orders
                SET checkin_date = %s, checkout_date = %s,
                    total_amount_minor = %s
                WHERE order_id = %s
                """,
                (start_date, end_date, total_amount, order_id),
            )
            cursor.execute(
                """
                UPDATE orders
                SET service_start_at = %s, service_end_at = %s,
                    total_amount_minor = %s
                WHERE id = %s
                """,
                (to_utc_datetime(start_date), to_utc_datetime(end_date), total_amount, order_id),
            )
            legacy_id = get_legacy_product_id(cursor, detail[0])
            if legacy_id is not None:
                cursor.execute(
                    """
                    UPDATE hotels SET checkin_date = %s, checkout_date = %s
                    WHERE id = %s AND owner_passenger_id = %s
                    """,
                    (start_date, end_date, legacy_id, passenger_id),
                )
            add_status_history(cursor, order_id, "confirmed", "confirmed", "用户修改酒店入住日期")
    return f"酒店订单 {order['order_no']} 已修改成功。"


def update_car_order(
    order_id: int,
    passenger_id: str,
    start_date: Union[datetime, date],
    end_date: Union[datetime, date],
) -> str:
    pickup_date = normalize_date(start_date, date.today())
    return_date = normalize_date(end_date, pickup_date + timedelta(days=1))
    if return_date <= pickup_date:
        raise ValueError("还车日期必须晚于取车日期。")

    with get_connection() as conn:
        with conn.cursor() as cursor:
            user = get_user_by_passenger_id(cursor, passenger_id)
            current_order = get_order_for_update(cursor, order_id, user["id"])
            require_confirmed_order(current_order, "car")
    execute_supplier_operation(order_id, "update", {
        "start_date": pickup_date.isoformat(),
        "end_date": return_date.isoformat(),
    })

    with get_connection() as conn:
        with conn.cursor() as cursor:
            user = get_user_by_passenger_id(cursor, passenger_id)
            order = get_order_for_update(cursor, order_id, user["id"])
            require_confirmed_order(order, "car")
            cursor.execute(
                """
                SELECT car_product_id, unit_amount_minor
                FROM car_orders WHERE order_id = %s
                """,
                (order_id,),
            )
            detail = cursor.fetchone()
            if detail is None:
                raise ValueError("租车订单明细不存在。")
            pickup_at = to_utc_datetime(pickup_date)
            return_at = to_utc_datetime(return_date)
            total_amount = detail[1] * (return_date - pickup_date).days
            cursor.execute(
                """
                UPDATE car_orders
                SET pickup_at = %s, return_at = %s, total_amount_minor = %s
                WHERE order_id = %s
                """,
                (pickup_at, return_at, total_amount, order_id),
            )
            cursor.execute(
                """
                UPDATE orders
                SET service_start_at = %s, service_end_at = %s,
                    total_amount_minor = %s
                WHERE id = %s
                """,
                (pickup_at, return_at, total_amount, order_id),
            )
            legacy_id = get_legacy_product_id(cursor, detail[0])
            if legacy_id is not None:
                cursor.execute(
                    """
                    UPDATE car_rentals SET start_date = %s, end_date = %s
                    WHERE id = %s AND owner_passenger_id = %s
                    """,
                    (pickup_date, return_date, legacy_id, passenger_id),
                )
            add_status_history(cursor, order_id, "confirmed", "confirmed", "用户修改租车日期")
    return f"租车订单 {order['order_no']} 已修改成功。"


def update_flight_order(order_id: int, passenger_id: str, new_flight_id: int) -> str:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            user = get_user_by_passenger_id(cursor, passenger_id)
            current_order = get_order_for_update(cursor, order_id, user["id"])
            require_confirmed_order(current_order, "flight")
    execute_supplier_operation(order_id, "update", {"new_flight_id": new_flight_id})

    with get_connection() as conn:
        with conn.cursor() as cursor:
            user = get_user_by_passenger_id(cursor, passenger_id)
            order = get_order_for_update(cursor, order_id, user["id"])
            require_confirmed_order(order, "flight")
            product = get_product(cursor, "flight", new_flight_id)
            cursor.execute(
                """
                SELECT flight_no, departure_airport, arrival_airport,
                       scheduled_departure, scheduled_arrival
                FROM flights WHERE flight_id = %s
                """,
                (new_flight_id,),
            )
            flight = cursor.fetchone()
            if flight is None:
                raise ValueError(f"未找到航班 {new_flight_id}。")
            cursor.execute(
                """
                UPDATE flight_order_segments
                SET flight_product_id = %s, flight_no = %s,
                    departure_airport = %s, arrival_airport = %s,
                    departure_at = %s, arrival_at = %s, amount_minor = %s
                WHERE order_id = %s AND segment_no = 1
                """,
                (
                    product["id"], flight[0], flight[1], flight[2], flight[3],
                    flight[4], product["unit_amount_minor"], order_id,
                ),
            )
            cursor.execute(
                """
                UPDATE orders
                SET supplier_id = %s, total_amount_minor = %s, currency = %s,
                    service_start_at = %s, service_end_at = %s
                WHERE id = %s
                """,
                (
                    product["supplier_id"], product["unit_amount_minor"],
                    product["currency"], flight[3], flight[4], order_id,
                ),
            )
            add_status_history(cursor, order_id, "confirmed", "confirmed", "用户修改航班")
    return f"航班订单 {order['order_no']} 已改签至 {flight[0]}。"


def update_trip_order(
    order_id: int,
    passenger_id: str,
    visit_date: Union[datetime, date],
    participant_count: int = 1,
) -> str:
    normalized_visit_date = normalize_date(visit_date, date.today() + timedelta(days=1))
    if participant_count <= 0:
        raise ValueError("参与人数必须大于 0。")

    with get_connection() as conn:
        with conn.cursor() as cursor:
            user = get_user_by_passenger_id(cursor, passenger_id)
            current_order = get_order_for_update(cursor, order_id, user["id"])
            require_confirmed_order(current_order, "trip")
    execute_supplier_operation(order_id, "update", {
        "visit_date": normalized_visit_date.isoformat(),
        "participant_count": participant_count,
    })

    with get_connection() as conn:
        with conn.cursor() as cursor:
            user = get_user_by_passenger_id(cursor, passenger_id)
            order = get_order_for_update(cursor, order_id, user["id"])
            require_confirmed_order(order, "trip")
            cursor.execute(
                """
                SELECT unit_amount_minor FROM trip_orders WHERE order_id = %s
                """,
                (order_id,),
            )
            detail = cursor.fetchone()
            if detail is None:
                raise ValueError("行程订单明细不存在。")
            total_amount = detail[0] * participant_count
            cursor.execute(
                """
                UPDATE trip_orders
                SET visit_date = %s, participant_count = %s,
                    total_amount_minor = %s
                WHERE order_id = %s
                """,
                (normalized_visit_date, participant_count, total_amount, order_id),
            )
            cursor.execute(
                """
                UPDATE orders
                SET service_start_at = %s, service_end_at = %s,
                    total_amount_minor = %s
                WHERE id = %s
                """,
                (
                    to_utc_datetime(normalized_visit_date),
                    to_utc_datetime(normalized_visit_date + timedelta(days=1)),
                    total_amount,
                    order_id,
                ),
            )
            add_status_history(cursor, order_id, "confirmed", "confirmed", "用户修改行程日期或人数")
    return f"行程订单 {order['order_no']} 已修改成功。"


def query_user_orders(
    user_id: int,
    *,
    status: Optional[str] = None,
    order_type: Optional[str] = None,
    page: int = 1,
    page_size: int = 10,
) -> Dict[str, Any]:
    page = max(page, 1)
    page_size = min(max(page_size, 1), 50)
    conditions = ["user_id = %s"]
    params: List[Any] = [user_id]
    if status:
        conditions.append("status = %s")
        params.append(status)
    if order_type:
        conditions.append("order_type = %s")
        params.append(order_type)
    where_clause = " AND ".join(conditions)

    with get_connection(rows_as_dict=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) AS total FROM orders WHERE {where_clause}", params)
            total = cursor.fetchone()["total"]
            cursor.execute(
                f"""
                SELECT id, order_no, order_type, total_amount_minor, currency,
                       status, supplier_confirmation_no, service_start_at,
                       service_end_at, created_at
                FROM orders
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                [*params, page_size, (page - 1) * page_size],
            )
            rows = cursor.fetchall()
    return {
        "items": [
            {key: serialize_value(value) for key, value in row.items()}
            for row in rows
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


def get_order_detail(user_id: int, order_id: int) -> Dict[str, Any]:
    with get_connection(rows_as_dict=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT o.id, o.order_no, o.order_type, o.total_amount_minor,
                       o.currency, o.status, o.supplier_confirmation_no,
                       o.service_start_at, o.service_end_at, o.created_at,
                       o.updated_at, o.confirmed_at, o.cancelled_at,
                       s.name AS supplier_name
                FROM orders o
                JOIN suppliers s ON s.id = o.supplier_id
                WHERE o.id = %s AND o.user_id = %s
                """,
                (order_id, user_id),
            )
            order = cursor.fetchone()
            if order is None:
                raise ValueError("订单不存在或不属于当前用户。")

            detail_queries = {
                "flight": "SELECT * FROM flight_orders WHERE order_id = %s",
                "hotel": "SELECT * FROM hotel_orders WHERE order_id = %s",
                "car": "SELECT * FROM car_orders WHERE order_id = %s",
                "trip": "SELECT * FROM trip_orders WHERE order_id = %s",
            }
            cursor.execute(detail_queries[order["order_type"]], (order_id,))
            detail = cursor.fetchone()
            segments = []
            passengers = []
            if order["order_type"] == "flight":
                cursor.execute(
                    "SELECT * FROM flight_order_segments WHERE order_id = %s ORDER BY segment_no",
                    (order_id,),
                )
                segments = cursor.fetchall()
                cursor.execute(
                    "SELECT * FROM flight_order_passengers WHERE order_id = %s ORDER BY id",
                    (order_id,),
                )
                passengers = cursor.fetchall()
            cursor.execute(
                """
                SELECT from_status, to_status, reason, operator_type, created_at
                FROM order_status_history
                WHERE order_id = %s ORDER BY created_at, id
                """,
                (order_id,),
            )
            history = cursor.fetchall()
            cursor.execute(
                """
                SELECT operation, status, error_message, created_at, completed_at
                FROM supplier_booking_attempts
                WHERE order_id = %s ORDER BY created_at, id
                """,
                (order_id,),
            )
            supplier_attempts = cursor.fetchall()

    serialize_row = lambda row: {
        key: serialize_value(value) for key, value in row.items()
    } if row else None
    result = serialize_row(order)
    result["detail"] = serialize_row(detail)
    result["segments"] = [serialize_row(row) for row in segments]
    result["passengers"] = [serialize_row(row) for row in passengers]
    result["status_history"] = [serialize_row(row) for row in history]
    result["supplier_attempts"] = [serialize_row(row) for row in supplier_attempts]
    return result
