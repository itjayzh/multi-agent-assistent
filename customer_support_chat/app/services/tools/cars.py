from vectorizer.app.vectordb.vectordb import VectorDB
from customer_support_chat.app.core.settings import get_settings
from langchain_core.tools import tool
from customer_support_chat.app.core.humanloop_manager import humanloop_adapter # 导入审批适配器
import sqlite3
from typing import List, Dict, Optional, Union
from datetime import datetime, date

settings = get_settings()
db = settings.SQLITE_DB_PATH

cars_vectordb = VectorDB(table_name="car_rentals", collection_name="car_rentals_collection")

@tool
def search_car_rentals(
    query: str,
    limit: int = 2,
) -> List[Dict]:
    """根据自然语言查询搜索租车信息。"""
    search_results = cars_vectordb.search(query, limit=limit)

    rentals = []
    for result in search_results:
        payload = result.payload
        rentals.append({
            "id": payload["id"],
            "name": payload["name"],
            "location": payload["location"],
            "price_tier": payload["price_tier"],
            "start_date": payload["start_date"],
            "end_date": payload["end_date"],
            "booked": payload["booked"],
            "chunk": payload["content"],
            "similarity": result.score,
        })
    return rentals

@tool
@humanloop_adapter.require_approval(execute_on_reject=False)
async def book_car_rental(rental_id: int, approval_result=None) -> str:
    """根据租车 ID 进行预订。"""
    # 如果审批被拒绝，将不会执行下面的函数体。
    # 如果审批通过，approval_result 中会包含审批详情。
    
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE car_rentals SET booked = 1 WHERE id = ?", (rental_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"租车记录 {rental_id} 预订成功。"
    else:
        conn.close()
        return f"未找到 ID 为 {rental_id} 的租车记录。"

@tool
@humanloop_adapter.require_approval(execute_on_reject=False)
async def update_car_rental(
    rental_id: int,
    start_date: Optional[Union[datetime, date]] = None,
    end_date: Optional[Union[datetime, date]] = None,
    approval_result=None
) -> str:
    """根据租车 ID 更新租期开始和结束日期。"""
    # 如果审批被拒绝，将不会执行下面的函数体。
    # 如果审批通过，approval_result 中会包含审批详情。
    
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    if start_date:
        cursor.execute(
            "UPDATE car_rentals SET start_date = ? WHERE id = ?",
            (start_date.strftime('%Y-%m-%d'), rental_id),
        )
    if end_date:
        cursor.execute(
            "UPDATE car_rentals SET end_date = ? WHERE id = ?",
            (end_date.strftime('%Y-%m-%d'), rental_id),
        )

    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"租车记录 {rental_id} 更新成功。"
    else:
        conn.close()
        return f"未找到 ID 为 {rental_id} 的租车记录。"

@tool
@humanloop_adapter.require_approval(execute_on_reject=False)
async def cancel_car_rental(rental_id: int, approval_result=None) -> str:
    """根据租车 ID 取消租车。"""
    # 如果审批被拒绝，将不会执行下面的函数体。
    # 如果审批通过，approval_result 中会包含审批详情。
    
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE car_rentals SET booked = 0 WHERE id = ?", (rental_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"租车记录 {rental_id} 已成功取消。"
    else:
        conn.close()
        return f"未找到 ID 为 {rental_id} 的租车记录。"
