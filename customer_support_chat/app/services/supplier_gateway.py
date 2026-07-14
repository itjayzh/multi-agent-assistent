from dataclasses import dataclass
from typing import Any, Dict, Protocol

from customer_support_chat.app.core.database import get_connection


@dataclass
class SupplierResult:
    confirmation_no: str
    payload: Dict[str, Any]


class SupplierGateway(Protocol):
    def book(self, request: Dict[str, Any]) -> SupplierResult:
        ...

    def update(self, request: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def cancel(self, request: Dict[str, Any]) -> Dict[str, Any]:
        ...


class DatabaseSupplierGateway:
    """使用 PostgreSQL 产品数据完成本地供应商操作。"""

    def book(self, request: Dict[str, Any]) -> SupplierResult:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT active, base_amount_minor
                    FROM products
                    WHERE id = %s
                    """,
                    (request["product_id"],),
                )
                product = cursor.fetchone()
        if product is None or not product[0]:
            raise ValueError("供应商产品不存在或已停用。")
        if product[1] <= 0:
            raise ValueError("供应商产品价格尚未配置。")

        order_type = str(request["order_type"]).upper()
        order_suffix = str(request["order_no"]).rsplit("-", maxsplit=1)[-1]
        confirmation_no = f"DB-{order_type}-{order_suffix}"
        return SupplierResult(
            confirmation_no=confirmation_no,
            payload={
                "status": "confirmed",
                "provider": "postgresql",
                "confirmation_no": confirmation_no,
            },
        )

    def update(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "updated", "provider": "postgresql"}

    def cancel(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "cancelled", "provider": "postgresql"}


gateways: Dict[str, SupplierGateway] = {}
database_gateway = DatabaseSupplierGateway()


def register_supplier_gateway(supplier_code: str, gateway: SupplierGateway) -> None:
    gateways[supplier_code] = gateway


def get_supplier_gateway(supplier_code: str) -> SupplierGateway:
    if supplier_code.startswith("DB_"):
        return gateways.get(supplier_code, database_gateway)
    gateway = gateways.get(supplier_code)
    if gateway is None:
        raise RuntimeError(f"供应商 {supplier_code} 尚未配置真实接口适配器。")
    return gateway
