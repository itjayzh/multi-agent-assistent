from customer_support_chat.app.core.database import get_connection
from customer_support_chat.app.core.logger import logger
from qdrant_client import QdrantClient
from customer_support_chat.app.core.settings import get_settings
from typing import List, Dict, Callable

from langchain_core.messages import ToolMessage
from customer_support_chat.app.core.state import State

settings = get_settings()


def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        # 处理消息中的全部工具调用，而不只是第一个
        last_message = state["messages"][-1]
        tool_messages = []
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                tool_messages.append(
                    ToolMessage(
                        content=(
                            f"当前由 {assistant_name} 接手处理。请结合上方主助手与用户之间的对话继续完成任务。"
                            f"用户的诉求尚未满足，请使用提供的工具继续协助用户。请记住，你当前扮演的是 {assistant_name}，"
                            "预订、修改或其他操作只有在成功调用对应工具后才算真正完成。"
                            "如果用户改变主意，或需要处理其他类型的任务，请调用 CompleteOrEscalate，将控制权交还给主助手。"
                            "不要特别说明你的身份，只需继续以该助手的身份代理处理即可。"
                        ),
                        tool_call_id=tool_call["id"],
                    )
                )
        else:
            # 未发现工具调用时的兜底处理（正常委派流程中一般不会发生）
            tool_messages.append(
                ToolMessage(
                    content=(
                        f"当前由 {assistant_name} 接手处理。请结合上方主助手与用户之间的对话继续完成任务。"
                        f"用户的诉求尚未满足，请使用提供的工具继续协助用户。请记住，你当前扮演的是 {assistant_name}，"
                        "预订、修改或其他操作只有在成功调用对应工具后才算真正完成。"
                        "如果用户改变主意，或需要处理其他类型的任务，请调用 CompleteOrEscalate，将控制权交还给主助手。"
                        "不要特别说明你的身份，只需继续以该助手的身份代理处理即可。"
                    ),
                    tool_call_id="fallback_tool_call_id",
                )
            )
        
        return {
            "messages": tool_messages,
            "dialog_state": new_dialog_state,
        }
    return entry_node


def download_and_prepare_db():
    """验证 PostgreSQL 数据库连接可用。"""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            {
                "type": "tool",
                "content": f"错误信息：{repr(error)}\n请修正后重新尝试。",
                "tool_call_id": tc["id"],
            }
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list):
    from langchain_core.messages import ToolMessage
    from langchain_core.runnables import RunnableLambda
    from langgraph.prebuilt import ToolNode

    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def get_qdrant_client():
    settings = get_settings()
    try:
        # 如果提供了 API Key（云端 Qdrant），则使用认证；否则按本地方式直接连接
        if settings.QDRANT_KEY:
            client = QdrantClient(
                url=settings.QDRANT_URL, 
                api_key=settings.QDRANT_KEY,
                timeout=60  # 将超时时间提高到 60 秒
            )
        else:
            client = QdrantClient(url=settings.QDRANT_URL, timeout=60)
        # 测试连接
        collections = client.get_collections()
        logger.info(f"Qdrant 连接成功，当前发现 {len(collections.collections)} 个已存在集合。")
        return client
    except Exception as e:
        logger.error(f"连接 Qdrant 服务器失败，地址：{settings.QDRANT_URL}。错误信息：{str(e)}")
        raise

def flight_info_to_string(flight_info: List[Dict]) -> str:
    info_lines = [] 
    i = 0
    for flight in flight_info:
        i += 1
        line = (
            f"机票 [{i}]：\n"
            f"票号：{flight['ticket_no']}\n"
            f"预订编号：{flight['book_ref']}\n"
            f"航班 ID：{flight['flight_id']}\n"
            f"航班号：{flight['flight_no']}\n"
            f"出发：{flight['departure_airport']}，时间：{flight['scheduled_departure']}\n"
            f"到达：{flight['arrival_airport']}，时间：{flight['scheduled_arrival']}\n"
            f"座位号：{flight['seat_no']}\n"
            f"舱位等级：{flight['fare_conditions']}\n"
            f"\n\n"
        )
        info_lines.append(line)

    info_lines = f"用户当前已预订的航班信息如下：\n" + "\n".join(info_lines)

    return "\n".join(info_lines)
