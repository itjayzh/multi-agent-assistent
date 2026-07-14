from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from customer_support_chat.app.services.tools import (
    search_flights,
    lookup_policy,
    fetch_order_detail,
    fetch_user_orders,
)
from customer_support_chat.app.services.assistants.assistant_base import Assistant, llm
from customer_support_chat.app.core.state import State
from pydantic import BaseModel, Field

# Define task delegation tools
class ToFlightBookingAssistant(BaseModel):
    """将任务转交给专门处理航班预订、改签与取消的助手。"""
    request: str = Field(description="航班预订或变更助手需要继续处理的用户请求，应包含已选航班信息。")

class ToBookCarRental(BaseModel):
    """将任务转交给专门处理租车预订的助手。"""
    location: str = Field(description="用户想要租车的地点，未知时使用 Unknown。", default="Unknown")
    start_date: str = Field(description="租车开始日期，未知时使用 Unknown。", default="Unknown")
    end_date: str = Field(description="租车结束日期，未知时使用 Unknown。", default="Unknown")
    request: str = Field(description="用户关于租车的其他补充信息或要求。")

class ToHotelBookingAssistant(BaseModel):
    """将任务转交给专门处理酒店预订、修改和取消的助手。"""
    location: str = Field(description="用户想要预订酒店的地点。若是取消请求且未提供地点，请使用“Unknown”。", default="Unknown")
    checkin_date: str = Field(description="酒店入住日期。若是取消请求且未提供日期，请使用“Unknown”。", default="Unknown")
    checkout_date: str = Field(description="酒店退房日期。若是取消请求且未提供日期，请使用“Unknown”。", default="Unknown")
    request: str = Field(description="用户关于酒店操作（预订、取消、修改）的其他补充信息或要求。")

class ToBookExcursion(BaseModel):
    """将任务转交给专门处理国内行程推荐及本地出游预订的助手。"""
    location: str = Field(description="用户想要咨询或预订行程的城市或地区。")
    request: str = Field(description="用户关于本地游、周边游或景点行程的其他补充信息或要求。")

# Primary assistant prompt
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名面向中国国内航空出行场景的乐于助人的客服助手。"
            "你的主要职责是查询国内航班信息、航司政策以及相关出行服务信息，以回答客户的问题。"
            "只有当用户明确询问‘我的订单’、订单状态、确认号，或提供正式 order_id/order_no 时，才使用 fetch_user_orders 或 fetch_order_detail 查询 PostgreSQL。"
            "hotel_id、rental_id、flight_id、recommendation_id 都是待预订产品 ID，不是 order_id；用户带这些产品 ID 表达‘选择、预订、下单’时必须委派给对应子助手，绝不能查询已有订单。"
            "当客户需要专门服务的帮助时，你必须将任务委派给合适的助手："
            "\n\n委派规则（必须始终委派，绝不要尝试自行处理以下事项）："
            "- 航班预订/改签/取消 → ToFlightBookingAssistant"
            "- 租车预订/修改/取消 → ToBookCarRental"
            "- 酒店预订/修改/取消/状态查询 → ToHotelBookingAssistant"
            "- 国内行程推荐/本地出游项目 → ToBookExcursion"
            "\n\n对于酒店相关操作，即使是以下表达也必须委派："
            "- “取消我的酒店”“把它取消掉”（当上下文指的是酒店时）"
            "- “查看酒店状态”“酒店预订状态”"
            "- “修改我的酒店预订”“更改酒店日期”"
            "\n\n重要：如果用户在一次提问中涉及多个服务（例如“查看租车和酒店状态”），"
            "不要同时委派给多个助手。你应当自行处理，方式如下："
            "1. 使用 search_flights 检查当前预订"
            "2. 总结你能看到的信息"
            "3. 请用户明确说明希望优先获得哪项服务的详细帮助"
            "\n\n一次只能委派给一个助手。单次回复中绝不要发起多个委派调用。"
            "\n\n只有这些专门助手才有权限执行相关变更。"
            "用户并不知道这些不同的专门助手存在，所以不要向用户提及；只需通过函数调用静默委派。"
            "请向客户提供详细信息，并且在认定信息不可用之前，始终再次核对数据库。"
            "搜索时要保持足够坚持。如果第一次搜索没有结果，就扩大搜索范围。"
            "如果搜索结果为空，在放弃之前先进一步扩展搜索。"
            "\n\n当前用户的航班信息：\n<Flights>\n{user_info}\n</Flights>"
            "\n当前时间：{time}。",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# Primary assistant tools
primary_assistant_tools = [
    search_flights,
    lookup_policy,
    fetch_user_orders,
    fetch_order_detail,
    ToFlightBookingAssistant,
    ToBookCarRental,
    ToHotelBookingAssistant,
    ToBookExcursion,
]

# Create the primary assistant runnable
primary_assistant_runnable = primary_assistant_prompt | llm.bind_tools(primary_assistant_tools)

# Instantiate the primary assistant
primary_assistant = Assistant(primary_assistant_runnable)
