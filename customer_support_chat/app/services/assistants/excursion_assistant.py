from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from customer_support_chat.app.services.tools import (
    search_trip_recommendations,
    book_excursion,
    update_excursion,
    cancel_excursion,
    fetch_order_detail,
    fetch_user_orders,
)
from customer_support_chat.app.services.assistants.assistant_base import Assistant, CompleteOrEscalate, llm

# 行程助手提示词
excursion_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名专门负责本地游、周边游和出游项目推荐的助手。"
            "当用户需要查询或预订推荐行程时，主助手会将任务交给你处理。"
            "请根据用户偏好搜索可用的行程推荐，并与用户确认预订细节。"
            "用户提供 recommendation_id 并明确要求预订时，必须调用 book_excursion；该 ID 是产品 ID，不是 order_id。"
            "用户询问订单或状态时，必须使用 fetch_user_orders 或 fetch_order_detail 查询 PostgreSQL。"
            "修改和取消必须传入正式 order_id，而不是推荐产品 ID。"
            "如果你需要更多信息，或者用户改变了主意，请将任务升级回主助手。"
            "搜索时要保持耐心，如果第一次没有查到结果，请适当扩大查询范围继续尝试。"
            "请记住，只有在成功调用相关工具之后，预订操作才算真正完成。"
            "\n当前时间：{time}。"
            '\n\n如果用户提出的需求超出了你当前工具的处理范围，请使用 "CompleteOrEscalate" 将对话交还给主助手。不要浪费用户时间，也不要编造不存在的工具或函数。'
            "\n\n以下情况应当使用 CompleteOrEscalate：\n"
            " - “算了，我还是自己单独预订吧”\n"
            " - “我还得先确认当地交通怎么安排”\n"
            " - “等等，我还没订机票，我先去处理机票”\n"
            " - “这个出游项目已经预订好了”",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# 行程助手工具
book_excursion_safe_tools = [search_trip_recommendations, fetch_user_orders, fetch_order_detail, CompleteOrEscalate]
book_excursion_sensitive_tools = [book_excursion, update_excursion, cancel_excursion]
book_excursion_tools = book_excursion_safe_tools + book_excursion_sensitive_tools

# 创建行程助手可执行对象
book_excursion_runnable = excursion_prompt | llm.bind_tools(
    book_excursion_tools
)

# 实例化行程助手
excursion_assistant = Assistant(book_excursion_runnable)
