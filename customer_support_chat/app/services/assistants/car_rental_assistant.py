from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from customer_support_chat.app.services.tools import (
    search_car_rentals,
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
)
from customer_support_chat.app.services.assistants.assistant_base import Assistant, CompleteOrEscalate, llm

# 租车助手提示词
car_rental_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名专门负责租车预订相关事务的助手。"
            "当用户需要查询、预订、修改或取消租车时，主助手会将任务交给你处理。"
            "请根据用户偏好搜索可用的租车方案，并与用户确认预订细节。"
            "搜索时要保持耐心，如果第一次搜索没有结果，请适当扩大查询范围继续尝试。"
            "如果你需要更多信息，或者用户改变了想法，请将任务升级回主助手。"
            "请记住，只有在成功调用相关工具之后，租车操作才算真正完成。"
            "\n当前时间：{time}。"
            '\n\n如果用户提出的需求超出了你当前工具的处理范围，请使用 "CompleteOrEscalate" 将对话交还给主助手。不要浪费用户时间，也不要编造不存在的工具或函数。'
            "\n\n以下情况应当使用 CompleteOrEscalate：\n"
            " - “这个季节那边天气怎么样？”\n"
            " - “目前有哪些航班可选？”\n"
            " - “算了，我还是自己单独预订吧”\n"
            " - “等等，我还没订机票，我先去处理机票”\n"
            " - “租车已经确认好了”",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# 租车助手工具
book_car_rental_safe_tools = [search_car_rentals, CompleteOrEscalate]
book_car_rental_sensitive_tools = [book_car_rental, update_car_rental, cancel_car_rental]
book_car_rental_tools = book_car_rental_safe_tools + book_car_rental_sensitive_tools

# 创建租车助手可执行对象
book_car_rental_runnable = car_rental_prompt | llm.bind_tools(
    book_car_rental_tools
)

# 实例化租车助手
car_rental_assistant = Assistant(book_car_rental_runnable)
