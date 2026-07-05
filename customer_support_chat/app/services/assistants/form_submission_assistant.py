from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from customer_support_chat.app.services.tools.forms import submit_form
from customer_support_chat.app.services.assistants.assistant_base import Assistant, llm, CompleteOrEscalate
from pydantic import BaseModel, Field
from typing import Dict, Any

# 定义表单提交任务委派工具
class ToFormSubmission(BaseModel):
    """将任务转交给专门处理表单提交的助手。"""
    form_data: Dict[str, Any] = Field(description="包含表单字段名和用户输入值的字典。")

# 表单提交助手提示词
form_submission_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名专门负责用户表单提交的助手。"
            "你的主要职责是先收集用户所需信息，然后调用 submit_form 工具将数据提交到指定的接口地址。"
            "该表单包含以下必填字段："
            "- 'your-name'：用户姓名 "
            "- 'your-email'：用户邮箱地址 "
            "- 'your-subject'：咨询主题 "
            "此外，表单还固定包含参数 '_wpcf7': 942。"
            "在提交表单前，你必须先向用户收集完整的三个必填字段。"
            "如果用户没有提供完整信息，请礼貌地补问缺失字段。"
            "在正式提交前，一定要先与用户确认提交内容。"
            "如果用户的需求与表单提交无关，请使用 CompleteOrEscalate 工具将控制权交还给主助手。"
            "为便于排查问题，请在合适时提供本次提交数据的详细信息。"
            "当前时间：{time}。",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# 表单提交助手工具
form_submission_assistant_tools = [
    submit_form,
    CompleteOrEscalate,
]

# 创建表单提交助手可执行对象
form_submission_assistant_runnable = form_submission_assistant_prompt | llm.bind_tools(form_submission_assistant_tools)

# 实例化表单提交助手
form_submission_assistant = Assistant(form_submission_assistant_runnable)
