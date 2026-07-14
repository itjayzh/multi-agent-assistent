from customer_support_chat.app.core.settings import get_settings

settings = get_settings()
APPROVAL_DISABLED_MESSAGE = "当前版本未启用审批功能。"


def approvals_enabled() -> bool:
    """返回是否启用 Web 审批流程。"""
    return settings.ENABLE_APPROVALS
