# customer_support_chat/app/services/tools/forms.py

import httpx
from langchain_core.tools import tool
from customer_support_chat.app.core.settings import get_settings
from typing import Dict, Any

settings = get_settings()

@tool
def submit_form(form_data: Dict[str, Any]) -> str:
    """将用户表单数据提交到指定 API。
    
    参数:
        form_data: 以字段名为键、用户输入为值的表单数据字典。
                   必须包含以下必填字段：'your-name'、'your-email'、'your-subject'。
        
    返回:
        API 返回的确认消息或错误消息。
    """
    if not settings.FORM_SUBMISSION_API_URL:
        raise ValueError("表单提交 API URL 未配置。")
    
    # 校验必填字段
    required_fields = ['your-name', 'your-email', 'your-subject']
    missing_fields = [field for field in required_fields if field not in form_data or not form_data[field]]
    
    if missing_fields:
        raise ValueError(f"缺少必填表单字段：{', '.join(missing_fields)}")
    
    # 添加固定的 _wpcf7 参数
    final_form_data = form_data.copy()
    final_form_data["_wpcf7"] = 946
    
    with httpx.Client() as client:
        try:
            response = client.post(
                settings.FORM_SUBMISSION_API_URL,
                json=final_form_data
            )
            
            # 输出详细响应信息，便于调试
            print(f"表单提交 API 响应状态码: {response.status_code}")
            print(f"表单提交 API 响应头: {dict(response.headers)}")
            
            try:
                result = response.json()
                print(f"表单提交 API 响应 JSON: {result}")
            except Exception as json_error:
                print(f"表单提交 API 响应文本（非 JSON）: {response.text}")
                result = {}
            
            response.raise_for_status()
            
            # 返回成功消息或 API 给出的具体结果
            # 这里可能需要根据实际 API 响应格式进一步调整
            if result.get("status") == "success" or response.status_code == 200:
                return f"表单提交成功，感谢你的提交！"
            else:
                return f"表单提交可能遇到问题。API 返回：{result}"
                
        except httpx.HTTPStatusError as e:
            raise Exception(f"提交表单时发生 HTTP 错误：{e}")
        except Exception as e:
            raise Exception(f"提交表单时发生错误：{e}")
