import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from psycopg.types.json import Jsonb

from customer_support_chat.app.core.database import get_connection


def serialize_time(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def normalize_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def ensure_session(session_id: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO user_sessions (session_id)
                VALUES (%s)
                ON CONFLICT (session_id) DO NOTHING
                """,
                (session_id,),
            )


def build_default_session(session_id: str) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "chat_history": [],
        "pending_action": None,
        "user_decision": None,
        "operation_log": [],
        "created_at": datetime.now().isoformat(),
        "user_profile": {},
        "config": {},
    }


def get_user_session(session_id: str) -> Dict[str, Any]:
    """从 PostgreSQL 读取完整会话数据。"""
    ensure_session(session_id)
    with get_connection(rows_as_dict=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT session_id, user_profile, config, created_at
                FROM user_sessions
                WHERE session_id = %s
                """,
                (session_id,),
            )
            session = cursor.fetchone()
            cursor.execute(
                """
                SELECT user_message, ai_response, created_at
                FROM chat_history
                WHERE session_id = %s
                ORDER BY created_at, id
                """,
                (session_id,),
            )
            history = cursor.fetchall()
            cursor.execute(
                """
                SELECT action, decision
                FROM pending_actions
                WHERE session_id = %s AND status = 'pending'
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (session_id,),
            )
            pending = cursor.fetchone()
            cursor.execute(
                """
                SELECT log_type, title, content, created_at
                FROM operation_logs
                WHERE session_id = %s
                ORDER BY created_at, id
                """,
                (session_id,),
            )
            logs = cursor.fetchall()

    return {
        "session_id": session["session_id"],
        "chat_history": [
            {
                "timestamp": serialize_time(item["created_at"]),
                "user_message": item["user_message"],
                "ai_response": item["ai_response"],
            }
            for item in history
        ],
        "pending_action": pending["action"] if pending else None,
        "user_decision": pending["decision"] if pending else None,
        "operation_log": [
            {
                "type": item["log_type"],
                "title": item["title"],
                "content": item["content"],
                "timestamp": serialize_time(item["created_at"]),
            }
            for item in logs
        ],
        "created_at": serialize_time(session["created_at"]),
        "user_profile": session["user_profile"],
        "config": session["config"],
    }


def sync_session_identity(
    session_id: str,
    user_profile: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """将登录用户身份同步到 PostgreSQL 会话。"""
    user_id = user_profile.get("id")
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO user_sessions (
                    session_id, user_id, user_profile, config
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    user_profile = EXCLUDED.user_profile,
                    config = EXCLUDED.config
                """,
                (session_id, user_id, Jsonb(user_profile), Jsonb(config)),
            )
    return get_user_session(session_id)


def update_user_chat_history(session_id: str, user_message: str, ai_response: str) -> None:
    ensure_session(session_id)
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO chat_history (session_id, user_message, ai_response)
                VALUES (%s, %s, %s)
                """,
                (session_id, user_message, ai_response),
            )


def set_pending_action(session_id: str, action_details: Dict[str, Any]) -> int:
    """创建新的待审批动作，并取消该会话之前未完成的动作。"""
    ensure_session(session_id)
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE pending_actions
                SET status = 'cancelled', resolved_at = CURRENT_TIMESTAMP
                WHERE session_id = %s AND status IN ('pending', 'processing')
                """,
                (session_id,),
            )
            cursor.execute(
                """
                INSERT INTO pending_actions (session_id, action)
                VALUES (%s, %s)
                RETURNING id
                """,
                (session_id, Jsonb(action_details)),
            )
            return cursor.fetchone()[0]


def get_pending_action(session_id: str) -> Optional[Dict[str, Any]]:
    ensure_session(session_id)
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT action
                FROM pending_actions
                WHERE session_id = %s AND status = 'pending'
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (session_id,),
            )
            row = cursor.fetchone()
    return row[0] if row else None


def claim_pending_action(session_id: str) -> Optional[Dict[str, Any]]:
    """原子领取待审批动作，避免重复批准并发执行。"""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE pending_actions
                SET status = 'processing'
                WHERE id = (
                    SELECT id
                    FROM pending_actions
                    WHERE session_id = %s AND status = 'pending'
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING action
                """,
                (session_id,),
            )
            row = cursor.fetchone()
    return row[0] if row else None


def resolve_pending_action(
    session_id: str,
    decision: str,
    *,
    error_message: Optional[str] = None,
) -> None:
    status = "failed" if error_message else ("approved" if decision == "approve" else "rejected")
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE pending_actions
                SET status = %s,
                    decision = %s,
                    error_message = %s,
                    resolved_at = CURRENT_TIMESTAMP
                WHERE id = (
                    SELECT id
                    FROM pending_actions
                    WHERE session_id = %s AND status = 'processing'
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                )
                """,
                (status, decision, error_message, session_id),
            )


def clear_pending_action(session_id: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE pending_actions
                SET status = 'cancelled', resolved_at = CURRENT_TIMESTAMP
                WHERE session_id = %s AND status IN ('pending', 'processing')
                """,
                (session_id,),
            )


def set_user_decision(session_id: str, decision: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE pending_actions
                SET decision = %s
                WHERE id = (
                    SELECT id FROM pending_actions
                    WHERE session_id = %s AND status IN ('pending', 'processing')
                    ORDER BY created_at DESC, id DESC LIMIT 1
                )
                """,
                (decision, session_id),
            )


def get_user_decision(session_id: str) -> Optional[str]:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT decision FROM pending_actions
                WHERE session_id = %s AND status IN ('pending', 'processing')
                ORDER BY created_at DESC, id DESC LIMIT 1
                """,
                (session_id,),
            )
            row = cursor.fetchone()
    return row[0] if row else None


def clear_user_decision(session_id: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE pending_actions SET decision = NULL
                WHERE session_id = %s AND status = 'pending'
                """,
                (session_id,),
            )


def add_operation_log(session_id: str, log_entry: Dict[str, Any]) -> None:
    ensure_session(session_id)
    timestamp = log_entry.get("timestamp")
    created_at = datetime.fromisoformat(timestamp) if timestamp else datetime.now()
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO operation_logs (
                    session_id, log_type, title, content, created_at
                ) VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    log_entry.get("type", "system_message"),
                    log_entry.get("title", "操作日志"),
                    normalize_content(log_entry.get("content", "")),
                    created_at,
                ),
            )


def get_operation_log(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    ensure_session(session_id)
    with get_connection(rows_as_dict=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT log_type, title, content, created_at
                FROM operation_logs
                WHERE session_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (session_id, limit),
            )
            rows = cursor.fetchall()
    return [
        {
            "type": row["log_type"],
            "title": row["title"],
            "content": row["content"],
            "timestamp": serialize_time(row["created_at"]),
        }
        for row in reversed(rows)
    ]


def clear_operation_log(session_id: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM operation_logs WHERE session_id = %s", (session_id,))


def attach_external_approval(
    action_id: int,
    *,
    provider: str,
    instance_code: str,
    task_id: Optional[str] = None,
    external_status: Optional[str] = None,
) -> None:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE pending_actions
                SET provider = %s,
                    external_instance_code = %s,
                    external_task_id = %s,
                    external_status = %s
                WHERE id = %s
                """,
                (provider, instance_code, task_id, external_status, action_id),
            )


def get_action_by_external_instance(instance_code: str) -> Optional[Dict[str, Any]]:
    with get_connection(rows_as_dict=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, session_id, action, status, decision, provider,
                       external_instance_code, external_task_id,
                       external_approver_user_id, external_approver_id_type,
                       external_status
                FROM pending_actions
                WHERE external_instance_code = %s
                """,
                (instance_code,),
            )
            row = cursor.fetchone()
    return dict(row) if row else None


def attach_external_approval_task(
    instance_code: str,
    task_id: str,
    approver_id: Optional[str],
    approver_id_type: Optional[str],
) -> None:
    """保存飞书审批任务，供后续任务操作使用。"""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE pending_actions
                SET external_task_id = %s,
                    external_approver_user_id = %s,
                    external_approver_id_type = %s
                WHERE external_instance_code = %s
                """,
                (task_id, approver_id, approver_id_type, instance_code),
            )


def update_external_approval_status(instance_code: str, external_status: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE pending_actions
                SET external_status = %s
                WHERE external_instance_code = %s
                """,
                (external_status, instance_code),
            )
