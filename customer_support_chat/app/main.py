import uuid
import os  # 导入 os 模块，用于文件操作
from customer_support_chat.app.graph import multi_agentic_graph
from customer_support_chat.app.services.utils import download_and_prepare_db
from customer_support_chat.app.core.logger import logger
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage


def main():
    # 确保数据库已下载并完成初始化
    download_and_prepare_db()

    # 生成并保存图结构可视化文件
    try:
        # 使用 xray=True 生成包含节点细节的图对象
        graph = multi_agentic_graph.get_graph(xray=True)
        # 使用 Mermaid 将图绘制为 PNG 图片
        graph_image = graph.draw_mermaid_png()
        graphs_dir = "./graphs"
        if not os.path.exists(graphs_dir):
            os.makedirs(graphs_dir)
        image_path = os.path.join(graphs_dir, "multi-agent-rag-system-graph.png")
        with open(image_path, "wb") as f:
            f.write(graph_image)
        print(f"图结构文件已保存至：{image_path}")
    except Exception as e:
        logger.error(f"生成图结构可视化文件时发生错误：{e}")
        print("图结构可视化文件生成失败，将继续运行主程序。")

    # 为本次会话生成唯一的线程 ID
    thread_id = str(uuid.uuid4())

    # 配置 passenger_id 和 thread_id
    config = {
        "configurable": {
            "passenger_id": "5102 899977",  # 如有需要，可替换为有效的乘客 ID
            "thread_id": thread_id,
        }
    }

    # 用于跟踪已打印的消息 ID，避免重复输出
    printed_message_ids = set()

    try:
        while True:
            user_input = input("用户：")
            if user_input.strip().lower() in ["quit", "exit", "q"]:
                print("再见！")
                break

            # 通过图工作流处理用户输入
            events = multi_agentic_graph.stream(
                {"messages": [("user", user_input)]}, config, stream_mode="values"
            )

            for event in events:
                messages = event.get("messages", [])
                for message in messages:
                    if message.id not in printed_message_ids:
                        message.pretty_print()
                        printed_message_ids.add(message.id)

            # 检查是否发生中断
            snapshot = multi_agentic_graph.get_state(config)
            while snapshot.next:
                # 在执行敏感工具前发生中断
                user_input = input(
                    "\n是否批准上述操作？输入 'y' 继续；否则请说明你希望修改的内容。\n\n"
                )
                if user_input.strip().lower() == "y":
                    # 继续执行
                    result = multi_agentic_graph.invoke(None, config)
                else:
                    # 向助手反馈用户意见
                    tool_call_id = snapshot.value["messages"][-1].tool_calls[0]["id"]
                    result = multi_agentic_graph.invoke(
                        {
                            "messages": [
                                ToolMessage(
                                    tool_call_id=tool_call_id,
                                    content=f"该 API 调用已被用户拒绝。原因：'{user_input}'。请结合用户反馈继续提供帮助。",
                                )
                            ]
                        },
                        config,
                    )
                # 处理执行结果并输出新增消息
                messages = result.get("messages", [])
                for message in messages:
                    if message.id not in printed_message_ids:
                        message.pretty_print()
                        printed_message_ids.add(message.id)

                        # 更新快照
                snapshot = multi_agentic_graph.get_state(config)

    except Exception as e:
        logger.error(f"程序运行时发生错误：{e}")
        print("程序运行时发生异常错误，请查看日志了解详细信息。")


if __name__ == "__main__":
    main()
