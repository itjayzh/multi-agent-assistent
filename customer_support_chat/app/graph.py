from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage
from customer_support_chat.app.core.settings import get_settings

from customer_support_chat.app.core.state import State
from customer_support_chat.app.core.logger import logger
from customer_support_chat.app.services.utils import (
  create_tool_node_with_fallback,
  flight_info_to_string,
  create_entry_node,
)
from customer_support_chat.app.services.tools.flights import fetch_user_flight_information
# Import guardrail agents
from customer_support_chat.app.services.guardrails.guardrail_agents import (
    jailbreak_guardrail_agent,
    jailbreak_guardrail_agent_instructions,
    relevance_guardrail_agent,
    relevance_guardrail_agent_instructions,
)
from customer_support_chat.app.services.assistants.assistant_base import (
  Assistant,
  CompleteOrEscalate,
  llm,
)
from customer_support_chat.app.services.assistants.primary_assistant import (
  primary_assistant,
  primary_assistant_tools,
  ToFlightBookingAssistant,
  ToBookCarRental,
  ToHotelBookingAssistant,
  ToBookExcursion,
)
from customer_support_chat.app.services.assistants.flight_booking_assistant import (
  flight_booking_assistant,
  update_flight_safe_tools,
  update_flight_sensitive_tools,
)
from customer_support_chat.app.services.assistants.car_rental_assistant import (
  car_rental_assistant,
  book_car_rental_safe_tools,
  book_car_rental_sensitive_tools,
)
from customer_support_chat.app.services.assistants.hotel_booking_assistant import (
  hotel_booking_assistant,
  book_hotel_safe_tools,
  book_hotel_sensitive_tools,
)
from customer_support_chat.app.services.assistants.excursion_assistant import (
  excursion_assistant,
  book_excursion_safe_tools,
  book_excursion_sensitive_tools,
)

# Initialize the graph
builder = StateGraph(State)
settings = get_settings()

def user_info(state: State, config: RunnableConfig):
  # Fetch user flight information
  flight_info = fetch_user_flight_information.invoke(input={}, config=config)
  user_info_str = flight_info_to_string(flight_info)
  return {"user_info": user_info_str}

builder.add_node("fetch_user_info", user_info)

# --- Security Guardrail Node ---
def guardrail_check(state: State, config: RunnableConfig):
    """Node to check user input for safety and relevance."""
    # Get the latest user message
    # Assuming the last message is always from the user in this context
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not user_messages:
        logger.warning("No user message found for guardrail check. Allowing.")
        return {
            "messages": [AIMessage(content="未找到需要处理的用户消息，请重新输入。")],
            "guardrail_blocked": True,
        }
    
    latest_user_message = user_messages[-1]
    user_input = latest_user_message.content
    
    logger.info(f"🛡️ Checking safety and relevance for user input: '{user_input}'")
    
    # 1. Check for Jailbreak attempts
    jailbreak_prompt = f"{jailbreak_guardrail_agent_instructions}\n\nUser Input: {user_input}"
    jailbreak_result = jailbreak_guardrail_agent.invoke(jailbreak_prompt)
    
    if not jailbreak_result.is_safe:
        logger.warning(f"🚨 Jailbreak attempt detected: {jailbreak_result.reasoning}")
        return {
            "messages": [AIMessage(content="抱歉，我无法处理试图绕过系统安全规则的请求。")],
            "guardrail_blocked": True,
        }

    # 2. Check for Relevance
    relevance_prompt = f"{relevance_guardrail_agent_instructions}\n\nUser Input: {user_input}"
    relevance_result = relevance_guardrail_agent.invoke(relevance_prompt)
    
    if not relevance_result.is_relevant:
        logger.warning(f"⚠️ Irrelevant input detected: {relevance_result.reasoning}")
        return {
            "messages": [AIMessage(content="抱歉，我目前只能协助处理航班、酒店、租车、行程推荐及相关政策问题。")],
            "guardrail_blocked": True,
        }
        
    # If both checks pass, the input is safe and (at least potentially) relevant.
    logger.info("✅ Input passed safety and relevance checks.")
    return {"messages": [], "guardrail_blocked": False}

builder.add_node("guardrail_check", guardrail_check)

# --- Graph Edges ---
builder.add_edge(START, "fetch_user_info")
builder.add_edge("fetch_user_info", "guardrail_check")


def route_after_guardrail(state: State) -> Literal["primary_assistant", "__end__"]:
  return END if state.get("guardrail_blocked", False) else "primary_assistant"


builder.add_conditional_edges("guardrail_check", route_after_guardrail)

# Flight Booking Assistant
builder.add_node(
  "enter_update_flight",
  create_entry_node("Flight Updates & Booking Assistant", "update_flight"),
)
builder.add_node("update_flight", flight_booking_assistant)
builder.add_edge("enter_update_flight", "update_flight")
builder.add_node(
  "update_flight_safe_tools",
  create_tool_node_with_fallback(update_flight_safe_tools),
)
builder.add_node(
  "update_flight_sensitive_tools",
  create_tool_node_with_fallback(update_flight_sensitive_tools),
)

def route_update_flight(state: State) -> Literal[
  "update_flight_safe_tools",
  "update_flight_sensitive_tools",
  "primary_assistant",
  "__end__",
]:
  route = tools_condition(state)
  if route == END:
      return END
  tool_calls = state["messages"][-1].tool_calls
  safe_toolnames = [t.name for t in update_flight_safe_tools]
  if all(tc["name"] in safe_toolnames for tc in tool_calls):
      return "update_flight_safe_tools"
  return "update_flight_sensitive_tools"

# Helper function to check if CompleteOrEscalate was executed
def should_route_to_primary(state: State) -> bool:
    if state["messages"] and len(state["messages"]) > 0:
        last_message = state["messages"][-1]
        # 检查最后一条工具消息是否来自 CompleteOrEscalate。
        if hasattr(last_message, 'content') and isinstance(last_message.content, str):
            return '任务已完成或已升级给主助手' in last_message.content
    return False

# Route functions for tool execution results
def route_update_flight_tools(state: State) -> Literal["update_flight", "primary_assistant"]:
    return "primary_assistant" if should_route_to_primary(state) else "update_flight"

def route_car_rental_tools(state: State) -> Literal["book_car_rental", "primary_assistant"]:
    return "primary_assistant" if should_route_to_primary(state) else "book_car_rental"

def route_hotel_tools(state: State) -> Literal["book_hotel", "primary_assistant"]:
    return "primary_assistant" if should_route_to_primary(state) else "book_hotel"

def route_excursion_tools(state: State) -> Literal["book_excursion", "primary_assistant"]:
    return "primary_assistant" if should_route_to_primary(state) else "book_excursion"

builder.add_conditional_edges("update_flight_safe_tools", route_update_flight_tools)
builder.add_conditional_edges("update_flight_sensitive_tools", route_update_flight_tools)
builder.add_conditional_edges("update_flight", route_update_flight)

# Car Rental Assistant
builder.add_node(
  "enter_book_car_rental",
  create_entry_node("Car Rental Assistant", "book_car_rental"),
)
builder.add_node("book_car_rental", car_rental_assistant)
builder.add_edge("enter_book_car_rental", "book_car_rental")
builder.add_node(
  "book_car_rental_safe_tools",
  create_tool_node_with_fallback(book_car_rental_safe_tools),
)
builder.add_node(
  "book_car_rental_sensitive_tools",
  create_tool_node_with_fallback(book_car_rental_sensitive_tools),
)

def route_book_car_rental(state: State) -> Literal[
  "book_car_rental_safe_tools",
  "book_car_rental_sensitive_tools",
  "primary_assistant",
  "__end__",
]:
  route = tools_condition(state)
  if route == END:
      return END
  tool_calls = state["messages"][-1].tool_calls
  safe_toolnames = [t.name for t in book_car_rental_safe_tools]
  if all(tc["name"] in safe_toolnames for tc in tool_calls):
      return "book_car_rental_safe_tools"
  return "book_car_rental_sensitive_tools"

builder.add_conditional_edges("book_car_rental_safe_tools", route_car_rental_tools)
builder.add_conditional_edges("book_car_rental_sensitive_tools", route_car_rental_tools)
builder.add_conditional_edges("book_car_rental", route_book_car_rental)

# Hotel Booking Assistant
builder.add_node(
  "enter_book_hotel",
  create_entry_node("Hotel Booking Assistant", "book_hotel"),
)
builder.add_node("book_hotel", hotel_booking_assistant)
builder.add_edge("enter_book_hotel", "book_hotel")
builder.add_node(
  "book_hotel_safe_tools",
  create_tool_node_with_fallback(book_hotel_safe_tools),
)
builder.add_node(
  "book_hotel_sensitive_tools",
  create_tool_node_with_fallback(book_hotel_sensitive_tools),
)

def route_book_hotel(state: State) -> Literal[
  "book_hotel_safe_tools",
  "book_hotel_sensitive_tools",
  "primary_assistant",
  "__end__",
]:
  route = tools_condition(state)
  if route == END:
      return END
  tool_calls = state["messages"][-1].tool_calls
  safe_toolnames = [t.name for t in book_hotel_safe_tools]
  if all(tc["name"] in safe_toolnames for tc in tool_calls):
      return "book_hotel_safe_tools"
  return "book_hotel_sensitive_tools"

builder.add_conditional_edges("book_hotel_safe_tools", route_hotel_tools)
builder.add_conditional_edges("book_hotel_sensitive_tools", route_hotel_tools)
builder.add_conditional_edges("book_hotel", route_book_hotel)

# Excursion Assistant
builder.add_node(
  "enter_book_excursion",
  create_entry_node("Trip Recommendation Assistant", "book_excursion"),
)
builder.add_node("book_excursion", excursion_assistant)
builder.add_edge("enter_book_excursion", "book_excursion")
builder.add_node(
  "book_excursion_safe_tools",
  create_tool_node_with_fallback(book_excursion_safe_tools),
)
builder.add_node(
  "book_excursion_sensitive_tools",
  create_tool_node_with_fallback(book_excursion_sensitive_tools),
)

def route_book_excursion(state: State) -> Literal[
  "book_excursion_safe_tools",
  "book_excursion_sensitive_tools",
  "primary_assistant",
  "__end__",
]:
  route = tools_condition(state)
  if route == END:
      return END
  tool_calls = state["messages"][-1].tool_calls
  safe_toolnames = [t.name for t in book_excursion_safe_tools]
  if all(tc["name"] in safe_toolnames for tc in tool_calls):
      return "book_excursion_safe_tools"
  return "book_excursion_sensitive_tools"

builder.add_conditional_edges("book_excursion_safe_tools", route_excursion_tools)
builder.add_conditional_edges("book_excursion_sensitive_tools", route_excursion_tools)
builder.add_conditional_edges("book_excursion", route_book_excursion)

# Primary Assistant
builder.add_node("primary_assistant", primary_assistant)
builder.add_node(
  "primary_assistant_tools", create_tool_node_with_fallback(primary_assistant_tools)
)

def route_primary_assistant(state: State) -> Literal[
  "primary_assistant_tools",
  "enter_update_flight",
  "enter_book_car_rental",
  "enter_book_hotel",
  "enter_book_excursion",
  "__end__",
]:
  route = tools_condition(state)
  if route == END:
      return END
  tool_calls = state["messages"][-1].tool_calls
  if tool_calls:
      tool_name = tool_calls[0]["name"]
      if tool_name == ToFlightBookingAssistant.__name__:
          return "enter_update_flight"
      elif tool_name == ToBookCarRental.__name__:
          return "enter_book_car_rental"
      elif tool_name == ToHotelBookingAssistant.__name__:
          return "enter_book_hotel"
      elif tool_name == ToBookExcursion.__name__:
          return "enter_book_excursion"
      else:
          return "primary_assistant_tools"
  return "primary_assistant"

builder.add_conditional_edges(
  "primary_assistant",
  route_primary_assistant,
  {
      "enter_update_flight": "enter_update_flight",
      "enter_book_car_rental": "enter_book_car_rental",
      "enter_book_hotel": "enter_book_hotel",
      "enter_book_excursion": "enter_book_excursion",
      "primary_assistant_tools": "primary_assistant_tools",
      END: END,
  },
)
builder.add_edge("primary_assistant_tools", "primary_assistant")

# Compile the graph with interrupts
interrupt_nodes = [
  "update_flight_sensitive_tools",
  "book_car_rental_sensitive_tools",
  "book_hotel_sensitive_tools",
  "book_excursion_sensitive_tools",
  # No interrupts needed for new assistants as they don't have sensitive operations
]

memory = MemorySaver()
multi_agentic_graph = builder.compile(
  checkpointer=memory,
  interrupt_before=interrupt_nodes if settings.ENABLE_APPROVALS else [],
)
