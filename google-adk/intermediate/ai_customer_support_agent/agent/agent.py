from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from .tools import get_order_status, get_product_stock 

order_checker_tool = FunctionTool(func=get_order_status)
stock_checker_tool = FunctionTool(func=get_product_stock)

root_agent = LlmAgent(
    name="customer_support_agent",
    model="gemini-2.5-flash",
    description="A friendly and helpful chatbot for customer support.",
    instruction="""
    You are a helpful and professional customer support agent. Always greet the user, and assist them clearly and concisely.

    Handle the user's queries using the following logic:

    - If the user asks about **order status**, use the `get_order_status` tool.
    - If the user asks about **product availability** or **stock**, use the `get_product_stock` tool.
    - If you cannot find the requested order or product, apologize politely and offer further assistance.
    - If the request does not match any known tools, try to assist the user using your own knowledge or respond appropriately.

    Use tools only when necessary. If a user query includes both an order number and a product name, prioritize checking the order status first.
    """,
    tools=[order_checker_tool, stock_checker_tool],  
)
