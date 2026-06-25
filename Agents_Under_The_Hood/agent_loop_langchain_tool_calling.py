from dotenv import load_dotenv

load_dotenv()
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import SystemMessage,HumanMessage,ToolMessage
from langsmith import traceable
MAX_ITERATIONS=10
MODEL="claude-opus-4-8"

@tool
def get_product_price(product:str)->float:
    """looks for the product price in the catalog"""
    print(f"Execution of get_product_price({product})")
    prices={"laptop":100000,"mouse":2500,"keyboard":5000}
    return prices.get(product,0)
@tool
def get_discount(price:float,discount_tier:str)->float:
    """applies the discount for the product based on tier
    available tiers are gold,silver,bronze"""
    print(f"Execution of get_discount({price},{discount_tier})")
    discount_percentages={"gold":20,"silver":10,"bronze":5}
    discount=discount_percentages.get(discount_tier,0)
    return round(price*(1-discount/100),2)

@traceable(name="langchain agent loop")
def run_agent(question:str):
    tools=[get_product_price,get_discount]
    tools_dict={t.name:t for t in tools}
    llm=init_chat_model(MODEL)
    llm_with_tools=llm.bind_tools(tools)
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]
    for iteration in range(1,MAX_ITERATIONS+1):
        print(f"\n\nIteration {iteration}")
        ai_message=llm_with_tools.invoke(messages)
        tool_calls=ai_message.tool_calls
        if not tool_calls:
            print(f"final answer{ai_message.content}")
            return ai_message.content
        tool_call=tool_calls[0]
        toolcall_name=tool_call.get("name")
        toolcall_args=tool_call.get("args",{})
        toolcall_id=tool_call.get("id")
        print(f"tool selected {toolcall_name} with args {toolcall_args}")
        tool_to_use=tools_dict.get(toolcall_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {toolcall_name} not found")
        observation =tool_to_use.invoke(toolcall_args)
        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation),tool_call_id=toolcall_id))
    print("Error:max iterations reached without final answer")
    return None


if __name__=="__main__":
    print("hello from langchain agent")
    run_agent("What is the price of the laptop after applying gold discount?")