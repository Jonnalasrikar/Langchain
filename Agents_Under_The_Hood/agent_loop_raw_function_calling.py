from dotenv import load_dotenv
from anthropic import Anthropic
from langsmith import traceable

load_dotenv()

client = Anthropic()

MODEL = "claude-opus-4-1"
MAX_ITERATIONS = 10


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Looks for the product price in the catalog."""
    print(f"Execution of get_product_price({product})")
    prices = {
        "laptop": 100000,
        "mouse": 2500,
        "keyboard": 5000,
    }
    return prices.get(product, 0)


@traceable(run_type="tool")
def get_discount(price: float, discount_tier: str) -> float:
    """Applies discount based on customer tier."""
    print(f"Execution of get_discount({price}, {discount_tier})")

    discounts = {
        "gold": 20,
        "silver": 10,
        "bronze": 5,
    }

    discount = discounts.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


tools = [
    {
        "name": "get_product_price",
        "description": "Looks for the product price in the catalog.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product": {
                    "type": "string"
                }
            },
            "required": ["product"]
        }
    },
    {
        "name": "get_discount",
        "description": "Applies discount using customer tier.",
        "input_schema": {
            "type": "object",
            "properties": {
                "price": {
                    "type": "number"
                },
                "discount_tier": {
                    "type": "string",
                    "enum": ["gold", "silver", "bronze"]
                }
            },
            "required": ["price", "discount_tier"]
        }
    }
]


SYSTEM_PROMPT = """
You are a helpful shopping assistant.

Rules:

1. Always call get_product_price before answering product prices.
2. Never guess a price.
3. If a discount is requested, always call get_discount.
4. Never calculate discounts yourself.
"""


@traceable(run_type="llm")
def claude_chat(messages):
    return client.messages.create(
        model=MODEL,
        system=SYSTEM_PROMPT,
        max_tokens=1024,
        tools=tools,
        messages=messages,
    )


@traceable(name="Claude Agent Loop")
def run_agent(question: str):

    tool_map = {
        "get_product_price": get_product_price,
        "get_discount": get_discount,
    }

    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]

    for i in range(MAX_ITERATIONS):

        print(f"\n******** ITERATION {i+1} ********")

        response = claude_chat(messages)

        # Print text blocks
        for block in response.content:
            if block.type == "text":
                print(block.text)

        # Find tool call
        tool_use = next(
            (block for block in response.content if block.type == "tool_use"),
            None,
        )

        # No tool call -> final answer
        if tool_use is None:
            final_text = "".join(
                block.text
                for block in response.content
                if block.type == "text"
            )

            print("\nFINAL ANSWER:")
            print(final_text)
            return final_text

        print(f"\nTool Selected: {tool_use.name}")
        print(tool_use.input)

        tool_fn = tool_map[tool_use.name]

        result = tool_fn(**tool_use.input)

        print(f"Tool Result: {result}")

        # Add assistant message exactly as returned
        messages.append(
            {
                "role": "assistant",
                "content": response.content,
            }
        )

        # Send tool result back
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": str(result),
                    }
                ],
            }
        )

    print("Max iterations reached.")
    return None


if __name__ == "__main__":
    run_agent(
        "What is the price of the mouse after applying the silver discount?"
    )