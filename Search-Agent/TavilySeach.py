from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from tavily import TavilyClient

tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
        tool that searches over internet
    Args:
        query:The query to search for
    Returns:
        the search result
    """
    print(f"searching for {query}")
    return tavily.search(query=query)


llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.0)
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from search-agent!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="search for a job postings from linkedln and carrers website if there is any opening for langchain professionals?"
            )
        }
    )
    print(result)


if __name__ == "__main__":
    main()
