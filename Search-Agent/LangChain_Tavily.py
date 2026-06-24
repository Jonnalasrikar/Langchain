from dotenv import load_dotenv
load_dotenv()
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
llm=ChatAnthropic(model="claude-haiku-4-5-20251001",temperature=0.0)
tools=[TavilySearch()]
agent=create_agent(model=llm,tools=tools)
print(agent.invoke({"messages":[HumanMessage(content="search for a job postings from linkedln and carrers website if there is any opening for langchain professionals?")]}))
