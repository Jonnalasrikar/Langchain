from dotenv import load_dotenv
load_dotenv()
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from typing import List
from pydantic import BaseModel,Field
class Source(BaseModel):
    """schema for a source used by the agent"""
    url:str=Field(description="url of the source")
class AgentResponse(BaseModel):
    """Schema for agent response"""
    answer:str=Field(description="agent answer to the query")
    sources:List[Source]=Field(description="list of sources used by the agent",default_factory=list)
llm=ChatAnthropic(model="claude-haiku-4-5-20251001",temperature=0.0)
tools=[TavilySearch()]
agent=create_agent(model=llm,tools=tools,response_format=AgentResponse)
result=agent.invoke({"messages":[HumanMessage(content="search for a job postings from linkedln and carrers website if there is any opening for langchain professionals?")]})
print(result)

