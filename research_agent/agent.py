from langchain_community.tools.tavily_search import TavilySearchResults
from typing import  Annotated
from langgraph.graph import MessagesState
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from research_agent.instructions import question_instructions, search_instructions, answer_instructions

import os
import operator

from dotenv import load_dotenv

load_dotenv()

os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')

class ResearchState(MessagesState):
    context: Annotated[list, operator.add] # Source docs
    goals: str
    sections: list 

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

model_id = "llama-3.3-70b-versatile"
llm = ChatGroq(model = model_id) #'llama-3.3-70b-specdec')#'llama3-70b-8192')
tavily_search = TavilySearchResults(max_results=3)

def generate_search(state: ResearchState):
    """ Node to generate a search query """

    messages = state["messages"]
    goals = state["goals"]

    # Generate question 
    system_message = question_instructions.format(goals=goals)

    question = llm.invoke([SystemMessage(content=system_message)]+messages)
    
    # Write messages to state
    return {"messages": [question]}


def search_web(state: ResearchState):
    
    """ Retrieve docs from web search """

    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)
    
    search_query = structured_llm.invoke([search_instructions]+state['messages'])
    # Search
    search_docs = tavily_search.invoke(search_query.search_query)
     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 

def write_section(state: ResearchState):

    """ Node create the research """

    # Get state
    context = state["context"]
    messages = state["messages"]
    
    system_message = answer_instructions.format(context=context)

    section = llm.invoke([SystemMessage(content=system_message)]+messages) 
    
    # Append it to state
    return {"sections": [section]}

# Add nodes
graph_builder = StateGraph(ResearchState)
graph_builder.add_node("generate_search", generate_search)
graph_builder.add_node("search_web", search_web)
graph_builder.add_node("write_section", write_section)

# Add edges
graph_builder.add_edge(START, "generate_search")
graph_builder.add_edge("generate_search", "search_web")
graph_builder.add_edge("search_web", "write_section")
graph_builder.add_edge("write_section", END)

def create_reseach_assistant():

    interview_graph = graph_builder.compile().with_config(run_name="research_agent")

    return interview_graph