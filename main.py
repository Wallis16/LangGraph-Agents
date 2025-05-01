from research_agent.agent import create_reseach_assistant
from langchain_core.messages import HumanMessage
from langfuse.callback import CallbackHandler

interview_graph = create_reseach_assistant()

with open("research_agent.png", "wb") as png:
    png.write(interview_graph.get_graph().draw_mermaid_png())

topic = "The benefits of adopting AI agents."
goals= """

1. Highlight the key factors that are important to the topic.
2. Make a clear summarize to make the search easier.

"""

messages = [HumanMessage(f"Create a smart search looking for key factors from {topic}")]

langfuse_handler = CallbackHandler()

for s in interview_graph.stream({"goals": goals, "messages": messages},
                      config={"callbacks": [langfuse_handler]}):
    print(s)

# interview = interview_graph.invoke({"goals": goals, "messages": messages})
# print(interview['sections'][0])