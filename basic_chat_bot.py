from langchain_groq import ChatGroq
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import requests
import matplotlib.pyplot as plt
import os
import configparser

# Define the State Schema
class State(TypedDict):
    messages: Annotated[list, add_messages]
    continue_chat: bool

# Define the chat function
def chat(state: State) -> State:

    config_file = 'config.ini'
    config = configparser.RawConfigParser()
    config.read(config_file)

    os.environ["GROQ_API_KEY"] = config.get('LLM_config', 'groq_api_key')

    model_name = config.get('LLM_config', 'model_name')
    model_temperature = int(config.get('LLM_config', 'model_temperature'))

    llm = ChatGroq(model= model_name, temperature = model_temperature)

    # Generate a response using the language model
    response = llm.invoke(state["messages"])
    # Update the state with the new messages
    state["messages"].append(response)
    # Print the chatbot's response
    print("Chatbot:", state["messages"][-1].content)
    return state

def user_speak(state: State) ->State:
    user_input = input("You: ")
    state["messages"].append(HumanMessage(content = user_input))
    return state

def continue_conversation(state: State) -> State:
    if state["messages"][-1].content == "exit":
        state["continue_chat"] = False
        print("Chatbot: Goodbye!")
    else:
        state["continue_chat"] = True
    return state

def main():

    # Set up the state graph
    graph = StateGraph(State)

    graph.add_node("chat", chat)
    graph.add_node("continue_conversation", continue_conversation)
    graph.add_node("user_speak", user_speak)

    graph.add_edge(START, "user_speak")
    graph.add_edge("user_speak", "continue_conversation")
    graph.add_conditional_edges("continue_conversation", lambda x: x["continue_chat"],
            {
                True: "chat",
                False: END
            })
    graph.add_edge("chat", "user_speak")
    graph.add_edge("user_speak", "continue_conversation")


    state = {"messages": []}
    # Compile the graph
    compiled_graph = graph.compile()

    # Assuming compiled_graph.get_graph().draw_mermaid_png() returns the image data
    image_data = compiled_graph.get_graph().draw_mermaid_png()

    # If you have the image as a file (e.g., saved as PNG), you can display it like this:
    with open("basic_chat_bot.png", "wb") as f:
        f.write(image_data)
        
    # Run the chatbot
    state = compiled_graph.invoke(state)

if __name__ == "__main__":
    main() 