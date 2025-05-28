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
from langchain_core.runnables import graph_mermaid
from langchain_core.runnables.graph import MermaidDrawMethod
import os
import requests

# set chromium revision
LATEST_CHROMIUM = requests.get("https://www.googleapis.com/download/storage/v1/b/chromium-browser-snapshots/o/Win_x64%2FLAST_CHANGE?generation=1734282542160343&alt=media").text
os.environ["PYPPETEER_CHROMIUM_REVISION"] = LATEST_CHROMIUM

# Define the State Schema
class State(TypedDict):
    messages: Annotated[list, add_messages]
    relevant: bool
    input: list
    topic: str
    retry_limit: int
    final_answer: str
    answer_again: bool

def relevancy_check(state: State) -> State:
    topic = state['topic']
    input_list = state['input']
    input_str = ""
    
    for i,inp in enumerate(input_list):
        if i == len(input_list):
            input_str += f'{i+1}. {inp}'
        else:
            input_str += f'{i+1}. {inp}'
            input_str += '\n'
    
    human_message = f"""Are the following elements relevant to the {topic}?
    {input_str}

    Your answer should be Yes or No.
    """
    state['messages'].append(HumanMessage(content = human_message))
                            
    return state

def call_model_relevancy_check(state: State) -> State:
    
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

    try:
        if response.content.lower() == "yes" or response.content.lower() == "yes.": # type: ignore
            state['relevant'] = True
        elif response.content.lower() == "no" or response.content.lower() == "no.": # type: ignore
            state['relevant'] = False
        state['answer_again'] = False
    except:
        try:
            if state['retry_limit']>0:
                state['answer_again'] = True
                state['retry_limit'] -= 1
            else:
                state['answer_again'] = False
        except:
            state['retry_limit'] = int(config.get("LLM_config", "retry_limit"))
            state['retry_limit'] -= 1
            state['answer_again'] = True
    
    return state

def retry_prompt(state: State) -> State:

    human_message = f"""You response was not proper.
    Your answer should be Yes or No.
    """
    state['messages'].append(HumanMessage(content = human_message))

    return state

def establish_relationship(state: State) -> State:
    input_list = state['input']
    input_str = ""
    
    for i,inp in enumerate(input_list):
        if i == len(input_list):
            input_str += f'{i+1}. {inp}'
        else:
            input_str += f'{i+1}. {inp}'
            input_str += '\n'
    
    human_message = f"""How are following elements related?
    {input_str}

    If A, B, C, D, and E are element and there relationship should be mentioned as follows:
    A->B->C
    A->D
    If there are cyclic relationships then mention it as follows:
    A->B->C->A
    Do not print anything else.
    """
    state['messages'].append(HumanMessage(content = human_message))
    
    return state

def call_model_establish_relationship(state: State) -> State:
    
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

    state["final_answer"] = response.content # type: ignore

    return state

def main(input: list) -> None:
    
    # Set up the state graph
    graph = StateGraph(State)

    graph.add_node("relevancy_check", relevancy_check)
    graph.add_node("call_model_relevancy_check", call_model_relevancy_check)
    graph.add_node("retry_prompt", retry_prompt)
    graph.add_node("establish_relationship", establish_relationship)
    graph.add_node("call_model_establish_relationship", call_model_establish_relationship)

    graph.add_edge(START, "relevancy_check")
    graph.add_edge("relevancy_check", "call_model_relevancy_check")
    graph.add_conditional_edges("call_model_relevancy_check", lambda x:x['answer_again'], {
        True: "retry_prompt",
        False: "establish_relationship"
    })
    graph.add_edge("retry_prompt", "call_model_relevancy_check")
    graph.add_edge("establish_relationship", "call_model_establish_relationship")
    graph.add_edge("call_model_establish_relationship", END)

    # Compile the graph
    compiled_graph = graph.compile()

    # Assuming compiled_graph.get_graph().draw_mermaid_png() returns the image data
    image_data = compiled_graph.get_graph().draw_mermaid_png()

    # If you have the image as a file (e.g., saved as PNG), you can display it like this:
    with open("programmatic_chat.png", "wb") as f:
        f.write(image_data)
    
    state = {"messages": []}
    state["input"] = input
    state["topic"] = "water cycle" # type: ignore
    state = compiled_graph.invoke(state)
    print(state['final_answer'])

if __name__ == "__main__":
    main(["Evaporation", "Condensation", "Precipitation", "Runoff", "Infiltration", "Percolation", "Transpiration"])