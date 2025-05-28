import configparser
import os
import glob
import requests
import PyPDF2
from typing_extensions import TypedDict
from typing import List
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_chroma import Chroma
import re
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import matplotlib.pyplot as plt
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:  # type: ignore
        return [self.model.encode(d).tolist() for d in documents]

    def embed_query(self, query: str) -> List[float]: # type: ignore
        return self.model.encode([query])[0].tolist()

class State(TypedDict):
    messages: Annotated[list, add_messages]
    embeded: bool
    question: str
    context: List[Document]
    vector_db: dict

def extract_text_from_pdf_and_txt(folder_path):
    try:
        # List all PDF and TXT files in the specified folder
        pdf_and_txt_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf') or f.endswith('.txt')]
        
        # Dictionary to store the text extracted from each file
        extracted_text = {}
        
        for filename in pdf_and_txt_files:
            filepath = os.path.join(folder_path, filename)
            try:
                text = ''
                
                # If the file is a PDF
                if filename.endswith('.pdf'):
                    with open(filepath, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        for page in reader.pages:
                            text += page.extract_text()
                
                # If the file is a TXT file
                elif filename.endswith('.txt'):
                    with open(filepath, 'r', encoding='utf-8') as file:
                        text = file.read()

                # Remove special characters (anything that is not a letter, digit, or space)
                text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

                # Store the extracted text by the file name
                extracted_text[filename] = text

            except Exception as e:
                print(f"Error extracting text from {filename}: {e}")

        # Combine all the extracted text from the files
        all_text = ''.join(extracted_text.values())
        return all_text
    
    except Exception as e:
        print(f"Error processing files: {e}")
        return None
    
def check_db_exists(state: State) -> State:

    config_path = "config.ini"    
    config = configparser.RawConfigParser()
    config.read(config_path)

    persist_directory = config.get("Embedding_model_config", "persist_directory")
    
    # Check if the SQLite database file exists
    sql_path = os.path.join(persist_directory, 'chroma.sqlite3')
    if not os.path.exists(sql_path):
        state['embeded'] = False
        return state

    subdirs = [
        d for d in os.listdir(persist_directory)
        if os.path.isdir(os.path.join(persist_directory, d))
    ]

    bin_files_found = any(
        len(glob.glob(os.path.join(persist_directory, subdir, '*.bin'))) >= 4
        for subdir in subdirs
    )

    if not bin_files_found:
        print("No .bin files found in the vector store.")
        state['embeded'] = False
        return state

    embedding_model = CustomEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Vector store and embeddings are present.")
    persistent_client = chromadb.PersistentClient(path=persist_directory)

    vector_db = Chroma(
    client=persistent_client,
    embedding_function=embedding_model)

    state["vector_db"] = vector_db # type: ignore
    state['embeded'] = True
    return state

    embedding_model = CustomEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Vector store and embeddings are present.")
    persistent_client = chromadb.PersistentClient(path=persist_directory)

    vector_db = Chroma(
    client=persistent_client,
    embedding_function=embedding_model)

    state["vector_db"] = vector_db # type: ignore
    state['embeded'] = True
    return state

def emded_documents(state: State) -> State: 
    
    config_file = 'config.ini'
    config = configparser.RawConfigParser()
    config.read(config_file)
    
    chunk_size = int(config.get("Embedding_model_config", "chunk_size"))
    chunk_overlap = int(config.get("Embedding_model_config", "chunk_overlap"))
    doc_path = 'docs'

    embedding_model = CustomEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    persist_directory = config.get("Embedding_model_config", "PERSIST_DIRECTORY")
    corpus_doc = extract_text_from_pdf_and_txt(doc_path)

    vector_db = Chroma(
    embedding_function=embedding_model,
    persist_directory=persist_directory,  # Where to save data locally, remove if not necessary
    )

    document = Document(
    page_content = corpus_doc, # type: ignore
    # metadata={"source": "tweet"},
    id=1
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)

    doc_splits = text_splitter.split_documents([document])
    
    vector_db.add_documents(doc_splits)
                               
    state["vector_db"] = vector_db # type: ignore
    state["embeded"] = True
    return state

def retrieve(state: State):

    config_path = "config.ini"
    config = configparser.RawConfigParser()
    config.read(config_path)
    
    no_of_chunks = int(config.get("Embedding_model_config", "no_of_chunks"))

    vector_db = state["vector_db"]
    retrieved_docs = vector_db.similarity_search(state["question"], k = no_of_chunks) # type: ignore
    state["context"] = retrieved_docs
    return state

def generate(state: State):
    
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    question = state["question"]

    human_message = f'''
    {question}

    {docs_content}
    '''
    
    state["messages"].append(HumanMessage(content = human_message))

    config_path = "config.ini"
    config = configparser.RawConfigParser()
    config.read(config_path)

    os.environ["GROQ_API_KEY"] = config.get('LLM_config', 'groq_api_key')

    model_name = config.get('LLM_config', 'model_name')
    model_temperature = int(config.get('LLM_config', 'model_temperature'))

    llm = ChatGroq(model= model_name, temperature = model_temperature)

    # Generate a response using the language model
    response = llm.invoke(state["messages"])
    # Update the state with the new messages
    state["messages"].append(response)

    return state

def main():

    # Set up the state graph
    graph = StateGraph(State)

    graph.add_node("check_db_exists", check_db_exists)
    graph.add_node("emded_documents", emded_documents)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.add_edge(START, "check_db_exists")
    graph.add_conditional_edges("check_db_exists", lambda x:x["embeded"],
                                {
                                    True: "retrieve",
                                    False:"emded_documents"
                                })

    graph.add_edge("emded_documents", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    compiled_graph = graph.compile()
    # Assuming compiled_graph.get_graph().draw_mermaid_png() returns the image data
    image_data = compiled_graph.get_graph().draw_mermaid_png()

    # If you have the image as a file (e.g., saved as PNG), you can display it like this:
    with open("RAG.png", "wb") as f:
        f.write(image_data)

    state = {"messages": []}
    state["question"] = "What different aspects of water cycle? I know about Evaporation, Condensation, and Precipitation. Are there more?" # type: ignore

    state = compiled_graph.invoke(state)

    print(state["messages"][-1].content)

if __name__ == "__main__":
    main()
