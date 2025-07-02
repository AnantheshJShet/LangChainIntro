import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_ollama.chat_models import ChatOllama
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime

llm = ChatOllama(model="llama3.2:1b", temperature=0)

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State) -> State:
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

def show_graph_popup(graph):
    g = graph.get_graph()
    nx_graph = nx.DiGraph()
    # Add nodes
    for node in g.nodes:
        nx_graph.add_node(node)
    # Add edges (only use source and target)
    for edge in g.edges:
        nx_graph.add_edge(edge[0], edge[1])
    # Draw the graph
    pos = nx.spring_layout(nx_graph)
    plt.figure(figsize=(8, 6))
    nx.draw(nx_graph, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, font_weight='bold', edge_color='gray', arrows=True)
    plt.title("LangGraph StateGraph Visualization")
    plt.show()

# Usage after building your graph:
#show_graph_popup(graph)

if "summary" not in st.session_state:
    st.session_state.summary = ""

def stream_graph_updates():
    # Use the summary and the latest user message as context
    summary = st.session_state.summary
    latest_user_msg = st.session_state.chat_history[-1]["content"]
    prompt = f"Conversation so far (summary): {summary}\nUser: {latest_user_msg}"
    for event in graph.stream({"messages": [{"role": "user", "content": prompt}]}):
        for value in event.values():
            response = value["messages"][-1].content
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            return response

def summarize_history(messages):
    # Use the LLM to summarize the chat history
    history = "\n".join([
        f"User: {msg['content']}" if msg['role'] == 'user' else f"Assistant: {msg['content']}"
        for msg in messages
    ])
    prompt = f"Summarize the following conversation between a user and an assistant:\n{history}"
    summary = llm.invoke(prompt)
    return summary.content if hasattr(summary, 'content') else str(summary)

st.set_page_config(page_title="LangGraph Chatbot", page_icon=":speech_balloon:")
st.title("How can I help you today?")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history at the top using Streamlit's chat_message API (no markdown)
for msg in st.session_state.chat_history:
    timestamp = msg.get("timestamp", "")
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(f"{msg['content']}  [{timestamp}]")
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.write(f"{msg['content']}  [{timestamp}]")

# Chat input form
with st.form(key="my_form", clear_on_submit=True):
    user_input = st.text_area("Your question goes here", key="input", height=100)
    submit_button = st.form_submit_button(label="Send")
    if submit_button and user_input:
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        assistant_reply = stream_graph_updates()
        # Update the summary after the new assistant reply
        st.session_state.summary = summarize_history(st.session_state.chat_history)
        st.rerun()

# Button to summarize chat history
if st.button("Summarize Chat History"):
    if st.session_state.chat_history:
        summary = summarize_history(st.session_state.chat_history)
        st.markdown("---")
        st.subheader(":memo: Chat Summary")
        st.info(summary)
    else:
        st.info("No chat history to summarize yet.")

# Add a chat exit button
if st.button("Exit Chat"):
    st.success("Goodbye! Thanks for chatting.")
    st.session_state.chat_history = []
    st.stop()