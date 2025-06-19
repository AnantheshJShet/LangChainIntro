import streamlit as st
import os
import psutil
import time
from langchain_ollama.chat_models import ChatOllama
from langchain.schema import (
   AIMessage,
   HumanMessage,
   SystemMessage
)

st.set_page_config(page_title="Langchain Chat Demo", page_icon=":robot:")
st.header("Hey, I'm your Assistant")

chat = ChatOllama(model="llama3.2:1b", temperature=0)

if "sessionMessages" not in st.session_state:
   st.session_state.sessionMessages = [
      SystemMessage(content="You are a helpful assistant.")
   ]

def load_answer(question):
   st.session_state.sessionMessages.append(HumanMessage(content=question))
   #print(st.session_state.sessionMessages)
   assistant_answer = chat.invoke(st.session_state.sessionMessages)
   st.session_state.sessionMessages.append(AIMessage(content=assistant_answer.content))
   return assistant_answer.content

def get_text():
   input_text = st.text_input("You: ", key=input)
   return input_text

user_input = get_text()
submit = st.button("Generate")

if submit:
   response = load_answer(user_input)
   st.subheader("Response:")   
   st.write(response)
   
if st.button("Exit Application"):
    st.warning("Shutting down in 3 seconds...")
    time.sleep(3)
    pid = os.getpid()
    p = psutil.Process(pid)
    p.terminate()