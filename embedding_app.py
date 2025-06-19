import streamlit as st
import os
import psutil
import time
from langchain_ollama import OllamaEmbeddings

st.set_page_config(page_title="Langchain Embedding Demo", page_icon=":robot:")
st.header("Hey, I'm your Embedder")

def get_text():
   input_text = st.text_input("You: ", key=input)
   return input_text

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

user_input = get_text()
submit = st.button("Generate")

if submit:
   response = embeddings.embed_query(user_input)
   st.subheader("Embedding:")   
   st.write(f"Length: {len(response)}")
   response_as_string = ', '.join(map(str, response))
   st.text(response_as_string)
   
if st.button("Exit Application"):
    st.warning("Shutting down in 3 seconds...")
    time.sleep(3)
    pid = os.getpid()
    p = psutil.Process(pid)
    p.terminate()
