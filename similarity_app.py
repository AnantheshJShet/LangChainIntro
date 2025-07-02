import streamlit as st
import os
import psutil
import time
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores.faiss import FAISS

st.set_page_config(page_title="Similarity Demo", page_icon=":robot:")
st.header("Hey, provide me a game name & I will give you similar games")

games = CSVLoader(
   file_path='./data/games.csv',
   csv_args={
   'delimiter': ',',
   'quotechar': '"',
   }).load()

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

db = FAISS.from_documents(games, embeddings)

def get_text():
   input_text = st.text_input("You: ", key=input)
   return input_text

user_input = get_text()
submit = st.button("Find similar games")

if submit:
   response = db.similarity_search(user_input)
   st.subheader("Top 3 matches:")
   st.text(response[0].page_content)
   st.text(response[1].page_content)
   st.text(response[2].page_content)
   
if st.button("Exit Application"):
    st.warning("Shutting down in 3 seconds...")
    time.sleep(3)
    pid = os.getpid()
    p = psutil.Process(pid)
    p.terminate()
