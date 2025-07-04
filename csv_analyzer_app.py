import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama import ChatOllama

st.set_page_config(page_title="CSV Analyzer App", layout="wide")
st.title("📊 CSV Analyzer")

st.write("""
Upload a CSV file and ask questions about its contents. Powered by LangChain and Ollama LLMs.
""")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
df = None

if uploaded_file is not None:
   df = pd.read_csv(uploaded_file)
   st.write("### Preview of Uploaded CSV:")
   st.dataframe(df.head(20))

   # Set up LangChain agent
   llm = ChatOllama(
      model="mistral:7b",
      temperature=0.0,
   )
     
   agent = create_pandas_dataframe_agent(
      llm, 
      df, 
      verbose=True, 
      allow_dangerous_code=True,
      prefix="""You are a data analysis assistant. Only output either a tool action or a final answer, never both. E.g. outut format - "Final Answer: ...". Do not include extra thoughts or explanations unless asked. If you need to use Python, use the provided tools."""
   )

   st.write("---")
   st.write("### Ask a question about your CSV:")
   user_query = st.text_input("Enter your question:")

   if user_query:
      with st.spinner("Analyzing with LLM agent..."):
         try:
               response = agent.invoke({"input": user_query})
               st.success("Answer:")
               st.write(response["output"])
         except Exception as e:
               st.error(f"Error: {e}")
else:
   st.info("Please upload a CSV file to begin.")
