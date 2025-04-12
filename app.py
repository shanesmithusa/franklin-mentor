import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
import os

# Load persona and memory files
with open("persona_prompt.txt", "r") as f:
    persona = f.read()

with open("memory_volume.txt", "r") as f:
    raw_text = f.read()

docs = [Document(page_content=raw_text)]
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Embed and build retrieval system
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
db = FAISS.from_documents(chunks, embeddings)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo"),
    retriever=db.as_retriever()
)

# Streamlit UI
st.set_page_config(page_title="Ben Franklin Mentor", layout="centered")
st.title("⚡ Ask Benjamin Franklin")

query = st.text_input("What would you like to ask Ben Franklin?")

if query:
    full_prompt = f"{persona}\n\nQuestion: {query}"
    response = qa_chain.run(full_prompt)
    st.markdown(f"**Franklin:** {response}")

