import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load Franklin's persona description
with open("persona_prompt.txt", "r") as f:
    persona = f.read()

# Load Franklin's memory base
loader = TextLoader("memory_volume.txt")
docs = loader.load()

# Chunk and embed the memory
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_documents(docs)
db = FAISS.from_documents(chunks, OpenAIEmbeddings())

# Set up the question-answering chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo"),
    retriever=db.as_retriever()
)

# Streamlit interface
st.set_page_config(page_title="Ben Franklin Mentor", layout="centered")
st.title("⚡ Ask Benjamin Franklin")

query = st.text_input("What would you like to ask Ben Franklin?")

if query:
    full_prompt = f"{persona}\n\nQuestion: {query}"
    response = qa_chain.run(full_prompt)
    st.markdown(f"**Franklin:** {response}")


