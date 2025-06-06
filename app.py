import streamlit as st
from pathlib import Path
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
import os

load_dotenv()

client = OpenAI()

st.title("Chat with your PDF")
st.write("Upload a PDF file and ask questions about its content.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Process the uploaded file
if uploaded_file is not None:
    if (
        "vector_store" not in st.session_state
        or st.session_state.uploaded_file_name != uploaded_file.name
    ):
        st.session_state.uploaded_file_name = uploaded_file.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Loading
        loader = PyPDFLoader(file_path=tmp_file_path)
        docs = loader.load()

        # Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )

        split_docs = text_splitter.split_documents(documents=docs)

        # Vector Embeddings
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

        # Create or load vector store
        vector_store = QdrantVectorStore.from_documents(
            url=os.getenv("QDRANT_URL"),
            documents=split_docs,
            prefer_grpc=True,
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name="uploaded_pdf_vectors",
            embedding=embedding_model,
        )

        st.session_state.vector_store = vector_store
        st.success("PDF processed and ready to chat!")

        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get vector store from session state
        vector_store = st.session_state.vector_store

        # Vector Similarity Search
        search_results = vector_store.similarity_search(query=prompt)

        context = "\n\n\n".join(
            [
                f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}"
                for result in search_results
            ]
        )

        SYSTEM_PROMPT = f"""
            You are a helpfull AI Assistant who answers user query based on the available context
            retrieved from a PDF file along with page_contents and page number.

            You should only answer the user based on the following context and navigate the user
            to open the right page number to know more.

            Context: {context}
        """

        # Call OpenAI LLM
        try:
            chat_completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            response = chat_completion.choices[0].message.content

            with st.chat_message("assistant"):
                st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"An error occurred while processing your question: {e}")

else:
    if "vector_store" in st.session_state:
        del st.session_state.vector_store
    if "messages" in st.session_state:
        del st.session_state.messages
    if "uploaded_file_name" in st.session_state:
        del st.session_state.uploaded_file_name
