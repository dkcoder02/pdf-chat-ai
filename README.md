# PDF Chat AI

## Description
This is a simple Streamlit application that allows you to upload a PDF file and chat with an AI assistant about its content. The application processes the PDF, creates embeddings, stores them in a Qdrant vector store, and uses the OpenAI API to answer questions based on the PDF's content.

## Features
- Upload PDF files.
- Process PDF content into searchable chunks.
- Use vector embeddings for semantic search.
- Answer questions about the PDF using an AI model.
- Reference page numbers from the original PDF in the answers.

## Technologies Used
- Python
- Streamlit
- LangChain
- OpenAI API
- Qdrant Vector Database

## Usage

1. Run the Streamlit application from the project root directory:
   ```bash
   streamlit run app.py
   ```

2. Your web browser will open with the application.

3. Upload a PDF file using the file uploader.

4. Once the PDF is processed, a chat interface will appear.

5. Type your questions about the PDF content in the chat input box and press Enter.

6. The AI assistant will provide answers based on the document, citing relevant page numbers.

