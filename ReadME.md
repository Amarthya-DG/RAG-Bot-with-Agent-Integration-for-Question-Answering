## RAG Bot with Agent Integration for Question Answering

 This is a Streamlit web application that uses a Retrieval-Augmented Generation (RAG) approach to answer user questions by retrieving relevant documents from the web and generating concise answers. The application integrates the Google Generative AI (Gemini-1.5-pro) for text generation, FAISS for vector storage, and SerpAPI and Wikipedia as additional search tools for answering user queries.

## Features
Web Data Retrieval: Extracts and processes web content using BeautifulSoup.
Document Splitting and Vectorization: Utilizes FAISS to store vectorized document chunks.
RAG Pipeline: Combines retrieved information with language models to answer questions accurately.
Fallback Search Tools: Utilizes SerpAPI and Wikipedia for additional query handling if the RAG model cannot find a satisfactory answer.
Streamlit User Interface: Allows users to input questions and receive answers in an easy-to-use web app.

## Requirements
To run this project, you will need:

Python 3.9+
The following Python libraries:
bash
Copy code
streamlit
bs4
langchain
langchain_community
langchain_core
langchain_google_genai
faiss-cpu
You can install these dependencies using pip:


pip install streamlit bs4 langchain langchain_community langchain_core langchain_google_genai faiss-cpu

## API Keys
Make sure to set up the following environment variables:

GOOGLE_API_KEY: API key for Google Generative AI (Gemini-1.5).
SERPAPI_API_KEY: API key for SerpAPI.
You can add them to your environment by creating a .env file or setting them directly in your system.

export GOOGLE_API_KEY="your-google-api-key"
export SERPAPI_API_KEY="your-serpapi-api-key"

## How to Run
Clone the repository:
bash
Copy code
git clone My Respository
cd rag-model-question-answering
Install dependencies:

pip install -r requirements.txt
Run the Streamlit application:
streamlit run app.py
Open the provided local URL in your browser (e.g., http://localhost:8501)
