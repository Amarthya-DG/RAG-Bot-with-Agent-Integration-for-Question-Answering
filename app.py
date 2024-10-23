import streamlit as st
import bs4
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
import os
from langchain.agents import load_tools, create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate


os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
os.environ["SERPAPI_API_KEY"] =os.getenv('SERPAPI_API_KEY')


llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro')


loader = WebBaseLoader(
    web_paths=("https://www.mdpi.com/1424-8247/17/2/228",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header", "html-p")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = FAISS.from_documents(documents=splits, embedding=GoogleGenerativeAIEmbeddings(model='models/embedding-001'))

retriever = vectorstore.as_retriever()

def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know "i dont know", don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.


{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)


tools = load_tools(['serpapi', 'wikipedia'])
prompt1 = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt1)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def rag_model(user_question):
    result = rag_chain.invoke(user_question)


    if not result or "i dont know" in result:
        agent_result = agent_executor.invoke({"input": user_question})
        return agent_result
    else:
        return result


st.title("RAG Model Question Answering")
user_question = st.text_input("Ask me any question:")

if st.button("Get Answer"):
    if user_question:
        answer = rag_model(user_question)
        st.write(answer)
    else:
        st.write("Please enter a question.")
