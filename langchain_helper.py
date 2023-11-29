from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import os

#loading .env
load_dotenv()

# Loading llama2 from Ollama
model = Ollama(model="llama2", temperature=0)

#embedding
# embeddings = HuggingFaceEmbeddings()
from langchain.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


vector_db_filepath = "faiss_index"
#Change According to ur path
directory = '/Desktop/langchain/confluence/Respdf/documents/'

def load_docs():
  loader = DirectoryLoader(directory,glob="**/*.pdf",loader_cls=PyPDFLoader, use_multithreading=True)
  documents = loader.load()
  return documents



def split_docs(documents,chunk_size=500,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs


def create_vector_db():
    documents = load_docs()
    docs = split_docs(documents)
    vectordb = FAISS.from_documents(documents=docs,embedding=embeddings)
    vectordb.save_local(vector_db_filepath) 

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


def query_refiner(conversation_str,query):
    template =""" Given the following user query and conversation log, formulate a question that would be the most relevant to 
    provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"""
    prompt = ChatPromptTemplate.from_template(template) 
    chain =  prompt | model | StrOutputParser()
    response = chain.invoke({"conversation":conversation_str,"query":query})
    return response

def get_qa_chain():
    vectordb = FAISS.load_local(vector_db_filepath,embeddings)
    retriever= vectordb.as_retriever()
    
    template = """Answer the question based only on the following context:

{context}

Question: {question}
if question is not related to context answer as Sorry its not related to Subham
"""

    prompt = ChatPromptTemplate.from_template(template) 
    chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
    return chain



if __name__=="__main__":
    #to fill the vector space
    create_vector_db()
    chain = get_qa_chain()
    print(chain.invoke("Does Subham has ML skills"))














