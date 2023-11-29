from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from langchain_helper import *
from langchain.llms import Ollama

st.subheader("Subham's Q&A Bot 🙎‍♂️")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)

vectordb = FAISS.load_local(vector_db_filepath,embeddings)

system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question based only on the following context, 
and if question is not related to context answer as Sorry its not related to Subham or 'I don't know'""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

response_container = st.container()
textcontainer = st.container()

# Loading llama2 from Ollama
model = Ollama(model="llama2", temperature=0)


def get_conversation_string():
        conversation_string = ""
        for i in range(len(st.session_state['responses'])-1):        
            conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
            conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
        return conversation_string


conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=model, verbose=True)

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            context = vectordb.similarity_search(query)

            conversation_str = get_conversation_string()
            refined_query = query_refiner(conversation_str,query)
            st.subheader("Refined Query")
            st.write(refined_query)
            context = vectordb.similarity_search(refined_query)
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')












