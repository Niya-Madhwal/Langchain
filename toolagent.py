import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os 
from dotenv import load_dotenv

arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=20 )
wiki = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

search = DuckDuckGoSearchRun(name= "Search")

st.title("Langchain Chat Search")
"""
You can now use Microsoft Copilot in Intune to explore your Intune data using natural language, 
ake action on the results, manage policies and settings, understand your security posture, 
troubleshoot device issues, and view insights about enrolled Surface devices.
"""

st.sidebar.title("Add your Api key: Setings")
api_key = st.sidebar.text_input("Enter Api Groq", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role": "assistant", "content":"Hi I am chatbot"}

    ]
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt:= st.chat_input(placeholder="Whats is Ms Intune"):
    st.session_state.messages.append({"role":"user", "content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key= api_key, model= "Llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wiki]

    search_agent= initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors= True)

    with st.chat_message("assistant"):
        st_cb =StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant", "content":response})
        st.write(response)
        

