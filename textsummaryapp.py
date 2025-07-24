import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader

#stresmlit
st.set_page_config(page_title="Lanchain:Summarize the url")
st.title("Let's summarize the website content")
st.subheader("Summary")

from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq

with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value="",type="password")

url=st.text_input("URL",label_visibility="collapsed")

## Gemma Model USsing Groq API
llm =ChatGroq(model="Llama3-8b-8192", groq_api_key=groq_api_key)
prompt_template= ''' 
Provide summary of following content in 400 words
content : {text}
'''

prompt = PromptTemplate(template=prompt_template, input_variables=['text'])

if st.button("Summmarize the content"):
    if not groq_api_key.strip() or not url.strip():
        st.error("Plase provide the information")
    elif not validators.url(url):
        st.error("Plase provide the information")
    else:

        try:
            with st.spinner("Waiting...."):
                #laoding the data
                if "youtube.com" in url:
                    loader= YoutubeLoader.from_youtube_url(url, add_video_info=True)
                else:
                    loader= UnstructuredURLLoader(urls=[url], ssl_verify=False, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                data= loader.load()
                chain= load_summarize_chain(llm=llm, chain_type='stuff', prompt=prompt)
                summary=chain.run(data)


                st.success(summary)
        except Exception as e:
            st.exception(f"Exception:{e}")



