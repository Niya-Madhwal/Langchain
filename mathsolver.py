import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, LLMMathChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler

st.set_page_config(page_title="Maths problem solver")
st.title("Maths Solver - Smart and Intelligent")


groq_api_key = st.sidebar.text_input(label="Groq Api Key", type='password')

if not groq_api_key:
    st.info("Please add your groq api key to continue")
    st.stop()

llm= ChatGroq(groq_api_key=groq_api_key, model_name='Gemma2-9b-It')

wikipedia = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name= 'Wikipedia',
    func= wikipedia.run,
    description="A tool which will solve your math problem"
)

maths_chain= LLMMathChain.from_llm(llm=llm)
calculator= Tool(
    name= "Calculator",
    func= maths_chain.run,
    description="A tool which will solve your math problem"
)

prompt = ''' 
You are an agent which will solve users mathematical expression
and display the relevant results with steps
Question :{question}
Answer:
'''

prompt_temp= PromptTemplate(
    input_variables=['question'],
    template=prompt
)

#combine all tools
chain = LLMChain(llm=llm, prompt=prompt_temp)
reasoning = Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for reasoning"
)


assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handling_parsing_errors=True
)

#sessionstate
if "messages" not in st.session_state:
    st.session_state['messages']=[
        {'role':'assistant', 'content':"Hi, I'm a Math solver"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])


question= st.text_area("Enter your question")

if st.button("Find my answer"):
    if question:
        with st.spinner("Generate the reponse"):
            st.session_state.messages.append({'role': "user", "content":question})
            st.chat_message("user").write(question)
            st_cb= StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response= assistant_agent.run(st.session_state.messages, callbacks=[st_cb])

            st.session_state.messages.append({'role':'assistant', 'content': response})
            st.write('##Response:')
            st.success(response)

    else:
        st.warning("Enter the question")

