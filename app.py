import streamlit as st
from import_pdf import read_uploaded_file
from embedding_data import fetch_similar_text,clear_collections
from llm_generate_ans import generate_answer
from replicate_api import generate_ans_mistral

query=st.chat_input('Ask a question')

openAI_API_KEY=''
Replicate_api_Key=''           
with st.sidebar:
    st.title("MIRA- Your AI Research Companion")
    openAI_API_KEY=st.text_input('Enter your OpenAI API Key')
    Replicate_api_Key=st.text_input('Enter your Replicate API Key(Optional)')

    uploaded_file = st.file_uploader("Please upload your pdf file to chat with.", type=["pdf"], accept_multiple_files=False)

    if openAI_API_KEY != '':
        if uploaded_file is not None:
            with st.spinner('Please wait while Agent is analyzing the documents.'):
                read_uploaded_file(uploaded_file,openAI_API_KEY)  
        else:
            st.error('Upload a PDF of your research paper')
    else:
        st.error('Enter your OpenAI API key to get started') 
    st.write('Note: Using Mistral 7b requires Replicate API key. [Get it here](https://replicate.com/)')
    agent_type=st.radio("Agent Type",[":green[GPT3.5 Turbo]",":blue[Mistral 7b]"])
    if st.button("New Chat"):
        clear_collections()
        st.cache_data.clear()
        st.success("Cleared Cache")
    st.write('Made with ❤️ by [Sagar Gogineni](https://github.com/Sagar-Gogineni)')    
if query is not None:
    with st.chat_message('user'):
        st.write(query)
    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            if agent_type==":green[GPT3.5 Turbo]":
                print(agent_type)
                st.write(generate_answer(query,fetch_similar_text(query,openAI_API_KEY),openAI_API_KEY))
            elif agent_type==":blue[Mistral 7b]":
                print(agent_type)
                if Replicate_api_Key!='':
                    st.write(generate_ans_mistral(query,fetch_similar_text(query,openAI_API_KEY),Replicate_api_Key))
                else:
                    st.error('Enter Replicate API key to continue')
