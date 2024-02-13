import replicate
import os
import streamlit as st
def generate_ans_mistral(query,similar_text,Replicate_api_Key):
    os.environ["REPLICATE_API_TOKEN"] = Replicate_api_Key
    # The mistralai/mistral-7b-v0.1 model can stream output as it's running.
    string=""
    for event in replicate.stream(
        "mistralai/mistral-7b-instruct-v0.2",
        input={
            "debug": False,
            "top_k": -1,
            "top_p": 0.95,
            "prompt": "You are an Intelligent AI research assistant. Your task is to generate answers from similar_text to the user query. Here is the user query: "+query+ ". Here is the similar_text: "+ str(similar_text),
            "temperature": 0.7,
            "max_new_tokens": 200,
            "min_new_tokens": -1,
            "prompt_template": "<s>[INST] {prompt} [/INST]",
            "repetition_penalty": 1.15
        },
    ):
        #st.write_stream(event)
        string=string+str(event)
        print(string)
    return string