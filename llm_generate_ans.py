import os
from langchain_openai import ChatOpenAI

def generate_answer(query,similar_text,openAI_API_KEY):

    os.environ["OPENAI_API_KEY"] = openAI_API_KEY
    llm=ChatOpenAI(model_name="gpt-3.5-turbo-1106",
            max_tokens=10)

    llm_ans=llm.invoke(
        "You are an Intelligent AI research assistant. Your task is to generate answers from similar_text to the user query. Here is the user query: "+query+ ". Here is the similar_text: "+ str(similar_text)
    )
    return llm_ans.content
