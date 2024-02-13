# AI_Researcher_MIRA
MIRA - An AI research assistant built with GPT3.5 and Mistral 7b 
Demo @ https://mira-airesearcher.streamlit.app/

In this blog post, I will show you how to build an AI research assistant that can answer your questions based on a PDF file of your choice. You can use this assistant to chat with your research paper, get insights, and learn new things. 🧠

To build this assistant, we will use the following tools and frameworks:

Streamlit: A Python library that lets you create beautiful and interactive web apps for data science and machine learning. Streamlit makes it easy to build and deploy your app with minimal code and no web development skills. 🚀
GPT3.5 Turbo: A powerful language model that can generate natural and coherent text based on a given prompt. GPT3.5 Turbo is a fine-tuned version of GPT-3 that is optimized for question answering and knowledge retrieval. 🚀
Mistral 7B: A state-of-the-art language model that can generate high-quality and diverse text based on a given prompt. Mistral 7B is a custom model that is trained on a large and diverse corpus of text, including scientific papers, news articles, books, and more. 🚀
The basic idea of our app is to use Streamlit to create a user interface where you can upload a PDF file of your research paper and ask questions to the AI agent. The agent will then use GPT3.5 Turbo or Mistral 7B (depending on your choice) to generate an answer based on the content of the PDF file. To do this, we will use the following steps:

Import the necessary modules and libraries.
Create a chat input widget where you can type your questions.
Create a sidebar where you can enter your OpenAI and Replicate API keys, upload your PDF file, and choose the agent type.
Read and process the uploaded PDF file using the import_pdf module.
Fetch the most similar text from the PDF file based on your question using the embedding_data module.
Generate an answer using the llm_generate_ans module for GPT3.5 Turbo or the replicate_api module for Mistral 7B.
Display the answer in a chat message widget.
Let’s go through each step in detail and see the code.

Step 1: Import the necessary modules and libraries
First, we need to import Streamlit and the custom modules that we have created for this app. The custom modules are:

import_pdf: This module contains the function read_uploaded_file that takes an uploaded PDF file and an OpenAI API key as inputs and returns a list of paragraphs from the PDF file. It uses the pdfplumber library to extract the text from the PDF file and the openai library to tokenize and encode the text using GPT-3.
embedding_data: This module contains the functions fetch_similar_text and clear_collections that are used to find the most similar text from the PDF file based on the question. It uses the sentence_transformers library to compute the semantic similarity between the question and the paragraphs using a pre-trained model. It also uses the pymongo library to store and retrieve the encoded text from a MongoDB database.
llm_generate_ans: This module contains the function generate_answer that takes a question, a similar text, and an OpenAI API key as inputs and returns an answer generated by GPT3.5 Turbo. It uses the openai library to call the GPT-3 API with the appropriate parameters and settings.
replicate_api: This module contains the function generate_ans_mistral that takes a question, a similar text, and a Replicate API key as inputs and returns an answer generated by Mistral 7B. It uses the requests library to call the Replicate API with the appropriate parameters and settings.

Step 2: Create a chat input widget where you can type your questions
Next, we need to create a chat input widget where you can type your questions to the AI agent. Streamlit provides a st.chat_input function that creates a chat-like input box where you can enter your text and press enter to send it. We will store the input text in a variable called query. 

Step 3: Create a sidebar where you can enter your OpenAI and Replicate API keys, upload your PDF file, and choose the agent type
Then, we need to create a sidebar where you can enter your OpenAI and Replicate API keys, upload your PDF file, and choose the agent type. Streamlit provides a st.sidebar function that creates a sidebar on the left side of the app where you can add different widgets. We will use the following widgets:

st.title: This function creates a title for the sidebar. We will name our app as “MIRA- Your AI Research Companion”.
st.text_input: This function creates a text input box where you can enter your text. We will use two text input boxes, one for the OpenAI API key and one for the Replicate API key. The OpenAI API key is required to use GPT3.5 Turbo and to process the PDF file. The Replicate API key is optional and is only required to use Mistral 7B. We will store the keys in variables called openAI_API_KEY and Replicate_api_Key.
st.file_uploader: This function creates a file uploader widget where you can upload a file from your computer. We will use this widget to upload a PDF file of your research paper. We will specify the type of the file as “pdf” and the number of files as one. We will store the uploaded file in a variable called uploaded_file.
st.write: This function writes text or data to the app. We will use this function to write a note about using Mistral 7B and a link to get the Replicate API key. We will also use this function to write a credit line for the app creator.
st.radio: This function creates a radio button widget where you can choose one option from a list of options. We will use this widget to choose the agent type, either GPT3.5 Turbo or Mistral 7B. We will store the chosen option in a variable called agent_type.
st.button: This function creates a button widget that can trigger an action when clicked. We will use this widget to create a “New Chat” button that can clear the cache and start a new chat session. We will use the clear_collections function from the embedding_data module and the st.cache_data.clear function from Streamlit to clear the cache.

Step 4: Read and process the uploaded PDF file using the import_pdf module
After uploading the PDF file, we need to read and process it using the import_pdf module. The read_uploaded_file function takes the uploaded file and the OpenAI API key as inputs and returns a list of paragraphs from the PDF file.

Step 5: Fetch the most similar text from the PDF file based on your question using the embedding_data module
Once we have the paragraphs from the PDF file, we need to fetch the most similar text from them based on your question. We will use the fetch_similar_text function from the embedding_data module for finding the top 10 most relevant texts.

Step 6: Generate an answer using the llm_generate_ans module for GPT3.5 Turbo or the replicate_api module for Mistral 7B
After fetching the most similar text from the PDF file, we need to generate an answer using either GPT3.5 Turbo or Mistral 7B, depending on the agent type that you have chosen. We will use the generate_answer function from the llm_generate_ans module for GPT3.5 Turbo and the generate_ans_mistral function from the replicate_api module for Mistral 7B. Both functions take a question, a similar text, and an API key as inputs and return an answer generated by the corresponding language model.

Step 7: Display the answer in a chat message widget
Finally, we need to display the answer in a chat message widget. Streamlit provides a st.chat_message function that creates a chat-like message box where you can display your text. We will use two chat message boxes, one for the user’s question and one for the agent’s answer. We will also use a spinner widget to show a loading animation while the agent is thinking.

And that’s it! We have completed the code for our AI research assistant app. You can now run the app and chat with your research paper using GPT3.5 Turbo or Mistral 7B. You can also upload different PDF files and ask different questions to explore the capabilities of the AI agent. 🙌
