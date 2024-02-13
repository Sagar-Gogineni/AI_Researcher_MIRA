import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embedding_data import generate_embedding

@st.cache_data
def read_uploaded_file(uploaded_file,openAI_API_KEY):
    #st.write("entered the function")
    text_data = []
    chunks=[]
    table_data=[]
    source=[]
    metadata=[]
    with pdfplumber.open(uploaded_file) as pdf:
        pages=pdf.pages
        for i,page in enumerate(pages):
            text_data.append(page.extract_text())
            table_data.append(page.extract_tables())
            bold_chars = page.filter(lambda obj: obj['object_type'] == 'char' and 'Bold' in obj['fontname'])
            # Extract the text from the bold characters and append it to the list
           
            if bold_chars.extract_text() !='':
                source.append(bold_chars.extract_text())
                prev=bold_chars.extract_text()
            else:
                source.append(prev)
            metadata.append({'source':source[i],'pg_no':i})
        print(metadata)
        for text in text_data:
            chunks.append(initialize_chunking(text))
    #print("finished chunking")
    generate_embedding(chunks,metadata,openAI_API_KEY)
    #return ("chunks embedded successfully")     

@st.cache_data
def initialize_chunking(data):
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
    )
    texts = text_splitter.create_documents([data])
    return(texts)
       