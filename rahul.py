import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Page title
st.set_page_config(page_title='PragyanAI-Text Summarization App')

# Display logo and title
st.image("PragyanAI_Transperent.png")  # Make sure this file exists in your root repository
st.divider()
st.title('🦜🔗 Text Summarization App')
st.divider()

def generate_response(txt):
    # Instantiate the LLM model using Groq secrets
    llm = ChatGroq(
        model_name="llama3-8b-8192", 
        temperature=0, 
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    
    # Create Document objects
    docs = [Document(page_content=t) for t in texts]
    
    # Summarize documents
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    
    # Run the chain and safely extract the generated text
    res = chain.invoke(docs)
    return res.get("output_text", res)

# Text input
txt_input = st.text_area('Enter your text', '', height=200)

# Form for user input
with st.form('summarize_form', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted and txt_input.strip():
        with st.spinner('Calculating...'):
            response = generate_response(txt_input)
            st.info(response)
    elif submitted and not txt_input.strip():
        st.warning('Please enter text to summarize.')
