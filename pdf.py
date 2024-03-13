from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

import os
import streamlit as st
from streamlit_chat import message
import io


# nvapi_key = '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'
nvapi_key = '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'
os.environ["NVIDIA_API_KEY"] = nvapi_key

st.set_page_config(
    page_title="Nvidia Langchain",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)




def storeDocEmbeds(file):
    reader = PdfReader(file)
    corpus = ''.join([p.extract_text() for p in reader.pages if p.extract_text()])

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(corpus)
    embedding = NVIDIAEmbeddings(model="nvolveqa_40k",model_type='passage')

    vectors = FAISS.from_texts(chunks, embedding)
    return vectors



def conversational_chat(query):
    
    result = st.session_state['qa']({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]


def main():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Creating the chatbot interface
    st.title("Nvidia+Langchain PDF Reader:")

    uploaded_file = st.file_uploader("Choose a file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Processing..."):
            uploaded_file.seek(0)
            file = uploaded_file.read()
            if 'vectors' not in st.session_state and 'qa' not in st.session_state:

                st.session_state['vectors'] = storeDocEmbeds(io.BytesIO(file))
                st.session_state['qa'] = ConversationalRetrievalChain.from_llm(
                    ChatNVIDIA(model="mixtral_8x7b"),
                    retriever=st.session_state['vectors'].as_retriever(),
                    return_source_documents=True
                )

        st.session_state['ready'] = True

    if st.session_state.get('ready', False):
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Welcome! You can now ask any questions"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey!"]

        response_container = st.container()
        container = st.container()
       

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="e.g: Summarize the document", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
               
                output = conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    if i < len(st.session_state['past']):
                        st.markdown(
                            "<div style='background-color: #90caf9; color: black; padding: 10px; border-radius: 5px; width: 70%; float: right; margin: 5px;'>"+ st.session_state["past"][i] +"</div>",
                            unsafe_allow_html=True
                        )
                    # st.markdown(
                    #     "<div style='background-color: #c5e1a5; color: black; padding: 10px; border-radius: 5px; width: 70%; float: left; margin: 5px;'>"+ st.session_state["generated"][i] +"</div>",
                    #     unsafe_allow_html=True
                    # )
                    # message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

if __name__ == "__main__":
    main()