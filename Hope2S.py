import os
import streamlit as st
from streamlit_chat import message
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
import requests

# Function to download and extract text from PDF from URL
def load_pdf_from_url(pdf_url):
    response = requests.get(pdf_url)
    with open("downloaded_pdf.pdf", "wb") as f:
        f.write(response.content)

    text = ""
    pdf_reader = PdfReader("downloaded_pdf.pdf")
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_text(text)

# Function to generate vector store from text chunks
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return knowledge_base

# Function to perform question answering with Google Generative AI
def rag(vector_db, input_query, google_api_key):
    try:
        template = """You are an AI assistant that assists users by providing detailed and comprehensive answers to their questions by extracting information from the provided context:
        {context}.
        Please provide a thorough and well-explained answer. If you do not find any relevant information from the context for the given question, simply say 'Sorry, I have no idea about that. You can contact Hope To Skill AI Team.'. Do not try to make up an answer.
        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()})

        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5, google_api_key=google_api_key)
        output_parser = StrOutputParser()
        rag_chain = (
            setup_and_retrieval
            | prompt
            | model
            | output_parser
        )
        
        # Invoke the chain and check the type of response
        response = rag_chain.invoke(input_query)
        
        # Add debugging statements to understand the response structure
        st.write(f"Type of response: {type(response)}")
        st.write(f"Response content: {response}")

        # If response is a dictionary, access it accordingly
        if isinstance(response, dict):
            context = response.get('context', 'No context available')
            answer = response.get('answer', 'No answer available')
            st.write(f"Context used for query: {context[:500]}...")  # Display the first 500 characters of context
            st.write(f"Response: {answer}")
            return answer
        else:
            st.write("Unexpected response format.")
            return "Sorry, I have no idea about that. You can contact Hope To Skill AI Team."

    except Exception as ex:
        st.write(f"Exception occurred: {str(ex)}")
        return str(ex)

def main():
    st.set_page_config(page_title="Hope_To_Skill AI Chatbot", page_icon=":robot_face:")

    # Main content area with centered title and subtitle
    st.markdown(
        """
        <style>
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            font-size: 24px;
            color: gray;
        }
        </style>
        <div class="title">Hope To Skill AI Chatbot</div>
        <div class="subtitle">Welcome to Hope To Skill AI Chatbot. How can I help you today?</div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar with Google API Key input
    with st.sidebar:
        st.image("https://yt3.googleusercontent.com/G5iAGza6uApx12jz1CBkuuysjvrbonY1QBM128IbDS6bIH_9FvzniqB_b5XdtwPerQRN9uk1=s900-c-k-c0x00ffffff-no-rj", width=290)
        st.sidebar.subheader("Google API Key")
        user_google_api_key = st.sidebar.text_input("🔑 Enter your Google Gemini API key", type="password", placeholder="Your Google API Key")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    # Use the direct download link for Google Drive PDF
    pdf_url = "https://drive.google.com/uc?export=download&id=17N15I2kbfDXeruSV3TS94ZtjxiGLJmGv"
    default_google_api_key = ""
    
    google_api_key = user_google_api_key if user_google_api_key else default_google_api_key

    # Process the PDF in the background (hidden from user)
    if st.session_state.processComplete is None:
        files_text = load_pdf_from_url(pdf_url)
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.conversation = vectorstore
        st.session_state.processComplete = True
    
    # Display chat history above the input field
    for i, message_data in enumerate(st.session_state.chat_history):
        message(message_data["content"], is_user=message_data["is_user"], key=str(i))

    # Accept user input with Streamlit's chat input widget
    if input_query := st.chat_input("What is your question?"):
        response_text = rag(st.session_state.conversation, input_query, google_api_key)
        st.session_state.chat_history.append({"content": input_query, "is_user": True})
        st.session_state.chat_history.append({"content": response_text, "is_user": False})

    # Display chat history
    response_container = st.container()
    with response_container:
        for i, message_data in enumerate(st.session_state.chat_history):
            message(message_data["content"], is_user=message_data["is_user"], key=str(i + len(st.session_state.chat_history)))

if __name__ == '__main__':
    main()
