def main():
    st.set_page_config(page_title="Hope_To_Skill AI Chatbot", page_icon=":robot_face:")
    
    # CSS styling for black outline
    st.markdown(
        """
        <style>
        .header-container {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
        }
        .logo {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            overflow: hidden;
            margin-right: 15px;
        }
        .logo img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
        }
        input[type="text"] {
            border: 2px solid black;
            border-radius: 5px;
        }
        input[type="password"] {
            border: 2px solid black;
            border-radius: 5px;
        }
        </style>
        <div class="header-container">
            <div class="logo">
                <img src="https://yt3.googleusercontent.com/G5iAGza6uApx12jz1CBkuuysjvrbonY1QBM128IbDS6bIH_9FvzniqB_b5XdtwPerQRN9uk1=s900-c-k-c0x00ffffff-no-rj" alt="Logo">
            </div>
            <div class="title">
                Hope To Skill AI-Chatbot
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Hello, How can I help you today?:")

    # Search bar with black outline
    input_query = st.text_input("🔍Type your question here...")

    # Sidebar with black outline for API Key input
    st.sidebar.subheader("Google API Key")
    user_google_api_key = st.sidebar.text_input("🔑Enter your Google Gemini API key to Ask Questions", type="password")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    # Use the direct download link for Google Drive PDF
    pdf_url = "https://drive.google.com/uc?export=download&id=1C7I5Y7PJcIPzjH_4T_PxfMdEw13_vz6a"
    default_google_api_key = ""
    
    google_api_key = user_google_api_key if user_google_api_key else default_google_api_key

    # Process the PDF in the background (hidden from user)
    if st.session_state.processComplete is None:
        files_text = load_pdf_from_url(pdf_url)
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.conversation = vectorstore
        st.session_state.processComplete = True

    # Chatbot functionality
    if input_query:
        response_text = rag(st.session_state.conversation, input_query, google_api_key)
        st.session_state.chat_history.append({"content": input_query, "is_user": True})
        st.session_state.chat_history.append({"content": response_text, "is_user": False})

    # Display chat history
    response_container = st.container()
    with response_container:
        for i, message_data in enumerate(st.session_state.chat_history):
            message(message_data["content"], is_user=message_data["is_user"], key=str(i))
