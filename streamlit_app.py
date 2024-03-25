import streamlit as st
import time
import requests

st.title("Chatbot")
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
uploaded_file = st.sidebar.file_uploader("Text file")
if uploaded_file is not None:
    try:
        progress_text = "Loading embeddings.."
        with st.spinner(progress_text):
            emb = requests.post('http://flask-app:5858/create_embeddings',json={'text':uploaded_file.read().decode()}).json()['create_embeddings']
    except Exception as e:
        st.error(f'error occured while create embeddings call: {e}', icon="🚨")

    if prompt := st.chat_input("What is up?"):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                with st.spinner(''):
                    assistant_response = requests.post('http://flask-app:5858/chat_response',json={'text':prompt}).json()['response']
                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f'error occured: {e}', icon="🚨")
                print(e)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})