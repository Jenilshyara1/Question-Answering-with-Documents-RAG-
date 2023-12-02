import streamlit as st
import time
import requests
from src.query import Query
query = Query()

def query_search(db,prompt):
    similar_doc = db.similarity_search(prompt, k=1)
    context = similar_doc[0].page_content
    return context
st.title("Chatbot")
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
uploaded_file = st.sidebar.file_uploader("Text file")
# if data is not None:
#     print(data.read().decode())
# Accept user input

if uploaded_file is not None:
    db = query.create_embeddings(uploaded_file)
    if prompt := st.chat_input("What is up?"):
        context = query_search(db,prompt)
        print(context)
    # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = requests.post('http://127.0.0.1:5000/chat_response',json={'text':prompt,'context':context}).json()['response']
            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
