import streamlit as st
import requests

API_BASE = "http://api:5858"

st.title("Chatbot")

# --- Session state init ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []  # list of {doc_id, filename}
if "active_doc_id" not in st.session_state:
    st.session_state.active_doc_id = None  # None = search all

# --- Sidebar: file upload ---
with st.sidebar:
    st.header("Documents")
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf"])

    if uploaded_file is not None:
        already_uploaded = any(
            d["filename"] == uploaded_file.name for d in st.session_state.uploaded_docs
        )
        if not already_uploaded:
            try:
                with st.spinner("Processing document..."):
                    if uploaded_file.name.endswith(".pdf"):
                        resp = requests.post(
                            f"{API_BASE}/upload_pdf",
                            files={"file": (uploaded_file.name, uploaded_file.read(), "application/pdf")},
                        )
                    else:
                        resp = requests.post(
                            f"{API_BASE}/create_embeddings",
                            json={"text": uploaded_file.read().decode(), "filename": uploaded_file.name},
                        )
                    resp.raise_for_status()
                    data = resp.json()
                doc_id = data["doc_id"]
                st.session_state.uploaded_docs.append(
                    {"doc_id": doc_id, "filename": uploaded_file.name}
                )
                st.success(f"Uploaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error creating embeddings: {e}", icon="🚨")

    if st.session_state.uploaded_docs:
        st.divider()
        options = ["All documents"] + [d["filename"] for d in st.session_state.uploaded_docs]
        selected = st.radio("Search in", options)
        if selected == "All documents":
            st.session_state.active_doc_id = None
        else:
            for d in st.session_state.uploaded_docs:
                if d["filename"] == selected:
                    st.session_state.active_doc_id = d["doc_id"]
                    break

# --- Chat area ---
if not st.session_state.uploaded_docs:
    st.info("Upload a document in the sidebar to get started.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            with requests.post(
                f"{API_BASE}/chat_response_stream",
                json={"text": prompt, "doc_id": st.session_state.active_doc_id},
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"Error: {e}", icon="🚨")

    st.session_state.messages.append({"role": "assistant", "content": full_response})
