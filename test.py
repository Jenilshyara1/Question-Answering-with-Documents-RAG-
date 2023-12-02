# import time
# from langchain.llms import LlamaCpp
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from query import Query
# import streamlit as st
# import time
# q = Query()

# template = """<s>[INST] <<SYS>>
#         You are an assistant for question-answering tasks.
#         If you don't know the answer, just say that you don't know and return answer only nothing more.<</SYS>>
#         Use the bellow pieces of context to answer the question, if answer is not in context give appropriate answer by your own.
#         [context] : {context}
#         answer the question: {question}
#         Answer:
#         [/INST]"""
# vectorstore = q.main(load_embeddings=True)
# llm = LlamaCpp(
#         model_path=r"C:\Users\jenil\Downloads\llama-2-7b-chat.Q4_K_M.gguf",
#         temperature=0.75,
#         max_tokens=2000,
#         top_p=1,
#         verbose=True,  # Verbose is required to pass to the callback manager
#         n_ctx=2000,
#         n_threads = 12
#                         )
# chain = LLMChain(llm=llm, prompt=PromptTemplate(template=template, input_variables=["documents"]))

# st.title("Chatbot")
# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Accept user input
# if prompt := st.chat_input("What is up?"):
# # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Display assistant response in chat message container
#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         full_response = ""
#         similar_doc = vectorstore.similarity_search(prompt, k=1)
#         context = similar_doc[0].page_content
#         assistant_response = chain.run({"context": context, "question": prompt})
#         # Simulate stream of response with milliseconds delay
#         for chunk in assistant_response.split():
#             full_response += chunk + " "
#             time.sleep(0.05)
#             # Add a blinking cursor to simulate typing
#             message_placeholder.markdown(full_response + "▌")
#         message_placeholder.markdown(full_response)
#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": full_response})

# import  requests
# assistant_response = requests.post('http://127.0.0.1:5000/chat_response',json={'text':"what is cheese making?"})
# print(assistant_response.json())
# print(assistant_response.text)
# print(type(assistant_response.text))


