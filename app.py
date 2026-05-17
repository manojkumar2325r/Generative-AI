import streamlit as st
import requests

st.set_page_config(page_title="Cricket Assistant", layout="wide")

st.title("🏏 Cricket Knowledge Assistant")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
user_input = st.chat_input("Ask about cricket...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Call backend API
    response = requests.post(
    "http://127.0.0.1:8000/ask",
    json={"question": user_input}
    )
    data = response.json()
    answer = data["answer"]

    # Show bot response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)