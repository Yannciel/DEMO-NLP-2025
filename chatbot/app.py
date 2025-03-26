import streamlit as st
from graph import graph
from langchain_core.messages import AIMessage, HumanMessage


st.set_page_config(layout="wide", page_title="Demo Persona Chatbot", page_icon="💐")
st.title("Demo Persona Chatbot")
if "message_history" not in st.session_state:
    st.session_state.message_history = [
        AIMessage(
            content="""Hello, I'm a a highly intelligent and empathetic conversational assistant designed to simulate interactions with individuals who represent specific hair archetypes.
                   How can I help?"""
        )
    ]

left_col, main_col = st.columns([1, 2])

# 1. Buttons for chat - Clear Button

with left_col:
    if st.button("Clear Chat"):
        st.session_state.message_history = []

config = {"configurable": {"thread_id": "abc123"}}

# 2. Chat history and input
with main_col:
    user_input = st.chat_input("Type here...")

    if user_input:
        st.session_state.message_history.append(HumanMessage(content=user_input))

        response = graph.invoke(
            {"messages": st.session_state.message_history}, config=config
        )

        st.session_state.message_history = response["messages"]

    for i in range(1, len(st.session_state.message_history) + 1):
        this_message = st.session_state.message_history[-i]
        if isinstance(this_message, AIMessage):
            message_box = st.chat_message("assistant")
        else:
            message_box = st.chat_message("user")
        message_box.markdown(this_message.content)
