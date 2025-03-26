import streamlit as st
from graph import graph
from langchain_core.messages import AIMessage, HumanMessage


# Page configuration
st.set_page_config(layout="wide", page_title="Hair Archetype Chatbot", page_icon="💇")

# Add custom CSS styles
st.markdown(
    """
<style>
    :root {
        --primary-color: #8e44ad;
        --secondary-color: #e6f3ff;
        --accent-color: #e74c3c;
        --text-color: #333;
        --light-gray: #f8f9fa;
        --border-color: #e0e0e0;
    }
    
    .stApp {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: var(--text-color);
        background-color: #e6f3ff !important;
    }
    
    .main .block-container {
        max-width: 1400px;
        margin: 2rem auto;
        background: white;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        padding: 2rem;
    }
    
    .stChatMessage [data-testid="stChatMessageContent"] {
        border-radius: 12px;
        color: black;
        padding: 1rem;
        max-width: 80%;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .stChatMessage.user [data-testid="stChatMessageContent"] {
        background-color: var(--primary-color);
        color: white;
        border-bottom-right-radius: 4px;
        margin-left: auto;
    }
    
    .stChatMessage.assistant [data-testid="stChatMessageContent"] {
        background-color: #ffd700;
        color: #000000;
        border-bottom-left-radius: 4px;
        font-weight: 500;
        border: 1px solid #e6c200;
    }
    
    .sidebar .sidebar-content {
        background-color: var(--secondary-color);
        border-right: 1px solid var(--border-color);
    }
    
    .stButton button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
    }
    
    .stButton button:hover {
        background-color: #7d32a8;
    }

    .header-image {
        width: 100%;
        max-height: 200px;
        object-fit: cover;
        border-radius: 8px;
        margin-bottom: 2rem;
    }

    .style-icon {
        width: 30px;
        height: 30px;
        margin-right: 10px;
        vertical-align: middle;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Page title with icon
st.markdown("# 💇 Hair Archetype Chatbot")

st.subheader("ℹ️ About")
st.markdown(
    "This is an AI assistant specialized in hairstyling, helping you find the perfect hairstyle that suits you."
)

# Initialize session state
if "message_history" not in st.session_state:
    st.session_state.message_history = [
        AIMessage(
            content="""Hello! I'm an AI assistant specialized in hairstyling. I can help you understand different hairstyles and find the perfect one for you.
                   How can I assist you today?"""
        )
    ]

# Create two-column layout
left_col, main_col = st.columns([1, 2])


# Sidebar content
with left_col:
    st.subheader("💫 Chat Options")

    if st.button("Clear Chat History"):
        st.session_state.message_history = [
            AIMessage(
                content="""Hello! I'm an AI assistant specialized in hairstyling. I can help you understand different hairstyles and find the perfect one for you.
                       How can I assist you today?"""
            )
        ]

    st.markdown("---")
    st.subheader("✂️ Hairstyles")
    st.markdown(
        """
    - 👱‍♀️ POLISHED OPULENCE
    - 👩 3D CURLS
    - 👱‍♀️ CELEBRATE MY COILS
    - 👩‍🦱 LUMINOUS CHIC
    - 👰 AIRY COOL SLIM FACE
    - 👩‍🦰 NATURAL SUBLIMATION
    - 👩‍� YOUTH KEEPER

    """
    )

    # Add some example hairstyle images
    st.image(
        "https://images.unsplash.com/photo-1595476108010-b4d1f102b1b1?w=400&auto=format&fit=crop&q=60",
        caption="Classic Styles",
    )


config = {"configurable": {"thread_id": "abc123"}}

# Chat history and input
with main_col:
    st.markdown(
        "<div class='chat-header'><h3>💬 Chat with Hair Consultant</h3></div>",
        unsafe_allow_html=True,
    )

    # Display chat history
    for message in st.session_state.message_history:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="👩‍💼"):
                st.markdown(message.content)
        else:
            with st.chat_message("user", avatar="👤"):
                st.markdown(message.content)

    # User input
    user_input = st.chat_input("Type your question...")

    if user_input:
        # Add user message to history
        st.session_state.message_history.append(HumanMessage(content=user_input))

        # Get AI response
        with st.spinner("Thinking..."):
            response = graph.invoke(
                {"messages": st.session_state.message_history}, config=config
            )

            # Update message history
            st.session_state.message_history = response["messages"]

        # Force page rerender to show new messages
        st.rerun()
