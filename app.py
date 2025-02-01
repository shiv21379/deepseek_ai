import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate,
)
import PyPDF2
import os
import subprocess
import tempfile
from io import StringIO
import traceback
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
import uuid
from datetime import datetime

# Custom CSS for enhanced UI
st.markdown(
    """
    <style>
        /* Improved chat history styling */
        .chat-history-item {
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            background-color: #f8f9fa;
        }
        
        .chat-history-item:hover {
            background-color: #e9ecef;
        }
        
        .chat-history-item.active {
            background-color: #007bff !important;
            color: white !important;
        }

        /* Scrollable chat history */
        .scrollable-container {
            max-height: 60vh;
            overflow-y: auto;
            margin: 15px 0;
        }

        /* New chat button styling */
        .new-chat-btn {
            background-color: #007bff !important;
            color: white !important;
            margin-bottom: 15px !important;
        }

        /* Timestamp styling */
        .chat-timestamp {
            font-size: 0.8em;
            color: #6c757d;
            margin-top: 3px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main app
st.title("🧠 DeepSeek AI Coding Assistant")
st.caption("🚀 Your AI-powered coding companion for smarter development.")

# Session state management
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
    initial_chat_id = str(uuid.uuid4())
    st.session_state.current_chat = initial_chat_id
    st.session_state.chat_sessions[initial_chat_id] = {
        "messages": [
            {"role": "ai", "type": "text", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}
        ],
        "title": "New Chat",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "file_processed": False
    }

if "current_chat" not in st.session_state:
    st.session_state.current_chat = list(st.session_state.chat_sessions.keys())[0]

# Sidebar configuration
with st.sidebar:
    st.markdown("### Chat History")
    
    # New Chat button
    if st.button("+ New Chat", key="new_chat", use_container_width=True):
        new_chat_id = str(uuid.uuid4())
        st.session_state.chat_sessions[new_chat_id] = {
            "messages": [
                {"role": "ai", "type": "text", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}
            ],
            "title": "New Chat",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "file_processed": False
        }
        st.session_state.current_chat = new_chat_id
        st.rerun()
    
    # Chat history list
    st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)
    for chat_id in list(st.session_state.chat_sessions.keys())[::-1]:
        chat = st.session_state.chat_sessions[chat_id]
        is_active = chat_id == st.session_state.current_chat
        emoji = "🌐" if is_active else "💬"
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"`{chat['timestamp'][-8:]}`")
        with col2:
            if st.button(
                f"{emoji} {chat['title']}",
                key=f"btn_{chat_id}",
                help=chat['timestamp'],
                use_container_width=True
            ):
                st.session_state.current_chat = chat_id
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    
    # Model configuration
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model", ["deepseek-r1:1.5b", "deepseek-r1:7b"], index=0
    )
    temperature = st.slider("Temperature", 0.1, 1.0, 0.3, 0.1)
    max_tokens = st.number_input("Max Tokens", 50, 1000, 200)
    
    if st.button("Clear All Chats", use_container_width=True):
        st.session_state.chat_sessions = {}
        new_chat_id = str(uuid.uuid4())
        st.session_state.chat_sessions[new_chat_id] = {
            "messages": [
                {"role": "ai", "type": "text", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}
            ],
            "title": "New Chat",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "file_processed": False
        }
        st.session_state.current_chat = new_chat_id
        st.rerun()

# Initialize the chat engine
base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm_engine = ChatOllama(
    model=selected_model,
    base_url=base_url,
    temperature=temperature,
    max_tokens=max_tokens,
)

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant specializing in Python, JavaScript, and other programming languages. "
    "Provide concise, correct solutions with strategic print statements for debugging. Always respond in English. "
    "If the user uploads a file (e.g., Python scripts, text files, or PDFs), analyze its content and provide relevant insights or solutions. "
    "For code-related questions, include detailed explanations, best practices, and formatted code snippets. "
    "When debugging, guide the user step-by-step, explaining each step clearly. "
    "Highlight potential security vulnerabilities in the code and suggest secure alternatives. "
    "Suggest best practices for collaboration, such as using version control systems like Git."
)

# Helper functions
def get_current_chat():
    return st.session_state.chat_sessions.get(st.session_state.current_chat)

def add_user_message(content, message_type="text"):
    current_chat = get_current_chat()
    if current_chat:
        current_chat["messages"].append({"role": "user", "type": message_type, "content": content})
        # Update chat title if first user message
        if len(current_chat["messages"]) == 2:  # First user message after initial AI message
            current_chat["title"] = content[:25] + "..." if len(content) > 25 else content

def add_ai_message(content):
    current_chat = get_current_chat()
    if current_chat:
        current_chat["messages"].append({"role": "ai", "type": "text", "content": content})

def extract_file_content(uploaded_file):
    try:
        if uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        elif uploaded_file.type in ["text/x-python", "application/javascript"]:
            return uploaded_file.read().decode("utf-8")
        else:
            return None
    except Exception as e:
        st.error(f"Error extracting file content: {str(e)}")
        return None

def build_prompt_chain():
    current_chat = get_current_chat()
    prompt_sequence = [system_prompt]
    for msg in current_chat["messages"]:
        if msg["type"] == "text":
            if msg["role"] == "user":
                prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
            elif msg["role"] == "ai":
                prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

def generate_ai_response(prompt_chain):
    try:
        processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
        return processing_pipeline.invoke({})
    except Exception as e:
        st.error(f"Failed to generate response: {str(e)}")
        return "Sorry, I encountered an error while processing your request. Please try again."

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    current_chat = get_current_chat()
    if current_chat:
        for message in current_chat["messages"]:
            if message["role"] == "user":
                st.markdown(
                    f"""
                    <div class="chat-message user">
                        <div class="avatar">👤</div>
                        <div class="content">{message["content"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif message["role"] == "ai":
                st.markdown(
                    f"""
                    <div class="chat-message ai">
                        <div class="avatar">🤖</div>
                        <div class="content">{message["content"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# File uploader and chat input
current_chat = get_current_chat()
ALLOWED_FILE_TYPES = ["png", "jpg", "jpeg", "txt", "pdf", "py", "js"]
uploaded_file = st.file_uploader("Upload an image or file", type=ALLOWED_FILE_TYPES)
user_query = st.chat_input("Type your coding question here...")

# Handle file upload
if uploaded_file and current_chat and not current_chat["file_processed"]:
    if uploaded_file.size > 10 * 1024 * 1024:
        st.warning("File size exceeds 10MB limit.")
    else:
        with st.spinner("Processing uploaded file..."):
            current_chat["file_processed"] = True
            if uploaded_file.type.startswith("image"):
                add_user_message(uploaded_file, message_type="image")
            else:
                file_content = extract_file_content(uploaded_file)
                if file_content:
                    add_user_message(f"File content:\n\n{file_content}")
                else:
                    add_user_message(uploaded_file, message_type="file")
            st.toast("File uploaded successfully!", icon="✅")

# Handle user query
if user_query and current_chat:
    add_user_message(user_query)
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
        add_ai_message(ai_response)
    if uploaded_file:
        current_chat["file_processed"] = False
    st.rerun()