import datetime
import os
from openai import OpenAI
import streamlit as st
import threading
from tenacity import retry, wait_random_exponential, stop_after_attempt
from itertools import tee
from dotenv import load_dotenv
load_dotenv()

BASE_URL = os.environ.get("BASE_URL")
API_TOKEN = os.environ.get("API_TOKEN")
QUEUE_SIZE_ENV = os.environ.get("QUEUE_SIZE")
MAX_TOKENS_ENV = os.environ.get("MAX_TOKENS")
RETRY_COUNT_ENV = os.environ.get("RETRY_COUNT")
TOKEN_CHUNK_SIZE_ENV = os.environ.get("TOKEN_CHUNK_SIZE")
MODEL_ID_ENV = os.environ.get("MODEL_ID")

if BASE_URL is None:
    raise ValueError("BASE_URL environment variable must be set")
if API_TOKEN is None:
    raise ValueError("API_TOKEN environment variable must be set")



QUEUE_SIZE = 1
if QUEUE_SIZE_ENV is not None:
    QUEUE_SIZE = int(QUEUE_SIZE_ENV)

RETRY_COUNT = 3
if RETRY_COUNT_ENV is not None:
    RETRY_COUNT = int(RETRY_COUNT_ENV)
    
MAX_TOKENS = 512
if MAX_TOKENS_ENV is not None:
    MAX_TOKENS = int(MAX_TOKENS_ENV)
    
MODEL_ID = MODEL_ID_ENV
if MODEL_ID_ENV is not None:
    MODEL_ID = MODEL_ID_ENV
    
# To prevent streaming to fast, chunk the output into TOKEN_CHUNK_SIZE chunks
TOKEN_CHUNK_SIZE = 1
if TOKEN_CHUNK_SIZE_ENV is not None:
    TOKEN_CHUNK_SIZE = int(TOKEN_CHUNK_SIZE_ENV)

MODEL_AVATAR_URL= "./icon.png"

@st.cache_resource
def get_global_semaphore():
    return threading.BoundedSemaphore(QUEUE_SIZE)
global_semaphore = get_global_semaphore()

MSG_CLIPPED_AT_MAX_OUT_TOKENS = "Reached maximum output tokens for Playground"

EXAMPLE_PROMPTS = [
    "Compose a poem from the perspective of a tree in a bustling city.",
    "List the key differences between classical physics and quantum physics in a bullet point format.",
    "Provide a detailed recipe for making gluten-free chocolate chip cookies.",
    "Write a Python script that scrapes the latest news headlines from a website and prints them.",
    "Create a detailed character profile for a space pirate captain who has a secret love for gardening in json format.",
    "Write a dialogue between Cleopatra and Julius Caesar discussing the future of Egypt."
]

TITLE = "AI Chatbot"
DESCRIPTION="""
This is AI Chatbot.
"""

client = OpenAI(
    api_key=API_TOKEN,
    base_url=BASE_URL
)
GENERAL_ERROR_MSG = "An error occurred. Please refresh the page to start a new conversation."
# st.title(TITLE)
st.markdown(DESCRIPTION)

with open("style.css") as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

def clear_chat_history():
    st.session_state["messages"] = []

st.button('Clear Chat', on_click=clear_chat_history)

def last_role_is_user():
    return len(st.session_state["messages"]) > 0 and st.session_state["messages"][-1]["role"] == "user"
    
def get_system_prompt():
    date_str = datetime.datetime.now().strftime("%B %d, %Y")
    prompt = f"You are Meow, created by Vijay Maurya. The current date is {date_str}.\n"
    return prompt

@retry(wait=wait_random_exponential(min=0.5, max=2), stop=stop_after_attempt(3))
def chat_api_call(history):
    print(f"History :{history}")
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in history
        ],
        model=MODEL_ID,
        stream=True,
        max_tokens=MAX_TOKENS,
        temperature=0.1
    )
    return chat_completion
    

def text_stream(stream):
    for chunk in stream:
        if chunk["content"] is not None:
            yield chunk["content"]

def get_stream_warning_error(stream):
    error = None
    warning = None
    for chunk in stream:
        if chunk["error"] is not None:
            error = chunk["error"]
        if chunk["warning"] is not None:
            warning = chunk["warning"]
    return warning, error

def write_response():
    stream = chat_completion(st.session_state["messages"])
    content_stream, error_stream = tee(stream)
    response = st.write_stream(text_stream(content_stream))
    stream_warning, stream_error = get_stream_warning_error(error_stream)
    if stream_warning is not None:
        st.warning(stream_warning,icon="⚠️")
    if stream_error is not None:
        st.error(stream_error,icon="🚨")
    if isinstance(response, list):
        response = None 
    return response, stream_warning, stream_error
            
def chat_completion(messages):
    history_openai_format = [
        {"role": "system", "content": get_system_prompt()}
    ]
        
    history_openai_format = history_openai_format + messages

    chat_completion = None
    error = None 
    with global_semaphore:
        try: 
            chat_completion = chat_api_call(history_openai_format)
        except Exception as e:
            print("Getting Error in Chatcompletion"+str(e)  )
            error = e    
    if error is not None:
        yield {"content": None, "error": GENERAL_ERROR_MSG, "warning": None}
        return
    
    max_token_warning = None
    partial_message = ""
    chunk_counter = 0
    for chunk in chat_completion:
        try:      
            if chunk.choices[0].delta.  content is not None:
                chunk_counter += 1
                partial_message += chunk.choices[0].delta.content
                if chunk_counter % TOKEN_CHUNK_SIZE == 0:
                    chunk_counter = 0
                    yield {"content": partial_message, "error": None, "warning": None}
                    partial_message = ""
            if chunk.choices[0].finish_reason == "length":
                max_token_warning = MSG_CLIPPED_AT_MAX_OUT_TOKENS
        except Exception as e:
            print("Getting Error in Chatcompletion"+str(e)  )
    yield {"content": partial_message, "error": None, "warning": max_token_warning}


def handle_user_input(user_input):
    with history:
        response, stream_warning, stream_error = [None, None, None]
        if last_role_is_user():
            # retry the assistant if the user tries to send a new message
            with st.chat_message("assistant", avatar=MODEL_AVATAR_URL):
                response, stream_warning, stream_error = write_response()
        else:
            st.session_state["messages"].append({"role": "user", "content": user_input,  "warning": None,"error": None})
            with st.chat_message("user"):
                st.markdown(user_input)
            stream = chat_completion(st.session_state["messages"])
            with st.chat_message("assistant", avatar=MODEL_AVATAR_URL):
                response, stream_warning, stream_error = write_response()
        
        st.session_state["messages"].append({"role": "assistant", "content": response, "warning": stream_warning,"error": stream_error})
    
    
    
main = st.container()

with main:
    history = st.container(height=480,border=True)
    with history:
        for message in st.session_state["messages"]:
            avatar = None
            if message["role"] == "assistant":
                avatar = MODEL_AVATAR_URL
            with st.chat_message(message["role"],avatar=avatar):
                if message["content"] is not None:
                    st.markdown(message["content"])
                if message["error"] is not None:
                    st.error(message["error"],icon="🚨")
                if message["warning"] is not None:
                    st.warning(message["warning"],icon="⚠️")

    if prompt := st.chat_input("Type a message!"):

        handle_user_input(prompt)
    # st.markdown("\n") 

with st.sidebar:
    with st.container():
        st.title("Examples")
        for prompt in EXAMPLE_PROMPTS:
            st.button(prompt, args=(prompt,), on_click=handle_user_input)
