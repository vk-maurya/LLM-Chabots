import os
from typing import Any
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()
BASE_URL = os.environ.get("BASE_URL")
API_KEY= os.environ.get("API_KEY")

class APIHandler:
    def __init__(self, **kwargs: Any):
        self.API_KEY = os.getenv("API_KEY")
        self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        self.kwargs = kwargs

    def call_api(self, messages, model_name=None, max_tokens=512, temperature=0.5, stream=True):
        data = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        completion = self.client.chat.completions.create(**data)
        return completion
        # if stream:    
        #     for chunk in completion:
        #         if chunk.choices[0].delta.content:
        #             print(chunk.choices[0].delta.content, end="", flush=True)
                    # new_message["content"] += chunk.choices[0].delta.content
            # return new_message
        # else:
        #     new_message["content"] = completion.choices[0].message.content
        #     return new_message

if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "you are AI Assitance."},
        {"role": "user", "content": "Introduce yourself."},
    ]
    api_handler = APIHandler()
    completion = api_handler.call_api(messages, temperature=0.7, stream=True)
    for chunk in completion:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)