import gradio as gr
from llm_api import APIHandler

model = APIHandler()

def chat(message, history, temperature, do_sample, max_tokens):
    chat = []
    # add system prompt
    chat.append({"role": "system", "content": "you are helpful AI Assitance."})
    for item in history:
        chat.append({"role": "user", "content": item[0]})
        if item[1] is not None:
            chat.append({"role": "assistant", "content": item[1]})
    chat.append({"role": "user", "content": message})

    generate_kwargs = {
            "messages": chat,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

    completion = model.call_api(**generate_kwargs)

    partial_text = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            # print(chunk.choices[0].delta.content, end="", flush=True)
            new_text = chunk.choices[0].delta.content
            partial_text += new_text
        yield partial_text
    yield partial_text



demo = gr.ChatInterface(
    fn=chat,
    examples=[["Write me a poem about Machine Learning."],["what is capital of india"]],
    additional_inputs_accordion=gr.Accordion(
        label="⚙️ Parameters", open=False, render=False
    ),
    additional_inputs=[
        gr.Slider(
            minimum=0, maximum=1, step=0.1, value=0.9, label="Temperature", render=False
        ),
        gr.Slider(
            minimum=128,
            maximum=20000,
            step=1,
            value=512,
            label="Max new tokens",
            render=False,
        ),
    ],
    title="Chat With LLMs",
    description="This is chatbot using LLMs.",
)
demo.launch()