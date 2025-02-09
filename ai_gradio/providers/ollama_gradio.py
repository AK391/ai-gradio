import os
import gradio as gr
from typing import Callable
import ollama
from ollama import Client, ResponseError

def get_fn(model_name: str, preprocess: Callable, postprocess: Callable, **kwargs):
    # Initialize Ollama client
    client = Client(host='http://localhost:11434')
    
    def predict(message, history, temperature=0.7, max_tokens=512):
        # Create a new list for history to avoid sharing between sessions
        history = list(history) if history else []
        
        # Format conversation history
        if isinstance(message, dict):
            message = message["text"]
        
        messages = []
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": message})

        try:
            # Stream the response
            stream = client.chat(
                model=model_name,
                messages=messages,
                stream=True,
                options={
                    "temperature": float(temperature),
                    "num_predict": max_tokens
                }
            )
            
            response_text = ""
            for chunk in stream:
                if chunk and "message" in chunk and "content" in chunk["message"]:
                    delta = chunk["message"]["content"]
                    response_text += delta
                    yield response_text

        except ResponseError as e:
            error_message = f"Error: {str(e)}"
            if e.status_code == 404:
                error_message += f"\nModel '{model_name}' not found. Try running: ollama pull {model_name}"
            yield error_message

    return predict

def get_interface_args(pipeline):
    if pipeline == "chat":
        def preprocess(message, history):
            return {"message": message, "history": history}

        def postprocess(response):
            return response

        return None, None, preprocess, postprocess
    else:
        raise ValueError(f"Unsupported pipeline type: {pipeline}")

def get_pipeline(model_name):
    return "chat"

def registry(name: str = None, **kwargs):
    if name and ':' in name:
        # Extract just the model name after 'ollama:'
        name = name.split(':', 1)[1]
    
    if not name:
        raise ValueError("Model name must be provided")
    
    # Try to pull the model
    try:
        print(f"Ensuring model '{name}' is available...")
        ollama.pull(name)
        print(f"Successfully pulled/verified model '{name}'")
    except Exception as e:
        raise ValueError(f"Error pulling model '{name}': {str(e)}")
    
    pipeline = get_pipeline(name)
    inputs, outputs, preprocess, postprocess = get_interface_args(pipeline)
    fn = get_fn(name, preprocess, postprocess, **kwargs)

    interface = gr.ChatInterface(
        fn=fn,
        additional_inputs=[
            gr.Slider(0, 1, 0.7, label="Temperature"),
            gr.Slider(1, 2048, 512, label="Max tokens"),
        ]
    )
    
    return interface 