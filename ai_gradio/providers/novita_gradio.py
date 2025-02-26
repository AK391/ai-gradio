import os
import base64
from typing import Callable
import gradio as gr
import requests
import json

__version__ = "0.0.1"

# Base URL for Novita AI API
NOVITA_API_BASE_URL = "https://api.novita.ai/v3/openai"

SYSTEM_PROMPTS = {
    "default": "You are a helpful, harmless, and honest AI assistant.",
    "coder": """You are an expert web developer specializing in creating clean, efficient, and modern web applications.
Your task is to write complete, self-contained HTML files that include all necessary CSS and JavaScript.
Focus on:
- Writing clear, maintainable code
- Following best practices
- Creating responsive designs
- Adding appropriate styling and interactivity
Return only the complete HTML code without any additional explanation."""
}

def get_image_base64(url: str, ext: str):
    with open(url, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return "data:image/" + ext + ";base64," + encoded_string

def get_fn(model_name: str, preprocess: Callable, postprocess: Callable, api_key: str, system_prompt: str = None):
    def fn(message, history):
        inputs = preprocess(message, history, system_prompt)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        try:
            response = requests.post(
                f"{NOVITA_API_BASE_URL}/chat/completions",
                headers=headers,
                json={
                    "model": model_name,
                    "messages": inputs["messages"],
                    "stream": True,
                    "max_tokens": 1000
                },
                stream=True
            )
            
            response.raise_for_status()
            
            partial_message = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data:") and line != "data: [DONE]":
                        json_str = line[5:].strip()
                        try:
                            chunk = json.loads(json_str)
                            if chunk["choices"]:
                                delta = chunk["choices"][0]["delta"].get("content", "")
                                partial_message += delta
                                yield postprocess(partial_message)
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            yield error_message

    return fn

def handle_user_msg(message: str):
    if type(message) is str:
        return message
    elif type(message) is dict:
        if message["files"] is not None and len(message["files"]) > 0:
            ext = os.path.splitext(message["files"][-1])[1].strip(".")
            if ext.lower() in ["png", "jpg", "jpeg", "gif"]:
                encoded_str = get_image_base64(message["files"][-1], ext)
            else:
                raise NotImplementedError(f"Not supported file type {ext}")
            content = [
                    {"type": "text", "text": message["text"]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": encoded_str,
                        }
                    },
                ]
        else:
            content = message["text"]
        return content
    else:
        raise NotImplementedError

def get_interface_args(pipeline, system_prompt=None):
    if pipeline == "chat":
        inputs = None
        outputs = None

        def preprocess(message, history, system_prompt=system_prompt):           
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
                
            # Process history first
            for user_msg, assistant_msg in history:
                messages.append({"role": "user", "content": str(user_msg)})
                if assistant_msg is not None:
                    messages.append({"role": "assistant", "content": str(assistant_msg)})
            
            # Add current message
            messages.append({"role": "user", "content": str(message)})
            return {"messages": messages}

        postprocess = lambda x: x  # No post-processing needed
    else:
        raise ValueError(f"Unsupported pipeline type: {pipeline}")
    return inputs, outputs, preprocess, postprocess

def get_pipeline(model_name):
    # Determine the pipeline type based on the model name
    # For simplicity, assuming all models are chat models at the moment
    return "chat"

def registry(name: str, token: str | None = None, coder: bool = False, **kwargs):
    """
    Create a Gradio Interface for a model on Novita AI.

    Parameters:
        - name (str): The name of the model on Novita AI.
        - token (str, optional): The API key for Novita AI.
        - coder (bool, optional): Whether to use the coder system prompt. Defaults to False.
    """
    api_key = token or os.environ.get("NOVITA_API_KEY")
    if not api_key:
        raise ValueError("NOVITA_API_KEY environment variable is not set.")

    # Select system prompt based on coder parameter
    system_prompt = SYSTEM_PROMPTS["coder"] if coder else SYSTEM_PROMPTS["default"]

    pipeline = get_pipeline(name)
    inputs, outputs, preprocess, postprocess = get_interface_args(pipeline, system_prompt)
    fn = get_fn(name, preprocess, postprocess, api_key, system_prompt)

    if pipeline == "chat":
        interface = gr.ChatInterface(fn=fn, **kwargs)
    else:
        # For other pipelines, create a standard Interface (not implemented yet)
        interface = gr.Interface(fn=fn, inputs=inputs, outputs=outputs, **kwargs)

    return interface
