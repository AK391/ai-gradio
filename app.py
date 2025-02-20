import gradio as gr
import ai_gradio


gr.load(
    name='huggingface:deepseek-ai/DeepSeek-R1',
    src=ai_gradio.registry,
    coder=True,
    provider="novita"
).launch()
