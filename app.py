import gradio as gr
import ai_gradio


gr.load(
    name='huggingface:Qwen/QwQ-32B',
    src=ai_gradio.registry,
    coder=True,
    provider="hyperbolic"
).launch()
