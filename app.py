import gradio as gr
import ai_gradio


gr.load(
    name='openrouter:openai/o3-mini-high',
    src=ai_gradio.registry,
    coder=True
).launch()
