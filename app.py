import gradio as gr
import ai_gradio


gr.load(
    name='openai:o3-mini-2025-01-31',
    src=ai_gradio.registry,
    coder=True
).launch()
