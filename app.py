import gradio as gr
import ai_gradio


gr.load(
    name='openrouter:openai/gpt-4o',
    src=ai_gradio.registry,
    coder=True
).launch()
