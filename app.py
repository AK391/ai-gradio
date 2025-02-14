import gradio as gr
import ai_gradio


gr.load(
    name='openrouter:deepseek/deepseek-r1',
    src=ai_gradio.registry,
    coder=True
).launch()
