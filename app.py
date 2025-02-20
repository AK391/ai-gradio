import gradio as gr
import ai_gradio


gr.load(
    name='openrouter:perplexity/r1-1776',
    src=ai_gradio.registry,
    coder=True,
).launch()
