import gradio as gr
import ai_gradio


gr.load(
    name='openrouter:anthropic/claude-3.7-sonnet',
    src=ai_gradio.registry,
    coder=True,
).launch()
