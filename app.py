import gradio as gr
import ai_gradio


gr.load(
    name='ollama:smollm',
    src=ai_gradio.registry,
).launch()
