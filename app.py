import gradio as gr
import ai_gradio


gr.load(
    name='gemini:gemini-2.0-flash-lite-preview-02-05',
    src=ai_gradio.registry,
    coder=True
).launch()
