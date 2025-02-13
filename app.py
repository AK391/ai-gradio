import gradio as gr
import ai_gradio


gr.load(
    name='sambanova:DeepSeek-R1',
    src=ai_gradio.registry,
    coder=True
).launch()
