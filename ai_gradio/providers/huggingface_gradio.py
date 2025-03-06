import os
import asyncio
import base64
import time
from threading import Event, Thread
import re

import gradio as gr
import numpy as np
from huggingface_hub import InferenceClient
from gradio_webrtc import (
    AdditionalOutputs,
    StreamHandler,
    WebRTC,
    get_twilio_turn_credentials,
)
from pydub import AudioSegment
import modelscope_studio.components.base as ms
import modelscope_studio.components.legacy as legacy
import modelscope_studio.components.antd as antd

SAMPLE_RATE = 24000

SystemPrompt = """You are an expert web developer specializing in creating clean, efficient, and modern web applications.
Your task is to write complete, self-contained HTML files that include all necessary CSS and JavaScript.
Focus on:
- be concise
- Writing clear, maintainable code
- Following best practices
- Creating responsive designs
- Adding appropriate styling and interactivity
Return only the complete HTML code without any additional explanation."""

# Add default settings
DEFAULT_SETTINGS = {
    "system": SystemPrompt,
    "reasoning_effort": "high"  # Add default reasoning effort
}

DEMO_LIST = [
    {
        "card": {"index": 0},
        "title": "Simple Button",
        "description": "Create a button that changes color when clicked"
    },
    {
        "card": {"index": 1},
        "title": "Todo List",
        "description": "Create a simple todo list with add/remove functionality"
    },
    {
        "card": {"index": 2},
        "title": "Timer App",
        "description": "Create a countdown timer with start/pause/reset controls"
    }
]

def encode_audio(sample_rate, data):
    segment = AudioSegment(
        data.tobytes(),
        frame_rate=sample_rate,
        sample_width=data.dtype.itemsize,
        channels=1,
    )
    pcm_audio = (
        segment.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2).raw_data
    )
    return base64.b64encode(pcm_audio).decode("utf-8")

class RealtimeHandler(StreamHandler):
    def __init__(
        self,
        expected_layout="mono",
        output_sample_rate=SAMPLE_RATE,
        output_frame_size=480,
        model=None
    ) -> None:
        super().__init__(
            expected_layout,
            output_sample_rate,
            output_frame_size,
            input_sample_rate=SAMPLE_RATE,
        )
        self.model = model
        # Initialize Event objects first
        self.args_set = Event()
        self.quit = Event()
        self.connected = Event()
        self.reset_state()

    def reset_state(self):
        """Reset connection state for new recording session"""
        self.connection = None
        self.args_set.clear()
        self.quit.clear()
        self.connected.clear()
        self.thread = None
        self._generator = None
        self.current_session = None

    def copy(self):
        return RealtimeHandler(
            expected_layout=self.expected_layout,
            output_sample_rate=self.output_sample_rate,
            output_frame_size=self.output_frame_size,
            model=self.model
        )

    def _initialize_connection(self, api_key: str):
        self.client = InferenceClient(
            provider="together",
            api_key=api_key
        )
        with self.client.beta.realtime.connect(
            model=self.model
        ) as conn:
            conn.session.update(session={"turn_detection": {"type": "server_vad"}})
            self.connection = conn
            self.connected.set()
            self.current_session = conn.session
            while not self.quit.is_set():
                time.sleep(0.25)

    async def fetch_args(self):
        if self.channel:
            self.channel.send("tick")

    def set_args(self, args):
        super().set_args(args)
        self.args_set.set()

    def receive(self, frame: tuple[int, np.ndarray]) -> None:
        if not self.channel:
            return
        try:
            # Initialize connection if needed
            if not self.connection:
                asyncio.run_coroutine_threadsafe(self.fetch_args(), self.loop)
                self.args_set.wait()
                self.thread = Thread(
                    target=self._initialize_connection, args=(self.latest_args[-1],)
                )
                self.thread.start()
                self.connected.wait()
            
            # Send audio data
            assert self.connection, "Connection not initialized"
            sample_rate, array = frame
            array = array.squeeze()
            audio_message = encode_audio(sample_rate, array)
            
            # Send the audio data
            self.connection.input_audio_buffer.append(audio=audio_message)
            
        except Exception as e:
            print(f"Error in receive: {str(e)}")
            import traceback
            traceback.print_exc()

    def generator(self):
        while True:
            if not self.connection:
                yield None
                continue
            for event in self.connection:
                if event.type == "response.audio_transcript.done":
                    yield AdditionalOutputs(event)
                if event.type == "response.audio.delta":
                    yield (
                        self.output_sample_rate,
                        np.frombuffer(
                            base64.b64decode(event.delta), dtype=np.int16
                        ).reshape(1, -1),
                    )

    def emit(self) -> tuple[int, np.ndarray] | None:
        if not self.connection:
            return None
        if not self._generator:
            self._generator = self.generator()
        try:
            return next(self._generator)
        except StopIteration:
            self._generator = self.generator()
            return None

    def shutdown(self) -> None:
        if self.connection:
            self.quit.set()
            self.connection.close()
            if self.thread:
                self.thread.join(timeout=5)
            self.reset_state()  # Reset state after shutdown

def update_chatbot(chatbot: list[dict], response):
    chatbot.append({"role": "assistant", "content": response.transcript})
    return chatbot

def get_fn(api_key: str, model: str, provider: str = "together"):
    def chat_fn(message, history):
        client = InferenceClient(
            provider=provider,
            api_key=api_key
        )
        messages = []
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": message})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4000,
            stream=True
        )
        
        partial_message = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                partial_message += chunk.choices[0].delta.content
                yield partial_message
    
    return chat_fn

def demo_card_click(evt: gr.EventData):
    """Handle demo card clicks by returning the demo description"""
    if not evt:
        return ""
    # Access index directly from event data
    index = evt._data['component']['index']
    return DEMO_LIST[index]['description']

def registry(name: str, token: str | None = None, twilio_sid: str | None = None, twilio_token: str | None = None, enable_voice: bool = False, coder: bool = False, provider: str = "together", **kwargs):
    api_key = token or os.environ.get("HF_TOKEN")
    if not api_key:
        raise ValueError("HF_TOKEN environment variable is not set.")

    model = name if name else "gpt-4"
    
    if coder:
        interface = gr.Blocks(css="""
            .left_header {
                text-align: center;
                margin-bottom: 20px;
            }

            .right_panel {
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            }

            .render_header {
                background: #f5f5f5;
                padding: 8px;
                border-bottom: 1px solid #e8e8e8;
            }

            .header_btn {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
                background: #ff5f56;
            }

            .header_btn:nth-child(2) {
                background: #ffbd2e;
            }

            .header_btn:nth-child(3) {
                background: #27c93f;
            }

            .right_content {
                padding: 24px;
                height: 920px;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .html_content {
                height: 920px;
                width: 100%;
            }

            .history_chatbot {
                height: 100%;
            }

            .image-upload-section {
                margin-bottom: 16px;
            }
            
            .image-upload-label {
                font-size: 14px;
                margin-bottom: 8px;
            }
        """)
        
        with interface:
            history = gr.State([])
            setting = gr.State(DEFAULT_SETTINGS)  # Use default settings
            model_name = gr.State(model)
            provider_state = gr.State(provider)  # Add provider state

            with ms.Application() as app:
                with antd.ConfigProvider():
                    with antd.Row(gutter=[32, 12]) as layout:
                        # Left Column
                        with antd.Col(span=24, md=8):
                            with antd.Flex(vertical=True, gap="middle", wrap=True):
                                header = gr.HTML("""
                                    <div class="left_header">
                                        <h1>Hugging Face Code Generator</h1>
                                    </div>
                                """)
                                
                                # Add image upload section
                                with gr.Group(elem_classes="image-upload-section"):
                                    image_input = gr.Image(
                                        label="Upload Reference Images",
                                        type="filepath",
                                        height=200
                                    )
                                
                                input = antd.InputTextarea(
                                    size="large",
                                    allow_clear=True,
                                    placeholder="Describe the web application you want to create"
                                )
                                btn = antd.Button("Generate", type="primary", size="large")
                                clear_btn = antd.Button("Clear History", type="default", size="large")

                                antd.Divider("Examples")
                                with antd.Flex(gap="small", wrap=True):
                                    with ms.Each(DEMO_LIST):
                                        with antd.Card(hoverable=True, as_item="card") as demoCard:
                                            antd.CardMeta()
                                        
                                antd.Divider("Settings")
                                with antd.Flex(gap="small", wrap=True):
                                    settingPromptBtn = antd.Button("⚙️ System Prompt", type="default")
                                    reasoningBtn = antd.Button(
                                        "🎯 Reasoning Effort", 
                                        type="default",
                                        visible="o3" in model.lower()
                                    )
                                    providerBtn = antd.Button("🔄 Provider", type="default")
                                    codeBtn = antd.Button("🧑‍💻 View Code", type="default")
                                    historyBtn = antd.Button("📜 History", type="default")

                            # Modals and Drawers
                            with antd.Modal(open=False, title="System Prompt", width="800px") as system_prompt_modal:
                                systemPromptInput = antd.InputTextarea(SystemPrompt, auto_size=True)

                            with antd.Drawer(open=False, title="Code", placement="left", width="750px") as code_drawer:
                                code_output = legacy.Markdown()

                            with antd.Drawer(open=False, title="History", placement="left", width="900px") as history_drawer:
                                history_output = legacy.Chatbot(
                                    show_label=False,
                                    height=960,
                                    elem_classes="history_chatbot"
                                )

                            # Update Modal for reasoning effort - add value parameter
                            with antd.Modal(open=False, title="Reasoning Effort", width="400px") as reasoning_modal:
                                reasoningSelect = antd.Select(
                                    options=[
                                        {"label": "High", "value": "high"},
                                        {"label": "Medium", "value": "medium"},
                                        {"label": "Low", "value": "low"}
                                    ],
                                    default_value="high",
                                    value="high"  # Add explicit value
                                )

                            # Add Provider Modal
                            with antd.Modal(open=False, title="Select Provider", width="400px") as provider_modal:
                                providerSelect = antd.Select(
                                    options=[
                                        {"label": "Together", "value": "together"},
                                        {"label": "Fireworks AI", "value": "fireworks-ai"},
                                        {"label": "Hyperbolic", "value": "hyperbolic"},
                                        {"label": "SambaNova", "value": "sambanova"},
                                        {"label": "Nebius", "value": "nebius"},
                                        {"label": "Novita", "value": "novita"}
                                    ],
                                    default_value=provider,
                                    value=provider
                                )

                        # Right Column
                        with antd.Col(span=24, md=16):
                            with ms.Div(elem_classes="right_panel"):
                                gr.HTML('''
                                    <div class="render_header">
                                        <span class="header_btn"></span>
                                        <span class="header_btn"></span>
                                        <span class="header_btn"></span>
                                    </div>
                                ''')
                                with antd.Tabs(active_key="empty", render_tab_bar="() => null") as state_tab:
                                    with antd.Tabs.Item(key="empty"):
                                        empty = antd.Empty(
                                            description="Enter your request to generate code",
                                            elem_classes="right_content"
                                        )
                                    with antd.Tabs.Item(key="loading"):
                                        loading = antd.Spin(
                                            True,
                                            tip="Generating code...",
                                            size="large",
                                            elem_classes="right_content"
                                        )
                                    with antd.Tabs.Item(key="render"):
                                        preview = gr.HTML(elem_classes="html_content")

            # Wire up event handlers
            demoCard.click(demo_card_click, outputs=[input])
            settingPromptBtn.click(lambda: gr.update(open=True), outputs=[system_prompt_modal])
            system_prompt_modal.ok(
                lambda input: ({"system": input}, gr.update(open=False)),
                inputs=[systemPromptInput],
                outputs=[setting, system_prompt_modal]
            )
            system_prompt_modal.cancel(lambda: gr.update(open=False), outputs=[system_prompt_modal])
            
            codeBtn.click(lambda: gr.update(open=True), outputs=[code_drawer])
            code_drawer.close(lambda: gr.update(open=False), outputs=[code_drawer])
            
            historyBtn.click(
                lambda h: (gr.update(open=True), h),
                inputs=[history],
                outputs=[history_drawer, history_output]
            )
            history_drawer.close(lambda: gr.update(open=False), outputs=[history_drawer])
            
            reasoningBtn.click(lambda: gr.update(open=True), outputs=[reasoning_modal])
            reasoning_modal.ok(
                lambda value, current_setting: (
                    {"system": current_setting["system"], "reasoning_effort": value}, 
                    gr.update(open=False)
                ),
                inputs=[reasoningSelect, setting],
                outputs=[setting, reasoning_modal]
            )
            reasoning_modal.cancel(lambda: gr.update(open=False), outputs=[reasoning_modal])
            
            providerBtn.click(lambda: gr.update(open=True), outputs=[provider_modal])
            provider_modal.ok(
                lambda value: (value, gr.update(open=False)),
                inputs=[providerSelect],
                outputs=[provider_state, provider_modal]
            )
            provider_modal.cancel(lambda: gr.update(open=False), outputs=[provider_modal])
            
            btn.click(
                generate_code,
                inputs=[input, image_input, setting, history, model_name, provider_state],  # Use provider_state instead of provider string
                outputs=[code_output, history, preview, state_tab, code_drawer]
            )
            
            clear_btn.click(lambda: [], outputs=[history])

        return interface

    # Check if model is a realtime model or if enable_voice is True
    is_realtime_model = "realtime" in model.lower()
    use_voice_ui = enable_voice or is_realtime_model

    if use_voice_ui:
        # Voice UI implementation
        with gr.Blocks() as interface:
            # Set initial visibility based on whether API key is provided
            show_api_input = api_key is None
            
            with gr.Row(visible=show_api_input) as api_key_row:
                api_key_input = gr.Textbox(
                    label="OpenAI API Key",
                    placeholder="Enter your OpenAI API Key",
                    value=api_key,
                    type="password",
                )
                
            with gr.Row(visible=not show_api_input) as row:
                webrtc = WebRTC(
                    label="Conversation",
                    modality="audio",
                    mode="send-receive",
                    rtc_configuration=get_twilio_turn_credentials(twilio_sid, twilio_token),
                )
                    
                webrtc.stream(
                    RealtimeHandler(model=model),
                    inputs=[webrtc, api_key_input],
                    outputs=[webrtc],
                    time_limit=90,
                    concurrency_limit=10,
                )
                
            api_key_input.submit(
                lambda: (gr.update(visible=False), gr.update(visible=True)),
                None,
                [api_key_row, row],
            )
    else:
        # New chat interface implementation
        interface = gr.ChatInterface(
            fn=get_fn(api_key, model, provider),
            **kwargs
        )

    return interface

def remove_code_block(text):
    # First try to find HTML code block
    pattern = r'```html\n([\s\S]+?)\n```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If no HTML block found, try to find any code block
    pattern = r'```[^\n]*\n([\s\S]+?)\n```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
        
    # If no code block found, check if the text itself is HTML
    if text.strip().startswith('<') and text.strip().endswith('>'):
        return text.strip()
        
    # If no HTML found, return empty string to prevent text display
    return ''

def send_to_preview(code):
    encoded_html = base64.b64encode(code.encode('utf-8')).decode('utf-8')
    data_uri = f"data:text/html;charset=utf-8;base64,{encoded_html}"
    return f'<iframe src="{data_uri}" width="100%" height="920px"></iframe>'

def generate_code(query, image, setting, history, model, provider: str = "together"):
    # Get api_key from environment
    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        raise ValueError("HF_TOKEN environment variable is not set.")
        
    client = InferenceClient(
        provider=provider,
        api_key=api_key
    )
    messages = []
    messages.append({
        "role": "system", 
        "content": setting["system"]
    })
    
    # Add history
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    
    # Prepare the message content
    content = []
    if query:
        content.append({"type": "text", "text": query})
    
    if image:
        # Read and encode the image
        with open(image, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}"
            }
        })
    
    # Use vision model if image is present
    model_to_use = "deepseek-ai/DeepSeek-R1" if image else model
    
    # Add the content to messages
    messages.append({"role": "user", "content": content if image else query})
    
    # Create completion
    completion_params = {
        "model": model_to_use,
        "messages": messages,
        "max_tokens": 4000,
        "stream": True
    }
    
    response = client.chat.completions.create(**completion_params)
    
    response_text = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            response_text += chunk.choices[0].delta.content
            # Return all 5 required outputs
            yield (
                response_text,  # code_output (modelscopemarkdown)
                history,        # state
                None,          # preview (html)
                gr.update(active_key="loading"),  # state_tab (antdtabs)
                gr.update(open=True)  # code_drawer (antddrawer)
            )
    
    clean_code = remove_code_block(response_text)
    new_history = history + [(query, response_text)]
    
    # Final yield with all outputs
    yield (
        response_text,  # code_output (modelscopemarkdown)
        new_history,    # state
        send_to_preview(clean_code),  # preview (html)
        gr.update(active_key="render"),  # state_tab (antdtabs)
        gr.update(open=False)  # code_drawer (antddrawer)
    )