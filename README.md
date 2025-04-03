# ai-gradio

A Python package that makes it easy for developers to create machine learning apps powered by various AI providers. Built on top of Gradio, it provides a unified interface for multiple AI models and services.

## Features

### Core Features
- **Multi-Provider Support**: Integrate with 15+ AI providers including OpenAI, Google Gemini, Anthropic, and more
- **Text Chat**: Interactive chat interfaces for all text models
- **Voice Chat**: Real-time voice interactions with OpenAI models
- **Video Chat**: Video processing capabilities with Gemini models
- **Code Generation**: Specialized interfaces for coding assistance
- **Multi-Modal**: Support for text, image, and video inputs
- **Agent Teams**: CrewAI integration for collaborative AI tasks
- **Browser Automation**: AI agents that can perform web-based tasks
- **Computer-Use**: AI agents that can control a virtual local macOS/Linux environment

### Model Support

#### Core Language Models
| Provider | Models |
|----------|---------|
| OpenAI | gpt-4-turbo, gpt-4, gpt-3.5-turbo |
| Anthropic | claude-3-opus, claude-3-sonnet, claude-3-haiku |
| Gemini | gemini-pro, gemini-pro-vision, gemini-2.0-flash-exp |
| Groq | llama-3.2-70b-chat, mixtral-8x7b-chat |

#### Specialized Models
| Provider | Type | Models |
|----------|------|---------|
| LumaAI | Generation | dream-machine, photon-1 |
| DeepSeek | Multi-purpose | deepseek-chat, deepseek-coder, deepseek-vision |
| CrewAI | Agent Teams | Support Team, Article Team |
| Qwen | Language | qwen-turbo, qwen-plus, qwen-max |
| Browser | Automation | browser-use-agent |
| Cua | Computer-Use | OpenAI Computer-Use Preview |

## Installation

### Basic Installation
```bash
# Install core package
pip install ai-gradio

# Install with specific provider support
pip install 'ai-gradio[openai]'     # OpenAI support
pip install 'ai-gradio[gemini]'     # Google Gemini support
pip install 'ai-gradio[anthropic]'  # Anthropic Claude support
pip install 'ai-gradio[groq]'       # Groq support

# Install all providers
pip install 'ai-gradio[all]'
```

### Additional Providers
```bash
pip install 'ai-gradio[crewai]'     # CrewAI support
pip install 'ai-gradio[lumaai]'     # LumaAI support
pip install 'ai-gradio[xai]'        # XAI/Grok support
pip install 'ai-gradio[cohere]'     # Cohere support
pip install 'ai-gradio[sambanova]'  # SambaNova support
pip install 'ai-gradio[hyperbolic]' # Hyperbolic support
pip install 'ai-gradio[deepseek]'   # DeepSeek support
pip install 'ai-gradio[smolagents]' # SmolagentsAI support
pip install 'ai-gradio[fireworks]'  # Fireworks support
pip install 'ai-gradio[together]'   # Together support
pip install 'ai-gradio[qwen]'       # Qwen support
pip install 'ai-gradio[browser]'    # Browser support
pip install 'ai-gradio[cua]'        # Computer-Use support
```

## Usage

### API Key Configuration
```bash
# Core Providers
export OPENAI_API_KEY=<your token>
export GEMINI_API_KEY=<your token>
export ANTHROPIC_API_KEY=<your token>
export GROQ_API_KEY=<your token>
export TAVILY_API_KEY=<your token>  # Required for Langchain agents

# Additional Providers (as needed)
export LUMAAI_API_KEY=<your token>
export XAI_API_KEY=<your token>
export COHERE_API_KEY=<your token>
# ... (other provider keys)

# Twilio credentials (required for WebRTC voice chat)
export TWILIO_ACCOUNT_SID=<your Twilio account SID>
export TWILIO_AUTH_TOKEN=<your Twilio auth token>
```

### Quick Start
```python
import gradio as gr
import ai_gradio

# Create a simple chat interface
gr.load(
    name='openai:gpt-4-turbo',  # or 'gemini:gemini-1.5-flash', 'groq:llama-3.2-70b-chat'
    src=ai_gradio.registry,
    title='AI Chat',
    description='Chat with an AI model'
).launch()

# Create a chat interface with Transformers models
gr.load(
    name='transformers:phi-4',  # or 'transformers:tulu-3', 'transformers:olmo-2-13b'
    src=ai_gradio.registry,
    title='Local AI Chat',
    description='Chat with locally running models'
).launch()

# Create a coding assistant with OpenAI
gr.load(
    name='openai:gpt-4-turbo',
    src=ai_gradio.registry,
    coder=True,
    title='OpenAI Code Assistant',
    description='OpenAI Code Generator'
).launch()

# Create a coding assistant with Gemini
gr.load(
    name='gemini:gemini-2.0-flash-thinking-exp-1219',  # or 'openai:gpt-4-turbo', 'anthropic:claude-3-opus'
    src=ai_gradio.registry,
    coder=True,
    title='Gemini Code Generator',
).launch()
```

### Advanced Features

#### Voice Chat
```python
gr.load(
    name='openai:gpt-4-turbo',
    src=ai_gradio.registry,
    enable_voice=True,
    title='AI Voice Assistant'
).launch()
```

#### Camera Mode
```python
# Create a vision-enabled interface with camera support
gr.load(
    name='gemini:gemini-2.0-flash-exp',
    src=ai_gradio.registry,
    camera=True,
).launch()
```

#### Multi-Provider Interface
```python
import gradio as gr
import ai_gradio

with gr.Blocks() as demo:
    with gr.Tab("Text"):
        gr.load('openai:gpt-4-turbo', src=ai_gradio.registry)
    with gr.Tab("Vision"):
        gr.load('gemini:gemini-pro-vision', src=ai_gradio.registry)
    with gr.Tab("Code"):
        gr.load('deepseek:deepseek-coder', src=ai_gradio.registry)

demo.launch()
```

#### CrewAI Teams
```python
# Article Creation Team
gr.load(
    name='crewai:gpt-4-turbo',
    src=ai_gradio.registry,
    crew_type='article',
    title='AI Writing Team'
).launch()
```

#### Browser Automation

```bash
playwright install
```

use python 3.11+ for browser use

```python
import gradio as gr
import ai_gradio

# Create a browser automation interface
gr.load(
    name='browser:gpt-4-turbo',
    src=ai_gradio.registry,
    title='AI Browser Assistant',
    description='Let AI help with web tasks'
).launch()
```

Example tasks:
- Flight searches on Google Flights
- Weather lookups
- Product price comparisons
- News searches

#### Computer-Use Agent

```bash
# Install Computer-Use Agent support
pip install 'ai-gradio[cua]'

# Install Lume daemon (macOS only)
sudo /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/trycua/cua/main/libs/lume/scripts/install.sh)"

# Start the Lume daemon service (in a separate terminal)
lume serve

# Pull the pre-built macOS image
lume pull macos-sequoia-cua:latest --no-cache
```

Requires macOS with Apple Silicon (M1/M2/M3/M4) and macOS 14 (Sonoma) or newer.

```python
import gradio as gr
import ai_gradio
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# Create a computer-use automation interface with OpenAI
gr.load(
    name='cua:gpt-4-turbo',  # Format: 'cua:model_name'
    src=ai_gradio.registry,
    title='Computer-Use Agent',
    description='AI that can control a virtual macOS environment'
).launch()
```

Example tasks:
- Create Python virtual environments and run data analysis scripts
- Open PDFs in Preview, add annotations, and save compressed versions
- Browse Safari and manage bookmarks
- Clone and build GitHub repositories
- Configure SSH keys and remote connections
- Create automation scripts and schedule them with cron

#### Swarms Integration
```python
import gradio as gr
import ai_gradio

# Create a chat interface with Swarms
gr.load(
    name='swarms:gpt-4-turbo',  # or other OpenAI models
    src=ai_gradio.registry,
    agent_name="Stock-Analysis-Agent",  # customize agent name
    title='Swarms Chat',
    description='Chat with an AI agent powered by Swarms'
).launch()
```

#### Langchain Agents
```python
import gradio as gr
import ai_gradio

# Create a Langchain agent interface
gr.load(
    name='langchain:gpt-4-turbo',  # or other supported models
    src=ai_gradio.registry,
    title='Langchain Agent',
    description='AI agent powered by Langchain'
).launch()
```

## Requirements

### Core Requirements
- Python 3.10+
- gradio >= 5.9.1

### Optional Features
- Voice Chat: gradio-webrtc, numba==0.60.0, pydub, librosa
- Video Chat: opencv-python, Pillow
- Agent Teams: crewai>=0.1.0, langchain>=0.1.0

## Troubleshooting

### Authentication Issues
If you encounter 401 errors, verify your API keys:
```python
import os

# Set API keys manually if needed
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["GEMINI_API_KEY"] = "your-api-key"
```

### Provider Installation
If you see "no providers installed" errors:
```bash
# Install specific provider
pip install 'ai-gradio[provider_name]'

# Or install all providers
pip install 'ai-gradio[all]'
```


## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.






