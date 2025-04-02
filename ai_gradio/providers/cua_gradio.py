import os
import asyncio
from typing import Callable, Dict, List, Any, Optional
import gradio as gr
from importlib.util import find_spec
import uuid

"""
Computer-Use Agent (CUA) Provider for ai-gradio

This provider integrates the Cua framework (https://github.com/trycua/cua) with ai-gradio,
enabling AI agents to control virtual macOS machines. The Computer-Use Agent provides a secure,
isolated environment for AI systems to interact with desktop applications, browse the web,
write code, and perform complex workflows.

Supported Agent Loops and Models:
- AgentLoop.OPENAI: Uses OpenAI Operator CUA model
  • computer_use_preview
  
- AgentLoop.ANTHROPIC: Uses Anthropic Computer-Use models
  • claude-3-5-sonnet-20240620
  • claude-3-7-sonnet-20250219
  
- AgentLoop.OMNI (experimental): Uses OmniParser for element pixel-detection
  • claude-3-5-sonnet-20240620
  • claude-3-7-sonnet-20250219
  • gpt-4.5-preview
  • gpt-4o
  • gpt-4

Usage:
    import gradio as gr
    import ai_gradio
    from dotenv import load_dotenv
    
    # Load API keys from .env file
    load_dotenv()
    
    # Create a CUA interface with GPT-4 Turbo
    gr.load(
        name='cua:gpt-4-turbo',  # Format: 'cua:model_name'
        src=ai_gradio.registry,
        title="Computer-Use Agent",
        description="AI that can control a virtual computer"
    ).launch()

Example prompts:
    - "Search for a repository named trycua/cua on GitHub"
    - "Open VS Code and create a new Python file"
    - "Open Terminal and run the command 'ls -la'"
    - "Go to apple.com and take a screenshot"

Requirements:
    - Mac with Apple Silicon (M1/M2/M3/M4)
    - macOS 14 (Sonoma) or newer
    - Python 3.10+
    - Lume CLI installed (https://github.com/trycua/cua)
    - OpenAI or Anthropic API key
"""

# Create a registry object that will be exported regardless of whether cua is installed
registry = None
CUA_AVAILABLE = False

# Try to import cua libraries
if find_spec("computer") and find_spec("agent"):
    from computer import Computer
    from agent import ComputerAgent, LLM, AgentLoop, LLMProvider
    CUA_AVAILABLE = True
    
    # Global logging level
    LOGGING_LEVEL = os.environ.get("CUA_LOGGING_LEVEL", "INFO")
    import logging
    NUMERIC_LEVEL = getattr(logging, LOGGING_LEVEL.upper(), logging.INFO)
    
    # Create a single global Computer instance for all tasks
    # This avoids creating a new Computer for each session
    GLOBAL_COMPUTER = Computer(verbosity=NUMERIC_LEVEL)
    
    # Create a single global ComputerAgent with default settings
    # Parameters will be updated as needed for each request
    GLOBAL_AGENT = ComputerAgent(
        computer=GLOBAL_COMPUTER,
        loop=AgentLoop.OPENAI,  # Default loop
        model=LLM(
            provider=LLMProvider.OPENAI,  # Default provider
            name="computer_use_preview",  # Default model
        ),
        save_trajectory=True,
        only_n_most_recent_images=3,
        verbosity=NUMERIC_LEVEL,
        api_key=os.environ.get("OPENAI_API_KEY", "")  # Default API key
    )
    
else:
    # Provide helpful error message if libraries aren't installed
    raise ImportError(
        "The cua libraries could not be imported. Please install them with: "
        "pip install 'ai-gradio[cua]'"
    )

# Map model names to specific provider model names
MODEL_MAPPINGS = {
    "openai": {
        # Default to operator CUA model
        "default": "computer_use_preview",
        # Map standard OpenAI model names to CUA-specific model names
        "gpt-4-turbo": "computer_use_preview",
        "gpt-4o": "computer_use_preview",
        "gpt-4": "computer_use_preview",
        "gpt-4.5-preview": "computer_use_preview",
    },
    "anthropic": {
        # Default to newest model
        "default": "claude-3-7-sonnet-20250219",
        # Specific Claude models for CUA
        "claude-3-5-sonnet-20240620": "claude-3-5-sonnet-20240620",
        "claude-3-7-sonnet-20250219": "claude-3-7-sonnet-20250219",
        # Map standard model names to CUA-specific model names
        "claude-3-opus": "claude-3-7-sonnet-20250219",
        "claude-3-sonnet": "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet": "claude-3-5-sonnet-20240620",
        "claude-3-7-sonnet": "claude-3-7-sonnet-20250219",
    },
    "omni": {
        # OMNI works with any of these models
        "default": "gpt-4o",
        "gpt-4o": "gpt-4o",
        "gpt-4": "gpt-4",
        "gpt-4.5-preview": "gpt-4.5-preview",
        "claude-3-5-sonnet-20240620": "claude-3-5-sonnet-20240620",
        "claude-3-7-sonnet-20250219": "claude-3-7-sonnet-20250219",
    }
}

def get_provider_and_model(model_name: str, loop_provider: str) -> tuple:
    """
    Determine the provider and actual model name to use based on the input.
    
    Args:
        model_name: The requested model name
        loop_provider: The requested agent loop provider
    
    Returns:
        tuple: (provider, model_name_to_use, agent_loop)
    """
    # Get the agent loop
    loop_provider_map = {
        "OPENAI": AgentLoop.OPENAI,
        "ANTHROPIC": AgentLoop.ANTHROPIC,
        "OMNI": AgentLoop.OMNI
    }
    agent_loop = loop_provider_map.get(loop_provider, AgentLoop.OPENAI)
    
    # Set up the provider and model based on the loop and model_name
    if agent_loop == AgentLoop.OPENAI:
        provider = LLMProvider.OPENAI
        model_name_to_use = MODEL_MAPPINGS["openai"].get(model_name.lower(), MODEL_MAPPINGS["openai"]["default"])
    elif agent_loop == AgentLoop.ANTHROPIC:
        provider = LLMProvider.ANTHROPIC
        model_name_to_use = MODEL_MAPPINGS["anthropic"].get(model_name.lower(), MODEL_MAPPINGS["anthropic"]["default"])
    elif agent_loop == AgentLoop.OMNI:
        # For OMNI, select provider based on model name
        if "claude" in model_name.lower():
            provider = LLMProvider.ANTHROPIC
            model_name_to_use = MODEL_MAPPINGS["omni"].get(model_name.lower(), MODEL_MAPPINGS["omni"]["default"])
        else:
            provider = LLMProvider.OPENAI
            model_name_to_use = MODEL_MAPPINGS["omni"].get(model_name.lower(), MODEL_MAPPINGS["omni"]["default"])
    else:
        # Default to OpenAI if unrecognized loop
        provider = LLMProvider.OPENAI
        model_name_to_use = MODEL_MAPPINGS["openai"]["default"]
        agent_loop = AgentLoop.OPENAI
    
    return provider, model_name_to_use, agent_loop

def clean_text(text: str) -> str:
    """
    Clean up response text by improving status pattern formatting.
    
    Args:
        text: The text to clean
    
    Returns:
        str: Cleaned text with better formatting for status indicators
    """
    if not text:
        return ""
    
    # Replace awkward ". completed." with a better format
    text = text.replace(". completed.", ". ✓")
    text = text.replace(". completed", " ✓")
    
    return text

def extract_synthesized_text(result: Dict[str, Any]) -> str:
    """
    Extract synthesized text from the agent result.
    
    Args:
        result: The agent result
    
    Returns:
        str: The synthesized text
    """
    synthesized_text = ""
    
    if "output" in result and result["output"]:
        for output in result["output"]:
            if output.get("type") == "reasoning":
                content = output.get("content", "")
                if content:
                    synthesized_text += f"{content}\n"
                    
                # If there's a summary, use it
                if "summary" in output and output["summary"]:
                    for summary_item in output["summary"]:
                        if isinstance(summary_item, dict) and summary_item.get("text"):
                            synthesized_text += f"{summary_item['text']}\n"
            
            elif output.get("type") == "message":
                # Handle message type outputs - can contain rich content
                content = output.get("content", [])
                
                # Content is usually an array of content blocks
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "output_text":
                            text_value = block.get("text", "")
                            if text_value:
                                synthesized_text += f"{text_value}\n"
            
            elif output.get("type") == "computer_call":
                action = output.get("action", {})
                action_type = action.get("type", "unknown")
                
                # Create a descriptive text about the action
                if action_type == "click":
                    button = action.get("button", "")
                    x = action.get("x", "")
                    y = action.get("y", "")
                    synthesized_text += f"Clicked {button} at position ({x}, {y}).\n"
                elif action_type == "type":
                    text = action.get("text", "")
                    synthesized_text += f"Typed: {text}.\n"
                elif action_type == "keypress":
                    # Extract key correctly from either keys array or key field
                    if isinstance(action.get("keys"), list):
                        key = ", ".join(action.get("keys"))
                    else:
                        key = action.get("key", "")
                    
                    # Clean up any formatting issues
                    if ". completed" in key:
                        key = key.replace(". completed", "")
                        
                    synthesized_text += f"Pressed key: {key}\n"
                else:
                    synthesized_text += f"Performed {action_type} action.\n"
    
    return synthesized_text.strip()

def extract_reasoning_details(outputs: List[Dict[str, Any]]) -> List[str]:
    """
    Extract reasoning details from outputs.
    
    Args:
        outputs: List of output items
    
    Returns:
        List of reasoning detail strings
    """
    reasoning_details = []
    
    for output in outputs:
        if output.get("type") == "reasoning":
            # Include full reasoning details
            content = output.get("content", "")
            
            reasoning_text = f"Reasoning: {content}"
            
            # Add summary if available
            if "summary" in output:
                for summary_item in output["summary"]:
                    if summary_item.get("type") == "summary_text":
                        reasoning_text += f"\n↪ Summary: {summary_item.get('text', '')}"
            
            reasoning_details.append(reasoning_text)
        elif output.get("type") == "message":
            # Handle message type outputs - can contain rich content
            content = output.get("content", [])
            
            # Create a message_text variable to collect message content
            message_text = ""
            
            # Content is usually an array of content blocks
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        text_value = block.get("text", "")
                        if text_value:
                            message_text += f"{text_value}\n"
            
            # Add message to reasoning details if it has content
            if message_text.strip():
                reasoning_details.append(f"Message: {message_text.strip()}")
    
    return reasoning_details

def extract_tool_call_details(outputs: List[Dict[str, Any]]) -> List[str]:
    """
    Extract tool call details from outputs.
    
    Args:
        outputs: List of output items
    
    Returns:
        List of tool call detail strings
    """
    tool_call_details = []
    
    for output in outputs:
        if output.get("type") == "computer_call":
            # Extract tool call details
            tool = output.get("tool", "unknown")
            status = output.get("status", "")
            
            # Build a detailed tool call description
            tool_call_text = f"Action: {tool} (Status: {status})"
            
            # Add action details if available
            if "action" in output:
                action = output["action"]
                action_type = action.get("type", "")
                
                if action_type == "click":
                    button = action.get("button", "")
                    x = action.get("x", "")
                    y = action.get("y", "")
                    tool_call_text += f"\n↪ {action_type.capitalize()} {button} at ({x}, {y})"
                else:
                    tool_call_text += f"\n↪ {action_type.capitalize()}: {str(action)}"
            
            # Add safety checks if available
            if "pending_safety_checks" in output and output["pending_safety_checks"]:
                tool_call_text += "\n↪ Safety Checks:"
                for check in output["pending_safety_checks"]:
                    check_code = check.get("code", "")
                    check_msg = check.get("message", "").split(".")[0]  # First sentence only
                    tool_call_text += f"\n  • {check_code}: {check_msg}"
            
            tool_call_details.append(tool_call_text)
    
    return tool_call_details

def extract_usage_info(usage: Dict[str, Any]) -> List[str]:
    """
    Extract usage information.
    
    Args:
        usage: Usage data dictionary
    
    Returns:
        List of usage info strings
    """
    usage_info = []
    
    # Basic token usage
    tokens_text = f"Tokens: {usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out"
    usage_info.append(tokens_text)
    
    # Add detailed token breakdowns if available
    if "input_tokens_details" in usage:
        input_details = usage["input_tokens_details"]
        if isinstance(input_details, dict):
            for key, value in input_details.items():
                usage_info.append(f"Input {key}: {value}")
    
    if "output_tokens_details" in usage:
        output_details = usage["output_tokens_details"]
        if isinstance(output_details, dict):
            for key, value in output_details.items():
                usage_info.append(f"Output {key}: {value}")
    
    return usage_info

def update_global_agent(provider, agent_loop, model_name, api_key, save_trajectory, only_n_most_recent_images):
    """
    Update the global agent's parameters.
    
    Args:
        provider: The LLM provider
        agent_loop: The agent loop type
        model_name: The model name to use
        api_key: The API key
        save_trajectory: Whether to save the agent trajectory
        only_n_most_recent_images: Number of most recent images to keep
    """
    global GLOBAL_AGENT
    
    # Update the agent's parameters
    GLOBAL_AGENT.loop = agent_loop
    GLOBAL_AGENT.model = LLM(
        provider=provider,
        name=model_name,
    )
    GLOBAL_AGENT.save_trajectory = save_trajectory
    GLOBAL_AGENT.only_n_most_recent_images = only_n_most_recent_images
    GLOBAL_AGENT.api_key = api_key

def get_fn(
    model_name: str, 
    preprocess: Callable, 
    postprocess: Callable, 
    api_key: str,
    save_trajectory: bool = True,
    only_n_most_recent_images: int = 3,
    session_id: str = "default",
    loop_provider: str = "OPENAI"
):
    """Create a function that processes messages with the CUA agent.
    
    Args:
        model_name: The name of the LLM model to use
        preprocess: Function to preprocess messages
        postprocess: Function to postprocess results
        api_key: API key for the LLM provider
        save_trajectory: Whether to save the agent's trajectory
        only_n_most_recent_images: Number of most recent images to keep
        session_id: Unique identifier for the session
        loop_provider: Which agent loop to use (OPENAI, ANTHROPIC, OMNI)
    
    Returns:
        A function that processes messages with the CUA agent
    """
    
    async def fn(message, history):
        inputs = preprocess(message, history)
        
        try:
            # Get provider, model name, and agent loop
            provider, model_name_to_use, agent_loop = get_provider_and_model(model_name, loop_provider)
            
            # Update the global agent with the current parameters
            update_global_agent(
                provider,
                agent_loop,
                model_name_to_use,
                api_key,
                save_trajectory,
                only_n_most_recent_images
            )
            
            # Use the global agent
            agent = GLOBAL_AGENT
            
            # Process the message
            results = []
            async for result in agent.run(inputs["message"]):
                results.append(result)  # Store raw results
                yield postprocess([result])  # Process single result for streaming
            
            yield postprocess(results)  # Process all results for final output
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield {"role": "assistant", "content": f"Error: {str(e)}"}
    
    return fn

def get_interface_args(pipeline):
    if pipeline == "cua":
        def preprocess(message, history):
            return {"message": message}

        def postprocess(results):
            """Format agent results for the Gradio UI.
            
            Args:
                results: List of raw agent results
                
            Returns:
                A formatted message for the Gradio chatbot
            """
            if not results:
                return {
                    "role": "assistant",
                    "content": "No results were returned from the computer agent."
                }
            
            # Process raw results to extract needed information
            processed_results = []
            
            for result in results:
                processed = {}
                
                # Basic information
                processed["id"] = result.get("id", "")
                processed["status"] = result.get("status", "")
                processed["model"] = result.get("model", "")
                
                # Extract text content
                text_obj = result.get("text", {})
                
                # For OpenAI's Computer-Use Agent, text field is an object with format property
                if text_obj and isinstance(text_obj, dict) and "format" in text_obj and not text_obj.get("value", ""):
                    synthesized_text = extract_synthesized_text(result)
                    processed["text"] = synthesized_text if synthesized_text else ""
                else:
                    # For other types of results, try to get text directly
                    if isinstance(text_obj, dict):
                        if "value" in text_obj:
                            processed["text"] = text_obj["value"]
                        elif "text" in text_obj:
                            processed["text"] = text_obj["text"]
                        elif "content" in text_obj:
                            processed["text"] = text_obj["content"]
                        else:
                            processed["text"] = ""
                    else:
                        processed["text"] = str(text_obj) if text_obj else ""
                
                # Clean up the text
                processed["text"] = clean_text(processed["text"])
                
                # Extract detailed information
                if "output" in result and result["output"]:
                    output = result["output"]
                    processed["reasoning_details"] = extract_reasoning_details(output)
                    processed["tool_call_details"] = extract_tool_call_details(output)
                    
                    # If no text was found, try to generate a status message
                    if not processed["text"]:
                        if output and len(output) > 0:
                            most_recent_output = output[-1]
                            if most_recent_output.get("type") == "computer_call":
                                action = most_recent_output.get("action", {})
                                action_type = action.get("type", "")
                                if action_type:
                                    processed["text"] = f"Performing action: {action_type}"
                
                # Extract usage information
                if "usage" in result:
                    processed["usage_info"] = extract_usage_info(result["usage"])
                
                processed_results.append(processed)
            
            # Get the final result for display
            final_result = processed_results[-1] if processed_results else {}
            
            # Extract the main text response
            main_response = final_result.get("text", "")
            
            # Prepare intermediate steps if available
            steps = []
            for i, result in enumerate(processed_results[:-1], 1):
                text = result.get("text", "")
                if isinstance(text, str) and text.strip():
                    steps.append(f"Step {i}: {text}")
            
            # Is this a streaming update or final result?
            is_streaming = len(processed_results) == 1 and not steps
            
            # Prepare metadata for the rich response
            metadata = {}
            
            if not is_streaming:
                metadata["title"] = "🖥️ " + " → ".join(steps) if steps else "🖥️ Task in progress..."
            
            # Organize detailed information sections
            detailed_sections = []
            
            # Get information from the final result
            reasoning_details = final_result.get("reasoning_details", [])
            tool_call_details = final_result.get("tool_call_details", [])
            usage_info = final_result.get("usage_info", [])
            
            if reasoning_details:
                detailed_sections.append("🧠 Reasoning:")
                detailed_sections.extend([f"  {r}" for r in reasoning_details])
            
            if tool_call_details:
                detailed_sections.append("🔧 Actions:")
                detailed_sections.extend([f"  {t}" for t in tool_call_details])
            
            if usage_info and not is_streaming:
                detailed_sections.append("📊 Usage:")
                detailed_sections.extend([f"  {u}" for u in usage_info])
            
            if detailed_sections:
                metadata["subtitle"] = "\n".join(detailed_sections)
            
            # Return in Gradio's message format with metadata
            return {
                "role": "assistant",
                "content": main_response,
                "metadata": metadata
            }

        return preprocess, postprocess
    else:
        raise ValueError(f"Unsupported pipeline type: {pipeline}")

def create_advanced_demo():
    """
    Creates an advanced Gradio demo with model selection
    
    Usage:
        import ai_gradio
        from ai_gradio.providers.cua_gradio import create_advanced_demo
        
        # Create and launch the demo
        demo = create_advanced_demo()
        demo.launch()
    """
    
    # Check for API keys
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not openai_api_key and not anthropic_api_key:
        raise ValueError("Please set at least one of OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables")
    
    # Logo URLs
    logo_black = "https://github.com/trycua/cua/blob/main/img/logo_black.png?raw=true"  # For light theme
    logo_white = "https://github.com/trycua/cua/blob/main/img/logo_white.png?raw=true"  # For dark theme
    
    # Create a blocks-based interface for more customization
    with gr.Blocks(title="Advanced Computer-Use Agent Demo") as demo:
        with gr.Row():
            # Left column for settings
            with gr.Column(scale=1):
                # Add logo with HTML to support both light and dark themes
                gr.HTML(f"""
                <style>
                    /* Logo display based on theme */
                    .logo-container img.light-logo {{ display: block; }}
                    .logo-container img.dark-logo {{ display: none; }}
                    
                    /* Switch logos in dark mode */
                    @media (prefers-color-scheme: dark) {{
                        .logo-container img.light-logo {{ display: none; }}
                        .logo-container img.dark-logo {{ display: block; }}
                    }}
                    
                    /* Dark theme support for Gradio's theme toggle */
                    body.dark .logo-container img.light-logo {{ display: none; }}
                    body.dark .logo-container img.dark-logo {{ display: block; }}
                    
                    /* Light theme support for Gradio's theme toggle */
                    body:not(.dark) .logo-container img.light-logo {{ display: block; }}
                    body:not(.dark) .logo-container img.dark-logo {{ display: none; }}
                    
                    /* Logo sizing - smaller size */
                    .logo-container img {{
                        max-width: 80px;
                        height: auto;
                        margin: 0 auto 15px auto;
                    }}
                </style>
                
                <div class="logo-container" style="text-align: center;">
                    <img src="{logo_black}" alt="CUA Logo" class="light-logo" />
                    <img src="{logo_white}" alt="CUA Logo" class="dark-logo" />
                </div>
                """)
                
                # Add installation prerequisites at the top as a collapsible section
                with gr.Accordion("Prerequisites & Installation", open=False):
                    gr.Markdown("""
                    ## Prerequisites
                    
                    Before using the Computer-Use Agent, you need to set up the Lume daemon and pull the macOS VM image.
                    
                    ### 1. Install Lume daemon
                    
                    While a lume binary is included with Computer, we recommend installing the standalone version with brew, and starting the lume daemon service:
                    
                    ```bash
                    sudo /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/trycua/cua/main/libs/lume/scripts/install.sh)"
                    ```
                    
                    ### 2. Start the Lume daemon service
                    
                    In a separate terminal:
                    
                    ```bash
                    lume serve
                    ```
                    
                    ### 3. Pull the pre-built macOS image
                    
                    ```bash
                    lume pull macos-sequoia-cua:latest --no-cache
                    ```
                    
                    Initial download requires 80GB storage, but reduces to ~30GB after first run due to macOS's sparse file system.
                    
                    VMs are stored in `~/.lume`, and locally cached images are stored in `~/.lume/cache`.
                    
                    ### 4. Test the sandbox
                    
                    ```bash
                    lume run macos-sequoia-cua:latest
                    ```
                    
                    For more detailed instructions, visit the [CUA GitHub repository](https://github.com/trycua/cua).
                    """)
                
                # Prepare model choices based on available API keys
                openai_models = []
                anthropic_models = []
                omni_models = []
                
                if openai_api_key:
                    openai_models = [
                        "OpenAI: Computer-Use Preview"
                    ]
                    
                    omni_models += [
                        "OMNI: OpenAI GPT-4o",
                        "OMNI: OpenAI GPT-4.5-preview",
                    ]
                
                if anthropic_api_key:
                    anthropic_models = [
                        "Anthropic: Claude 3.7 Sonnet (20250219)",
                        "Anthropic: Claude 3.5 Sonnet (20240620)"
                    ]
                    
                    omni_models += [
                        "OMNI: Claude 3.7 Sonnet (20250219)",
                        "OMNI: Claude 3.5 Sonnet (20240620)"
                    ]
                
                # Configuration options
                agent_loop = gr.Dropdown(
                    choices=["OPENAI", "ANTHROPIC", "OMNI"],
                    label="Agent Loop",
                    value="OPENAI",
                    info="Select the agent loop provider"
                )
                
                # Function to filter models based on Agent Loop
                def filter_models(loop):
                    if loop == "OPENAI":
                        return gr.update(choices=openai_models, value=openai_models[0] if openai_models else None)
                    elif loop == "ANTHROPIC":
                        return gr.update(choices=anthropic_models, value=anthropic_models[0] if anthropic_models else None)
                    elif loop == "OMNI":
                        return gr.update(choices=omni_models, value=omni_models[0] if omni_models else None)
                    return gr.update(choices=[], value=None)
                
                # Model selection - will be updated when Agent Loop changes
                model_choice = gr.Dropdown(
                    choices=openai_models,
                    label="LLM Provider and Model",
                    value=openai_models[0] if openai_models else None,
                    info="Select the model appropriate for the Agent Loop",
                    allow_custom_value=True
                )
                
                logging_level = gr.Dropdown(
                    choices=["INFO", "DEBUG", "WARNING", "ERROR"],
                    label="Logging Level",
                    value="INFO",
                    info="Control the verbosity of agent logs"
                )
                
                save_trajectory = gr.Checkbox(
                    label="Save Trajectory",
                    value=True,
                    info="Save the agent's trajectory for detailed debugging"
                )
                
                recent_images = gr.Slider(
                    label="Recent Images",
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    info="Number of most recent images to keep in context"
                )
                
            # Right column for chat interface
            with gr.Column(scale=2):
                # Map the UI selection to the actual model and loop provider
                def get_model_and_loop(choice, loop_override=None):
                    """Convert model choice to model name and loop provider."""
                    if loop_override:
                        # If a loop provider is explicitly selected, use that
                        loop_provider = loop_override
                    else:
                        # Otherwise infer from the model choice
                        if choice.startswith("OpenAI:"):
                            loop_provider = "OPENAI"
                        elif choice.startswith("Anthropic:"):
                            loop_provider = "ANTHROPIC"
                        elif choice.startswith("OMNI:"):
                            loop_provider = "OMNI"
                        else:
                            # Default
                            loop_provider = "OPENAI"
                    
                    # Extract model name from the choice
                    if choice.startswith("OpenAI:"):
                        # Always map to computer_use_preview for OpenAI
                        return "computer_use_preview", loop_provider
                    elif choice.startswith("Anthropic:"):
                        if "3.7" in choice:
                            return "claude-3-7-sonnet-20250219", loop_provider
                        else:
                            return "claude-3-5-sonnet-20240620", loop_provider
                    elif choice.startswith("OMNI:"):
                        if "GPT-4o" in choice:
                            return "gpt-4o", loop_provider
                        elif "GPT-4.5" in choice:
                            return "gpt-4.5-preview", loop_provider
                        elif "3.7" in choice:
                            return "claude-3-7-sonnet-20250219", loop_provider
                        else:
                            return "claude-3-5-sonnet-20240620", loop_provider
                    else:
                        # Default
                        return "gpt-4-turbo", loop_provider
                    
                # Create the interface with the selected model
                def create_interface(model_choice, agent_loop_choice, logging_level, save_trajectory, recent_images):
                    model_name, loop_provider = get_model_and_loop(model_choice, agent_loop_choice)
                    
                    # We need to import here to avoid circular imports
                    import ai_gradio.providers
                    
                    # Create a wrapper that handles the async generator
                    async def wrapper_fn(message, history):
                        """Streaming wrapper that accumulates agent responses"""
                        # Get the CUA registry function
                        from ai_gradio.providers.cua_gradio import get_fn, get_interface_args
                        
                        # Set logging level
                        os.environ["CUA_LOGGING_LEVEL"] = logging_level
                        
                        # Get the API key based on the model
                        if loop_provider == "ANTHROPIC" or (loop_provider == "OMNI" and "claude" in model_name.lower()):
                            api_key = os.environ.get("ANTHROPIC_API_KEY")
                            if not api_key:
                                yield {"role": "assistant", "content": "ANTHROPIC_API_KEY environment variable is not set."}
                                return
                        else:
                            api_key = os.environ.get("OPENAI_API_KEY")
                            if not api_key:
                                yield {"role": "assistant", "content": "OPENAI_API_KEY environment variable is not set."}
                                return
                        
                        # Create the async generator
                        pipeline = "cua"
                        preprocess, postprocess = get_interface_args(pipeline)
                        
                        # Generate a session ID
                        session_id_for_interface = str(uuid.uuid4())
                        
                        # Tracking accumulated content
                        accumulated_content = ""
                        last_content = ""
                        has_real_content = False
                        
                        # Initialize the agent
                        try:
                            print(f"DEBUG: Creating agent with model={model_name}, loop={loop_provider}")
                            agent_fn = get_fn(
                                model_name, 
                                preprocess, 
                                postprocess, 
                                api_key,
                                save_trajectory=save_trajectory,
                                only_n_most_recent_images=recent_images,
                                session_id=session_id_for_interface,
                                loop_provider=loop_provider
                            )
                            
                            print(f"DEBUG: Initializing agent task with message: {message}")
                            # First message - only show minimal message
                            yield {"role": "assistant", "content": "Starting task..."}
                            
                            # Stream responses from the agent, accumulating content
                            async for response in agent_fn(message, history):
                                print(f"DEBUG: Got response: {response}")
                                
                                # Get the content from this response
                                current_content = response.get("content", "")
                                
                                # Skip empty or initial messages
                                if not current_content or current_content == "Starting task...":
                                    continue
                                
                                # Check if this is new content
                                if current_content != last_content and current_content.strip():
                                    # Only add if it's not already in accumulated content
                                    if current_content not in accumulated_content:
                                        # Add a separator if we already have content
                                        if accumulated_content:
                                            accumulated_content += "\n\n"
                                        accumulated_content += current_content
                                        
                                    last_content = current_content
                                    has_real_content = True
                                
                                # Yield the accumulated content with the original metadata
                                # This will completely replace the "Starting task..." message
                                if has_real_content:
                                    yield {
                                        "role": "assistant",
                                        "content": accumulated_content,
                                        "metadata": response.get("metadata", {})
                                    }
                            
                            # Only show completion message if we have no meaningful content
                            if not has_real_content:
                                yield {
                                    "role": "assistant", 
                                    "content": "Task completed."
                                }
                                
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            error_msg = f"Error: {str(e)}"
                            yield {"role": "assistant", "content": error_msg}
                    
                    # Create the interface with the wrapped function
                    return gr.ChatInterface(
                        fn=wrapper_fn,
                        description='Ask me to perform tasks in a virtual macOS environment.<br>Built with <a href="https://github.com/trycua/cua" target="_blank">github.com/trycua/cua</a>.',
                        examples=[
                            "Create a Python virtual environment, install pandas and matplotlib, then plot stock data",
                            "Open a PDF in Preview, add annotations, and save it as a compressed version",
                            "Open Safari, search for 'macOS automation tools', and save the first three results as bookmarks",
                            "Configure SSH keys and set up a connection to a remote server"
                        ]
                    )
                
                # Initial interface with default model
                interface = create_interface(
                    model_choice.value, 
                    agent_loop.value,
                    logging_level.value,
                    save_trajectory.value,
                    recent_images.value
                )
                
                # Update model choices when Agent Loop changes
                agent_loop.change(
                    fn=filter_models,
                    inputs=[agent_loop],
                    outputs=[model_choice]
                )
                
                # Update interface when parameters change
                for param in [model_choice, agent_loop, logging_level, save_trajectory, recent_images]:
                    param.change(
                        fn=lambda: gr.update(visible=False),
                        outputs=[interface]
                    ).then(
                        fn=create_interface,
                        inputs=[model_choice, agent_loop, logging_level, save_trajectory, recent_images],
                        outputs=[interface]
                    ).then(
                        fn=lambda: gr.update(visible=True),
                        outputs=[interface]
                    )
    
    return demo

def registry(
    name: str, 
    token: str | None = None, 
    advanced: bool = False,  # This parameter is ignored, always using advanced UI
    save_trajectory: bool = True,
    only_n_most_recent_images: int = 3,
    loop_provider: str = "OPENAI",
    session_id: str = "default",
    **kwargs
):
    """
    Create a Gradio interface for the Computer-Use Agent (CUA)
    
    Args:
        name: The model name in the format "model_name"
        token: Optional API key for the LLM provider
        advanced: Whether to create an advanced interface with model selection (always True now)
        save_trajectory: Whether to save the agent's trajectory
        only_n_most_recent_images: Number of most recent images to keep
        loop_provider: The loop provider to use (OPENAI, ANTHROPIC, OMNI)
        session_id: Unique identifier for the session
        **kwargs: Additional arguments to pass to the Gradio interface
    
    Returns:
        A Gradio interface
    
    Example:
        import gradio as gr
        import ai_gradio
        
        # Simple interface (still gets advanced UI)
        gr.load(
            name='cua:gpt-4-turbo',
            src=ai_gradio.registry
        ).launch()
        
        # Advanced interface with model selection
        gr.load(
            name='cua:gpt-4-turbo',
            src=ai_gradio.registry,
            advanced=True
        ).launch()
        
        # Conversation with custom settings (always multi-turn)
        gr.load(
            name='cua:gpt-4-turbo',
            src=ai_gradio.registry,
            save_trajectory=True,
            only_n_most_recent_images=5,
            loop_provider="OMNI"
        ).launch()
    """
    
    # Check if CUA is available
    if not CUA_AVAILABLE:
        error_msg = """
        Computer-Use Agent (CUA) is not available. Please install the required dependencies:
        
        pip install 'ai-gradio[cua]'
        
        Requires:
        - macOS with Apple Silicon
        - macOS 14 (Sonoma) or newer
        - Python 3.10+
        """
        
        def unavailable_fn(message, history):
            return error_msg
        
        interface = gr.ChatInterface(
            fn=unavailable_fn,
            title="Computer-Use Agent (Unavailable)",
            description=error_msg,
            **kwargs
        )
        
        return interface
    
    # Always return the advanced interface
    return create_advanced_demo() 