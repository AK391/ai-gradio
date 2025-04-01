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
else:
    # Provide helpful error message if libraries aren't installed
    raise ImportError(
        "The cua libraries could not be imported. Please install them with: "
        "pip install 'ai-gradio[cua]'"
    )

# Store the computer instance between chat turns
_computer_instances = {}

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
        import logging
        inputs = preprocess(message, history)
        
        try:
            # Determine provider and loop based on the inputs
            loop_provider_map = {
                "OPENAI": AgentLoop.OPENAI,
                "ANTHROPIC": AgentLoop.ANTHROPIC,
                "OMNI": AgentLoop.OMNI
            }
            
            # Get the agent loop
            agent_loop = loop_provider_map.get(loop_provider, AgentLoop.OPENAI)
            
            # Set up the provider and model based on the loop and model_name
            if agent_loop == AgentLoop.OPENAI:
                provider = LLMProvider.OPENAI
                # Map the model to the correct OpenAI CUA model name
                model_name_to_use = MODEL_MAPPINGS["openai"].get(model_name.lower(), MODEL_MAPPINGS["openai"]["default"])
            elif agent_loop == AgentLoop.ANTHROPIC:
                provider = LLMProvider.ANTHROPIC
                # Map the model to the correct Anthropic CUA model name
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
            
            # Let's define logging levels
            logging_level = os.environ.get("CUA_LOGGING_LEVEL", "INFO")
            numeric_level = getattr(logging, logging_level.upper(), logging.INFO)
            
            # Always use multi-turn conversation, reuse computer instance if available
            computer_key = f"{session_id}_{provider}_{model_name_to_use}"
            
            if computer_key in _computer_instances:
                macos_computer = _computer_instances[computer_key]
                agent = ComputerAgent(
                    computer=macos_computer,
                    loop=agent_loop,
                    model=LLM(
                        provider=provider,
                        name=model_name_to_use
                    ),
                    save_trajectory=save_trajectory,
                    only_n_most_recent_images=only_n_most_recent_images,
                    verbosity=numeric_level,
                    api_key=api_key
                )
                
                # Process the message through the CUA agent
                results = []
                async for result in agent.run(inputs["message"]):
                    # Process and collect detailed results
                    processed_result = process_agent_result(result)
                    results.append(processed_result)
                    
                    # Also yield intermediate results for streaming
                    yield postprocess([processed_result])
                
                # Return the final output for the chatbot
                yield postprocess(results)
            else:
                # Create a new computer instance for first message
                async with Computer(verbosity=numeric_level) as macos_computer:
                    # Store the computer instance for reuse in future messages
                    _computer_instances[computer_key] = macos_computer
                    
                    agent = ComputerAgent(
                        computer=macos_computer,
                        loop=agent_loop,
                        model=LLM(
                            provider=provider,
                            name=model_name_to_use,
                        ),
                        save_trajectory=save_trajectory,
                        only_n_most_recent_images=only_n_most_recent_images,
                        verbosity=numeric_level,
                        api_key=api_key
                    )
                    
                    # Process the message through the CUA agent
                    results = []
                    async for result in agent.run(inputs["message"]):
                        # Process and collect detailed results
                        processed_result = process_agent_result(result)
                        results.append(processed_result)
                        
                        # Also yield intermediate results for streaming
                        yield postprocess([processed_result])
                    
                    # Return the final output for the chatbot
                    yield postprocess(results)
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_message = f"Error: {str(e)}"
            
            # Clean up the computer instance in case of error
            computer_key = f"{session_id}_{provider}_{model_name_to_use}"
            if computer_key in _computer_instances:
                del _computer_instances[computer_key]
                
            yield {"role": "assistant", "content": error_message}
    
    return fn

def process_agent_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Process the detailed result from the agent into a standardized format.
    
    The response format aligns with the OpenAI Agent SDK specification.
    
    Args:
        result: The raw result from the agent
    
    Returns:
        A standardized format of the result
    """
    processed = {}
    
    # Debug print the full raw result
    print(f"DEBUG - Raw result: {type(result)}")
    
    # Basic information
    processed["id"] = result.get("id", "")
    processed["status"] = result.get("status", "")
    processed["model"] = result.get("model", "")
    
    # In OpenAI's Computer-Use Agent, the text field is an object with format property
    # But it doesn't actually contain the text content, so we need to extract it elsewhere
    text_obj = result.get("text", {})
    
    # For OpenAI Computer-Use, we need to synthesize text from output
    # Since there's often no direct text content
    if text_obj and isinstance(text_obj, dict) and "format" in text_obj and not text_obj.get("value", ""):
        synthesized_text = ""
        
        # Try to synthesize a text from output
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
                
                elif output.get("type") == "computer_call":
                    action = output.get("action", {})
                    action_type = action.get("type", "unknown")
                    status = output.get("status", "")
                    
                    # Create a descriptive text about the action
                    if action_type == "click":
                        button = action.get("button", "")
                        x = action.get("x", "")
                        y = action.get("y", "")
                        synthesized_text += f"Clicked {button} at position ({x}, {y}). {status}.\n"
                    elif action_type == "type":
                        text = action.get("text", "")
                        synthesized_text += f"Typed: {text}. {status}.\n"
                    elif action_type == "keypress":
                        key = action.get("key", "")
                        synthesized_text += f"Pressed key: {key}. {status}.\n"
                    else:
                        synthesized_text += f"Performed {action_type} action. {status}.\n"
        
        # If we couldn't create a meaningful text, use a generic message
        if not synthesized_text.strip():
            synthesized_text = "Working on your task..."
        
        processed["text"] = synthesized_text.strip()
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
                processed["text"] = "Working on your task..."
        else:
            processed["text"] = str(text_obj) if text_obj else "Working on your task..."
    
    # Extract usage information if available
    if "usage" in result:
        processed["usage"] = result["usage"]
    
    # Extract tool information
    if "tools" in result:
        processed["tools"] = result["tools"]
    
    # Extract outputs (reasoning, tool calls)
    if "output" in result:
        processed["output"] = result["output"]  # Preserve the original output structure
    
    # Debug the processed result
    print(f"DEBUG - Processed text: {processed.get('text', 'No text extracted')}")
    
    return processed

def get_interface_args(pipeline):
    if pipeline == "cua":
        def preprocess(message, history):
            return {"message": message}

        def postprocess(results):
            if not results:
                return {
                    "role": "assistant",
                    "content": "No results were returned from the computer agent."
                }
            
            # Get the final result
            final_result = results[-1] if results else {}
            
            # Extract the main text response
            main_response = final_result.get("text", "")
            
            # If this is a partial/streaming result at the beginning (just initializing)
            # Give a helpful status message
            if not main_response or main_response.strip() == "":
                if "output" in final_result and final_result["output"]:
                    # Show what's happening based on most recent output
                    most_recent_output = final_result["output"][-1]
                    
                    if most_recent_output.get("type") == "computer_call":
                        tool = most_recent_output.get("tool", "unknown")
                        status = most_recent_output.get("status", "")
                        
                        if "action" in most_recent_output:
                            action = most_recent_output["action"]
                            action_type = action.get("type", "")
                            main_response = f"Performing action: {action_type} ({status})"
                        else:
                            main_response = f"Working on task... ({status})"
                    
                    elif most_recent_output.get("type") == "reasoning":
                        main_response = "Thinking about how to approach this task..."
                else:
                    main_response = "Starting task execution..."
            
            # Prepare intermediate steps if available
            steps = []
            for i, result in enumerate(results[:-1], 1):
                text = result.get("text", "")
                if isinstance(text, str) and text.strip():
                    steps.append(f"Step {i}: {text}")
            
            # Collect any reasoning from the final result
            reasoning_details = []
            tool_call_details = []
            
            if "output" in final_result:
                for output in final_result.get("output", []):
                    if output.get("type") == "reasoning":
                        # Include full reasoning details
                        content = output.get("content", "")
                        output_id = output.get("id", "")
                        
                        reasoning_text = f"Reasoning: {content}"
                        
                        # Add summary if available
                        if "summary" in output:
                            for summary_item in output["summary"]:
                                if summary_item.get("type") == "summary_text":
                                    reasoning_text += f"\n↪ Summary: {summary_item.get('text', '')}"
                        
                        reasoning_details.append(reasoning_text)
                    
                    elif output.get("type") == "computer_call":
                        # Extract tool call details
                        tool = output.get("tool", "unknown")
                        call_id = output.get("call_id", "")
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
            
            # Add usage information if available
            usage_info = []
            if "usage" in final_result:
                usage = final_result["usage"]
                
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
            
            # Is this a streaming update or final result?
            is_streaming = len(results) == 1 and not steps
            
            # Prepare metadata for the rich response
            metadata = {}
            
            if not is_streaming:
                metadata["title"] = "🖥️ " + " → ".join(steps) if steps else "🖥️ Task in progress..."
            
            # Organize detailed information sections
            detailed_sections = []
            
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
    
    # Create a blocks-based interface for more customization
    with gr.Blocks(title="Advanced Computer-Use Agent Demo") as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("""
                # 🖥️ Advanced Computer-Use Agent
                
                This demo showcases the Computer-Use Agent (CUA) provider in ai-gradio.
                It creates a virtual macOS environment that the AI can fully control.
                
                ## Features
                - Run in isolated VM for security
                - Control applications like browsers, VS Code, Terminal
                - Take screenshots
                - Perform complex workflows
                
                ## Requirements
                - Mac with Apple Silicon (M1/M2/M3/M4)
                - macOS 14 (Sonoma) or newer
                - API key for OpenAI or Anthropic
                
                ## Try these examples:
                - "Search for a repository named trycua/cua on GitHub"
                - "Open VS Code and create a new Python file"
                - "Open Safari and go to apple.com"
                - "Take a screenshot of the desktop"
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
                
                # Configuration options - no accordion, with Agent Loop first
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
                        # Get the CUA registry function
                        from ai_gradio.providers.cua_gradio import get_fn, get_interface_args
                        
                        # Set logging level
                        os.environ["CUA_LOGGING_LEVEL"] = logging_level
                        
                        # Get the API key based on the model
                        if loop_provider == "ANTHROPIC" or (loop_provider == "OMNI" and "claude" in model_name.lower()):
                            api_key = os.environ.get("ANTHROPIC_API_KEY")
                            if not api_key:
                                yield {"role": "assistant", "content": "ANTHROPIC_API_KEY environment variable is not set."}
                                return  # End the generator without a value
                        else:
                            api_key = os.environ.get("OPENAI_API_KEY")
                            if not api_key:
                                yield {"role": "assistant", "content": "OPENAI_API_KEY environment variable is not set."}
                                return  # End the generator without a value
                        
                        # Create the async generator
                        pipeline = "cua"
                        preprocess, postprocess = get_interface_args(pipeline)
                        
                        # Generate a session ID only once for this interface instance
                        session_id_for_interface = str(uuid.uuid4())
                        
                        # Create the async generator with get_fn
                        generator = get_fn(
                            model_name, 
                            preprocess, 
                            postprocess, 
                            api_key,
                            save_trajectory=save_trajectory,
                            only_n_most_recent_images=recent_images,
                            session_id=session_id_for_interface,  # Use the fixed session ID
                            loop_provider=loop_provider
                        )
                        
                        # Use Gradio's streaming API by yielding each response
                        # This is the critical part for showing streaming in the UI
                        first_response = True
                        
                        # Keep track of all responses and a running transcript
                        all_responses = []
                        current_transcript = ""
                        actions_seen = set()  # Track actions we've already seen by ID
                        
                        # Initialize with starting message
                        yield {"role": "assistant", "content": "Starting your task..."}
                        
                        # Debug counter for tracking messages
                        message_counter = 0
                        
                        async for response in generator(message, history):
                            # Print for debugging
                            print(f"DEBUG: Received response #{message_counter}: {response.keys()}")
                            message_counter += 1
                            
                            # Remember all responses received
                            all_responses.append(response)
                            
                            # Extract action details to build a running transcript
                            new_content_added = False
                            
                            # Check if response has 'output' directly
                            outputs = response.get("output", [])
                            if not outputs and "metadata" in response:
                                # For processed responses that put output in metadata
                                metadata = response.get("metadata", {})
                                print(f"DEBUG: Found metadata: {metadata.keys() if metadata else 'none'}")
                                if "subtitle" in metadata:
                                    # Extract content from subtitle which contains processed output
                                    current_transcript = metadata["subtitle"]
                                    new_content_added = True
                            else:
                                # Process raw output directly
                                for output in outputs:
                                    # Only process each output once based on its ID
                                    output_id = output.get("id", "")
                                    if output_id in actions_seen:
                                        continue
                                    
                                    print(f"DEBUG: Processing new output: {output.get('type')}")
                                    actions_seen.add(output_id)
                                    
                                    if output.get("type") == "computer_call":
                                        action = output.get("action", {})
                                        action_type = action.get("type", "unknown")
                                        status = output.get("status", "")
                                        
                                        # Create message for this action
                                        action_msg = ""
                                        if action_type == "click":
                                            button = action.get("button", "")
                                            x = action.get("x", "")
                                            y = action.get("y", "")
                                            action_msg = f"• Clicked {button} at ({x}, {y}) - {status}\n"
                                        elif action_type == "type":
                                            text = action.get("text", "")
                                            action_msg = f"• Typed: \"{text}\" - {status}\n"
                                        elif action_type == "keypress":
                                            key = action.get("key", "")
                                            action_msg = f"• Pressed key: {key} - {status}\n"
                                        else:
                                            action_msg = f"• {action_type.capitalize()} action - {status}\n"
                                        
                                        current_transcript += action_msg
                                        new_content_added = True
                                    
                                    elif output.get("type") == "reasoning" and "summary" in output and output["summary"]:
                                        for summary_item in output["summary"]:
                                            if isinstance(summary_item, dict) and summary_item.get("text"):
                                                current_transcript += f"• {summary_item['text']}\n"
                                                new_content_added = True
                            
                            # Get any text content from the response itself
                            resp_content = response.get("content", "")
                            if resp_content and resp_content != "Working on task..." and "Working on your task" not in resp_content:
                                # If there's meaningful text in the response itself, add it
                                current_transcript += f"\n{resp_content}\n"
                                new_content_added = True
                            
                            # Update the UI with the current transcript if anything was added
                            if new_content_added:
                                print(f"DEBUG: Updating UI with new content (transcript length: {len(current_transcript)})")
                                yield {
                                    "role": "assistant", 
                                    "content": current_transcript
                                }
                                
                        # Final response - always provide a summary
                        if current_transcript:
                            yield {
                                "role": "assistant",
                                "content": "✅ Task completed!\n\n" + current_transcript
                            }
                    
                    # Create the interface with the wrapped function
                    return gr.ChatInterface(
                        fn=wrapper_fn,
                        description="Ask me to perform tasks in a virtual computer environment.",
                        examples=[
                            "Open Safari and go to helloworld.ai",
                            "Search GitHub for trycua/cua and open the first result", 
                            "Take a screenshot of the desktop",
                            "Open Visual Studio Code",
                            "Open a terminal and run ls -la"
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