import argparse
import os
import sys

# We import the components from openhands.sdk as requested
try:
    from openhands.sdk import LLM, Agent, Conversation, Tool
except ImportError:
    pass

def main():
    parser = argparse.ArgumentParser(description="Standalone Python worker script using OpenHands SDK")
    parser.add_argument("--prompt-file", required=True, help="Path to the file containing the prompt")
    parser.add_argument("--workspace", default=os.getcwd(), help="Workspace directory for the agent")
    args = parser.parse_args()

    # Read prompt from file
    if not os.path.exists(args.prompt_file):
        print(f"Error: Prompt file '{args.prompt_file}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt_text = f.read()
    except Exception as e:
        print(f"Error reading prompt file: {e}", file=sys.stderr)
        sys.exit(1)

    # Prepend strict system directive
    system_directive = (
        "You are an autonomous software engineer. Complete the task below. "
        "You MUST use the FileEditorTool to write or modify files directly on the disk. "
        "Do not just output the code in chat. "
        "Reply with 'ALL_TASKS_COMPLETED' only when you have finished writing all required files to disk.\n\n"
    )
    full_prompt = f"{system_directive}{prompt_text}"

    # Read environment variables
    llm_api_key = os.environ.get("LLM_API_KEY")
    llm_model = os.environ.get("LLM_MODEL")
    llm_base_url = os.environ.get("LLM_BASE_URL")

    if not llm_api_key:
        print("Error: Missing required environment variable LLM_API_KEY.", file=sys.stderr)
        sys.exit(1)

    if not llm_model:
        print("Error: Missing required environment variable LLM_MODEL.", file=sys.stderr)
        sys.exit(1)

    if 'LLM' not in globals():
        print("Error: openhands.sdk is not installed or available.", file=sys.stderr)
        sys.exit(1)

    # Initialize LLM
    llm_kwargs = {
        "model": llm_model,
        "api_key": llm_api_key,
    }
    if llm_base_url:
        llm_kwargs["base_url"] = llm_base_url

    llm = LLM(**llm_kwargs)

    # Equip the agent with FileEditorTool and TerminalTool
    # OpenHands tools are typically automatically provided, but to strictly adhere:
    try:
        from openhands.events.tool import Tool
        tools = [Tool(name="FileEditorTool"), Tool(name="TerminalTool")]
        agent = Agent(llm=llm, tools=tools)
    except ImportError:
        # Fallback to standard agent creation if tools can't be imported this way
        agent = Agent(llm=llm)

    # Start a Conversation in the specified workspace
    conversation = Conversation(
        agent=agent,
        workspace=args.workspace,
    )

    try:
        # Send the initial message and start the autonomous run loop
        conversation.send_message(full_prompt)
        conversation.run()
    except Exception as e:
        print(f"Error during conversation execution: {e}", file=sys.stderr)
        sys.exit(1)

    # Exit with code 0 upon completion
    sys.exit(0)

if __name__ == "__main__":
    main()
