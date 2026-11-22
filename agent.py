
import asyncio
import os
import uuid

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.mcp import MCPAdapter
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from llm import llm


load_dotenv()

# Online MCP server
# You can override this URL through your .env file.
MCP_URL = os.getenv(
    "MCP_URL",
    "https://mygpa.onrender.com/mcp"
)


SYSTEM_PROMPT = """
You are a GPA and CGPA assistant for a PUCIT BS(CS) student.

Rules you must always follow:
1. You do no arithmetic yourself. Every number in your answer must come from a
   tool call. Your only jobs are deciding which tool to call, with what
   arguments, and how to phrase the result.

2. You never invent marks, credit hours, a current CGPA, or a semester number.
   If you need one of these and do not have it, ask for it.

3. When something is missing, ask for one or two things at a time, not a long
   form of questions.

4. Math Deficiency courses (MD-001, MD-002) are non-credit pass/fail and are
   excluded from every GPA calculation. Every other course in the scheme
   counts, including Quran Translation courses at 0.5 credit hours each.

5. A required GPA above 4.0 is not a real answer. If required_gpa_for_target
   returns something above 4.0, do not report that number as the answer.
   Instead call get_remaining_credit_hours and required_gpa_for_target again
   over a longer horizon (one more semester, then one more), until the
   required GPA is at or below 4.0. Then tell the student which horizon
   actually works, and phrase it as advice.

6. Offer to save a report only after producing something worth keeping, such
   as a semester GPA, a projected CGPA, or a plan for reaching a target. Do
   not offer to save after a clarifying question or a single grade lookup.
   Never call save_report unless the student's most recent message explicitly
   asks you to save, download, or keep a report. A student asking a
   calculation question, expressing confusion, or giving you unclear input
   is never a request to save — do not treat it as one.
"""


def new_thread_id():
    """Generate a unique conversation ID."""
    return f"chat-{uuid.uuid4().hex[:8]}"


async def list_threads(memory):
    """Return all unique conversation IDs stored in SQLite."""
    thread_ids = []
    seen = set()

    async for checkpoint in memory.alist(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]

        if thread_id not in seen:
            seen.add(thread_id)
            thread_ids.append(thread_id)

    return thread_ids


async def create_gpa_agent(adapter, memory):
    """Create the GPA agent using tools from the MCP server."""
    tools = await adapter.list_tools()

    print(f"Connected to MCP server: {MCP_URL}")
    print(f"Loaded {len(tools)} MCP tools.")

    return create_agent(
        llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory
    )


def get_message_text(message):
    """Extract readable text from a LangChain message."""
    content = message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))

        return "".join(parts)

    return str(content)


async def run_turn(agent, user_text, thread_id):
    """Run one user message and stream the agent's response."""
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    inputs = {
        "messages": [
            HumanMessage(content=user_text)
        ]
    }

    print("\nAgent: ", end="", flush=True)

    try:
        async for chunk in agent.astream(
            inputs,
            config=config,
            stream_mode=["updates", "messages"],
            version="v2"
        ):
            if chunk["type"] == "messages":
                message_chunk, metadata = chunk["data"]

                if metadata.get("langgraph_node") == "model":
                    content = message_chunk.content

                    if isinstance(content, str):
                        print(content, end="", flush=True)

                    elif isinstance(content, list):
                        for block in content:
                            if (
                                isinstance(block, dict)
                                and block.get("type") == "text"
                            ):
                                print(
                                    block["text"],
                                    end="",
                                    flush=True
                                )

        print()

    except Exception as error:
        print(f"\nSomething went wrong: {error}")


async def show_history(agent, memory):
    """Display saved conversations."""
    thread_ids = await list_threads(memory)

    if not thread_ids:
        print("\nNo saved conversations yet.")
        return

    print("\n========== CONVERSATION HISTORY ==========")

    for index, thread_id in enumerate(thread_ids, start=1):
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        try:
            state = await agent.aget_state(config)
            messages = state.values.get("messages", [])

            if messages:
                last_message = get_message_text(messages[-1])
                print(f"\n{index}. {thread_id}")
                print(f"   Last message: {last_message[:100]}")
            else:
                print(f"\n{index}. {thread_id}")

        except Exception as error:
            print(f"\n{index}. {thread_id} (could not load: {error})")

    print("\n===========================================")


async def show_current_conversation(agent, thread_id):
    """Display messages from the current conversation."""
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:
        state = await agent.aget_state(config)
        messages = state.values.get("messages", [])

        if not messages:
            print("\nThis conversation is empty.")
            return

        print(f"\n========== {thread_id} ==========")

        for message in messages:
            role = message.type.upper()
            content = get_message_text(message)

            if content.strip():
                print(f"\n{role}: {content}")

        print("\n================================")

    except Exception as error:
        print(f"\nCould not load conversation: {error}")


async def handle_command(
    command,
    agent,
    memory,
    current_thread_id
):
    """Handle terminal commands."""

    if command == "/help":
        print("""
Available commands:

/new              Start a new conversation
/history          Show saved conversations
/show             Show current conversation
/switch <id>      Switch to another conversation
/forget            Delete current conversation
/forget <id>      Delete a specific conversation
/help              Show this help message
/exit              Exit the agent
""")
        return current_thread_id, False

    if command == "/new":
        current_thread_id = new_thread_id()
        print(f"\nStarted a new conversation: {current_thread_id}")
        return current_thread_id, False

    if command == "/history":
        await show_history(agent, memory)
        return current_thread_id, False

    if command == "/show":
        await show_current_conversation(agent, current_thread_id)
        return current_thread_id, False

    if command.startswith("/switch "):
        target_id = command[len("/switch "):].strip()
        thread_ids = await list_threads(memory)

        if target_id in thread_ids:
            current_thread_id = target_id
            print(f"\nSwitched to conversation: {target_id}")
        else:
            print("\nConversation not found.")

        return current_thread_id, False

    if command == "/forget" or command.startswith("/forget "):
        if command == "/forget":
            target_id = current_thread_id
        else:
            target_id = command[len("/forget "):].strip()

        thread_ids = await list_threads(memory)

        if target_id in thread_ids:
            await memory.adelete_thread(target_id)
            print(f"\nDeleted conversation: {target_id}")

            if target_id == current_thread_id:
                current_thread_id = new_thread_id()
                print(
                    f"Started a new conversation: {current_thread_id}"
                )
        else:
            print("\nConversation not found.")

        return current_thread_id, False

    if command == "/exit":
        print("\nGoodbye!")
        return current_thread_id, True

    print("\nUnknown command. Type /help for available commands.")
    return current_thread_id, False


async def main():
    print("===================================")
    print("          CGPA GUIDER")
    print("===================================")

    try:
        async with AsyncSqliteSaver.from_conn_string(
            "checkpoints.sqlite"
        ) as memory:

            async with MCPAdapter(MCP_URL) as adapter:

                agent = await create_gpa_agent(adapter, memory)

                current_thread_id = new_thread_id()

                print(f"\nConversation ID: {current_thread_id}")
                print("\nType /help to see available commands.")
                print("Type /exit to quit.")

                while True:
                    try:
                        user_text = input("\nYou: ").strip()

                    except (EOFError, KeyboardInterrupt):
                        print("\nGoodbye!")
                        break

                    if not user_text:
                        continue

                    if user_text.startswith("/"):
                        current_thread_id, should_exit = (
                            await handle_command(
                                user_text,
                                agent,
                                memory,
                                current_thread_id
                            )
                        )

                        if should_exit:
                            break

                        continue

                    await run_turn(
                        agent,
                        user_text,
                        current_thread_id
                    )

    except Exception as error:
        print(f"\nCould not start the CGPA Guider: {error}")


if __name__ == "__main__":
    asyncio.run(main())
