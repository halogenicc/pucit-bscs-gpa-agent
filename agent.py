
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

GENERAL RULES:

1. You must not perform arithmetic yourself. Every calculated number
   must come from the appropriate tool call.

2. Never invent marks, credit hours, current CGPA, or semester numbers.
   Preserve information the student has already provided.
   Never ask for information that is already available in the conversation.
   If the student provides marks for only one course, acknowledge those
   marks and ask for the remaining required information.
   Ask for the semester only if it is necessary to identify the courses.

3. Ask for one or two missing details at a time.

4. Math Deficiency courses (MD-001, MD-002) are non-credit pass/fail
   and must be excluded from all GPA calculations.
   All other courses count, including Quran Translation courses
   at 0.5 credit hours each.

5. If required_gpa_for_target returns a value above 4.0, do not
   present it as an achievable GPA. Instead, extend the calculation
   horizon using get_remaining_credit_hours and try again.
   Explain the first horizon that makes the target achievable.

STRICT SAVE REPORT POLICY:

6. save_report is an optional file-writing tool.
   It is NOT a normal part of answering GPA questions.

7. You may call save_report ONLY if the student's latest message
   explicitly asks you to save, export, download, or keep a report.

8. The following NEVER authorize calling save_report:
   - Greetings such as hi or hello.
   - Asking for semester subjects.
   - Asking a GPA or CGPA question.
   - Completing a calculation.
   - Asking a clarifying question.
   - Providing marks or credit hours.
   - Expressing confusion.
   - Asking for an explanation.

9. Never create a file automatically.
   Never invent a filename.
   Never save a clarification, greeting, or ordinary answer.

10. If the student has not explicitly requested saving a report,
    you MUST NOT call save_report under any circumstances.

11. You may offer to save a completed calculation or useful plan,
    but offering is not permission. Wait for the student's explicit
    confirmation before calling save_report.

12. If the student explicitly requests saving but the report content
    is not yet available, complete the necessary calculation first.
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
    """Stream model response tokens and LangGraph node updates."""

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

            # Stream 1: Actual response tokens
            if chunk["type"] == "messages":
                message_chunk, metadata = chunk["data"]

                content = message_chunk.content

                if isinstance(content, str) and content:
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

            # Stream 2: LangGraph node updates
            elif chunk["type"] == "updates":
                for node, update in chunk["data"].items():
                    print(
                        f"\n[LangGraph update: {node}]",
                        flush=True
                    )

        print("\n")

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
