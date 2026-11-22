
import asyncio
import html
import streamlit as st

from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain.mcp import MCPAdapter
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import (
    create_gpa_agent,
    new_thread_id,
    list_threads,
    MCP_URL,
)

load_dotenv()

DB_PATH = "checkpoints.sqlite"


# ==========================================
# STREAMLIT CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PUCIT BSCS GPA Agent",
    page_icon="🎓",
    layout="centered",
)


# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = new_thread_id()

if "notice" not in st.session_state:
    st.session_state.notice = None


# ==========================================
# ORIGINAL UI STYLING
# ==========================================

st.markdown(
    """
    <style>
    .stApp,
    header[data-testid="stHeader"],
    div[data-testid*="Bottom"],
    div[data-testid*="stBottom"],
    footer,
    div[data-testid="stToolbar"] {
        background: #ffffff !important;
    }

    h1, h1 span, .stMarkdown h1 {
        color: #1c3f60 !important;
        font-weight: 800 !important;
        -webkit-text-fill-color: #1c3f60 !important;
    }

    div[data-testid="stCaptionContainer"],
    div[data-testid="stCaptionContainer"] p,
    .stCaption, .stCaption p {
        color: #35597e !important;
        opacity: 1 !important;
    }

    section[data-testid="stSidebar"] {
        background: #f2f6fb !important;
        border-right: 1px solid #d6e2ef;
    }

    section[data-testid="stSidebar"] * {
        color: #10131a !important;
    }

    div[data-testid="stChatInput"] {
        background: #f2f6fb !important;
        border: 1px solid #6FB3E8 !important;
        border-radius: 12px !important;
    }

    div[data-testid="stChatInput"] textarea {
        background: #f2f6fb !important;
        color: #10131a !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: #4a5568 !important;
    }

    button[kind="secondary"] {
        background: #6FB3E8 !important;
        color: #ffffff !important;
        border: none !important;
    }

    .bubble-row {
        display: flex;
        margin: 6px 0;
        width: 100%;
    }

    .bubble-row.user {
        justify-content: flex-end;
    }

    .bubble-row.assistant {
        justify-content: flex-start;
    }

    .bubble {
        max-width: 75%;
        padding: 12px 16px;
        font-size: 15px;
        line-height: 1.45;
        position: relative;
        word-wrap: break-word;
        white-space: pre-wrap;
        color: #000000 !important;
    }

    .bubble.assistant {
        background: #AFDCF7;
        border-radius: 18px 18px 18px 4px;
    }

    .bubble.user {
        background: #6FB3E8;
        border-radius: 18px 18px 4px 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# HELPER FUNCTIONS
# ==========================================

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


def convert_messages(messages):
    """Convert LangChain messages into Streamlit messages."""

    converted = []

    for message in messages:

        if message.type == "human":
            role = "user"

        elif message.type == "ai":
            role = "assistant"

        else:
            continue

        content = get_message_text(message)

        if content.strip():
            converted.append({
                "role": role,
                "content": content
            })

    return converted


def render_bubble(role, text):
    """Render a custom chat bubble."""

    safe_text = html.escape(str(text))

    st.markdown(
        f'<div class="bubble-row {role}">'
        f'<div class="bubble {role}">{safe_text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ==========================================
# STREAMING AGENT RESPONSE
# ==========================================

async def stream_agent_response(user_text, thread_id):
    """Connect to the remote MCP server and stream the response."""

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

    try:

        async with AsyncSqliteSaver.from_conn_string(
            DB_PATH
        ) as memory:

            async with MCPAdapter(MCP_URL) as adapter:

                agent = await create_gpa_agent(adapter, memory)

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

                                if content:
                                    yield content

                            elif isinstance(content, list):

                                for block in content:

                                    if (
                                        isinstance(block, dict)
                                        and block.get("type") == "text"
                                    ):

                                        text = block.get("text", "")

                                        if text:
                                            yield text

    except Exception as error:

        yield (
            "\n\nSomething went wrong while connecting "
            f"to the CGPA Guider: {error}"
        )


# ==========================================
# DISPLAY ASYNCHRONOUS STREAM
# ==========================================

async def display_streamed_response(
    user_text,
    thread_id,
    placeholder
):
    """Consume async response chunks and update the custom bubble."""

    response_text = ""

    async for chunk in stream_agent_response(user_text, thread_id):

        response_text += chunk

        safe_text = html.escape(response_text)

        placeholder.markdown(
            f'<div class="bubble-row assistant">'
            f'<div class="bubble assistant">{safe_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    return response_text


# ==========================================
# COMMAND HANDLER
# ==========================================

async def execute_command(command, current_thread_id):

    parts = command.strip().split(maxsplit=1)

    action = parts[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""

    # NEW CONVERSATION
    if action in ("/new", "/clear"):

        return {
            "action": "new",
            "thread_id": new_thread_id(),
            "messages": [],
            "output": "Started a new conversation."
        }

    async with AsyncSqliteSaver.from_conn_string(
        DB_PATH
    ) as memory:

        async with MCPAdapter(MCP_URL) as adapter:

            agent = await create_gpa_agent(adapter, memory)

            thread_ids = await list_threads(memory)

            # SHOW CURRENT CONVERSATION
            if action == "/show":

                config = {
                    "configurable": {
                        "thread_id": current_thread_id
                    }
                }

                state = await agent.aget_state(config)

                messages = state.values.get("messages", [])

                return {
                    "action": "show",
                    "messages": convert_messages(messages),
                    "output": "Current conversation retrieved from SQLite."
                }

            # SHOW SAVED CONVERSATIONS
            elif action == "/history":

                if not thread_ids:

                    return {
                        "action": "notice",
                        "output": "No saved conversations found."
                    }

                history = ["### Saved conversations"]

                for index, thread_id in enumerate(thread_ids, start=1):

                    config = {
                        "configurable": {
                            "thread_id": thread_id
                        }
                    }

                    state = await agent.aget_state(config)

                    messages = state.values.get("messages", [])

                    last_message = "Empty conversation"

                    for message in reversed(messages):

                        if message.type in ("human", "ai"):

                            last_message = get_message_text(message)
                            break

                    history.append(
                        f"**{index}. {thread_id}**\n\n"
                        f"Last message: {last_message[:120]}"
                    )

                return {
                    "action": "notice",
                    "output": "\n\n".join(history)
                }

            # SWITCH CONVERSATION
            elif action == "/switch":

                if not argument:

                    return {
                        "action": "notice",
                        "output": "Usage: /switch chat-id"
                    }

                if argument not in thread_ids:

                    return {
                        "action": "notice",
                        "output": f"Conversation {argument} not found."
                    }

                config = {
                    "configurable": {
                        "thread_id": argument
                    }
                }

                state = await agent.aget_state(config)

                messages = state.values.get("messages", [])

                return {
                    "action": "switch",
                    "thread_id": argument,
                    "messages": convert_messages(messages),
                    "output": f"Switched to {argument}."
                }

            # FORGET CONVERSATION
            elif action == "/forget":

                target_id = argument if argument else current_thread_id

                if target_id not in thread_ids:

                    return {
                        "action": "notice",
                        "output": "Conversation not found."
                    }

                await memory.adelete_thread(target_id)

                result = {
                    "action": "forget",
                    "output": f"Deleted conversation {target_id}."
                }

                if target_id == current_thread_id:

                    result["thread_id"] = new_thread_id()
                    result["messages"] = []

                return result

            return {
                "action": "notice",
                "output": "Unknown command. Type /help."
            }


def handle_command(command):

    try:

        result = asyncio.run(
            execute_command(
                command,
                st.session_state.thread_id
            )
        )

        action = result["action"]

        if action in ("new", "forget"):

            if "thread_id" in result:
                st.session_state.thread_id = result["thread_id"]

            st.session_state.messages = []

        elif action == "switch":

            st.session_state.thread_id = result["thread_id"]
            st.session_state.messages = result["messages"]

        elif action == "show":

            st.session_state.messages = result["messages"]

        st.session_state.notice = result["output"]

    except Exception as error:

        st.session_state.notice = f"Command failed: {error}"

    st.rerun()


# ==========================================
# PAGE HEADER
# ==========================================

st.title("🎓 PUCIT BSCS GPA Agent")

st.caption(
    "Ask about your semester GPA, projected CGPA, "
    "or what you need to hit a target."
)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("About")

    st.write(
        "This assistant calculates GPA and CGPA using the official "
        "PUCIT BS(CS) grading scheme. It never guesses — if it needs "
        "your marks, credit hours, or current CGPA, it will ask."
    )

    st.divider()

    st.subheader("Try asking")

    st.markdown(
        "- *I got 78, 85, 64 — what's my GPA?*\n"
        "- *My CGPA is 3.0 with 64 credit hours, "
        "I want to graduate with 3.4*\n"
        "- *What courses are in semester 5?*"
    )

    st.divider()

    st.subheader("Conversation")

    st.caption(
        f"Current ID: {st.session_state.thread_id}"
    )

    if st.button("🆕 New conversation", use_container_width=True):

        st.session_state.thread_id = new_thread_id()
        st.session_state.messages = []
        st.session_state.notice = "Started a new conversation."
        st.rerun()

    if st.button("🗑️ Clear conversation", use_container_width=True):

        st.session_state.thread_id = new_thread_id()
        st.session_state.messages = []
        st.session_state.notice = (
            "Chat cleared. Previous saved history remains in SQLite."
        )
        st.rerun()


# ==========================================
# COMMAND FEEDBACK
# ==========================================

if st.session_state.notice:

    st.info(st.session_state.notice)
    st.session_state.notice = None


# ==========================================
# WELCOME MESSAGE
# ==========================================

if not st.session_state.messages:

    render_bubble(
        "assistant",
        "Hey! Tell me your marks, credit hours, or CGPA, "
        "and I'll work out your GPA, project your CGPA, "
        "or figure out what you need to hit a target."
    )


# ==========================================
# DISPLAY CONVERSATION
# ==========================================

for message in st.session_state.messages:

    render_bubble(
        message["role"],
        message["content"]
    )


# ==========================================
# CHAT INPUT
# ==========================================

user_text = st.chat_input(
    "Ask about your GPA or CGPA. Type /help for commands."
)


if user_text:

    # COMMANDS
    if user_text.startswith("/"):

        if user_text.strip().lower() == "/help":

            st.session_state.notice = """
**Available commands**

- `/show` — Show current conversation
- `/history` — List saved conversations
- `/new` — Start a new conversation
- `/switch chat-id` — Switch conversations
- `/forget` — Delete current conversation
- `/forget chat-id` — Delete a specific conversation
- `/clear` — Clear current chat and start fresh
- `/help` — Show this help
"""

            st.rerun()

        else:

            handle_command(user_text)

    # NORMAL CHAT MESSAGE
    else:

        st.session_state.messages.append({
            "role": "user",
            "content": user_text
        })

        render_bubble("user", user_text)

        response_placeholder = st.empty()

        response_text = asyncio.run(
            display_streamed_response(
                user_text,
                st.session_state.thread_id,
                response_placeholder
            )
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text
        })
