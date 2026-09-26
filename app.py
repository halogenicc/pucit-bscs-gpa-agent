import streamlit as st

from agent import create_gpa_agent, run_turn

st.set_page_config(
    page_title="PUCIT BSCS GPA Agent",
    page_icon="🎓",
    layout="centered",
)

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
    .bubble-row.user { justify-content: flex-end; }
    .bubble-row.assistant { justify-content: flex-start; }

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

st.title("🎓 PUCIT BSCS GPA Agent")
st.caption("Ask about your semester GPA, projected CGPA, or what you need to hit a target.")

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
        "- *My CGPA is 3.0 with 64 credit hours, I want to graduate with 3.4*\n"
        "- *What courses are in semester 5?*"
    )
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


def render_bubble(role, text):
    st.markdown(
        f'<div class="bubble-row {role}"><div class="bubble {role}">{text}</div></div>',
        unsafe_allow_html=True,
    )


if "agent" not in st.session_state:
    st.session_state.agent = create_gpa_agent()
    st.session_state.messages = []

if not st.session_state.messages:
    render_bubble(
        "assistant",
        "Hey! Tell me your marks, credit hours, or CGPA, and I'll work out "
        "your GPA, project your CGPA, or figure out what you need to hit a target.",
    )

for message in st.session_state.messages:
    if message.type == "human":
        render_bubble("user", message.text)
    elif message.type == "ai" and message.text.strip():
        render_bubble("assistant", message.text)

user_text = st.chat_input("Ask about your GPA or CGPA")
if user_text:
    render_bubble("user", user_text)
    placeholder = st.empty()
    with placeholder:
        with st.spinner("Calculating..."):
            st.session_state.messages = run_turn(
                st.session_state.agent, st.session_state.messages, user_text
            )
    placeholder.empty()
    render_bubble("assistant", st.session_state.messages[-1].text)