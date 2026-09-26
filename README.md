# PUCIT BSCS GPA Agent

A conversational agent that answers GPA/CGPA questions for a PUCIT BS(CS)
student. All arithmetic is done by tools (never by the model), and the agent
asks for anything it doesn't know instead of guessing.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and add your Groq API key:
   ```
   GROQ_API_KEY=your_key_here
   ```

## Run it

- Terminal chat: `python agent.py` (type `exit` to quit)
- Streamlit GUI (bonus): `streamlit run app.py`

## Files

- `llm.py` — Groq model setup
- `tools.py` — the 7 tools (marks→grade points, semester GPA, CGPA
  projection, required-GPA-for-target, semester course lookup, remaining
  credit hours, save report)
- `agent.py` — system prompt, agent creation, and the conversation loop
- `app.py` — Streamlit chat UI (bonus)
- `.streamlit/config.toml` — theme for the Streamlit UI
- `.env.example` — template for the required environment variable
- `requirements.txt` — Python dependencies

## Notes

- The agent never computes numbers itself — every figure comes from a tool
  call.
- Math Deficiency courses (MD-001, MD-002) are excluded from GPA; Quran
  Translation courses count at 0.5 credit hours.
- If a required GPA comes out above 4.0, the agent widens the horizon
  (checks further semesters) instead of reporting an impossible number.
- Do not submit `.env`, `venv/`, or `__pycache__/`.
