# PUCIT BSCS GPA Agent

A conversational AI agent that answers GPA and CGPA questions for PUCIT BS(CS) students. All arithmetic is performed by tools, never by the language model, and the agent asks for missing information instead of guessing.

Assignment 3 extends the original agent with a remote Model Context Protocol (MCP) server, persistent conversation memory, streaming responses, and a Streamlit chat interface.

## Features

- GPA and CGPA calculations using dedicated tools.
- Semester-wise course and credit-hour lookup.
- Remote MCP server deployed on Render.
- LangChain agent connected to MCP tools.
- Persistent conversation history using SQLite and LangGraph checkpointing.
- Streaming AI responses.
- Terminal-based chat interface.
- Streamlit graphical chat interface.
- Conversation management commands.

## Architecture

```text
User
 |
 v
Streamlit UI / Terminal
 |
 v
LangChain Agent
 |
 +-----------> Groq LLM
 |
 +-----------> SQLite Memory
 |
 v
Remote MCP Server (Render)
 |
 v
GPA Calculation and Academic Tools
```

The MCP server is deployed remotely, while the agent and Streamlit application run locally and connect to it over HTTP.

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/halogenicc/pucit-bscs-gpa-agent.git
   cd pucit-bscs-gpa-agent
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

   On Windows:

   ```bash
   venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your Groq API key and MCP server URL:

   ```env
   GROQ_API_KEY=your_groq_api_key
   MCP_URL=https://mygpa.onrender.com/mcp
   ```

   Keep your API key private. Never commit `.env` to GitHub.

## Run It

**Terminal chat:**

```bash
python agent.py
```

**Streamlit GUI:**

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## MCP Server

The MCP server is built with FastMCP and deployed on Render.

- **Hosting:** Render
- **Transport:** HTTP
- **MCP endpoint:** `https://mygpa.onrender.com/mcp`
- **Start command:** `python server/server.py`

Test the remote connection and list available tools:

```bash
fastmcp list https://mygpa.onrender.com/mcp
```

Test the connection tool:

```bash
fastmcp call https://mygpa.onrender.com/mcp test_connection
```

## Conversation Commands

Both interfaces support conversation management.

| Command | Description |
|---|---|
| `/new` | Start a new conversation |
| `/show` | Display the current conversation |
| `/history` | List saved conversations |
| `/switch <thread_id>` | Switch to an existing conversation |
| `/forget` | Delete the current conversation |
| `/forget <thread_id>` | Delete a specific conversation |
| `/clear` | Start a fresh conversation without deleting previous history |
| `/help` | Display available commands |

The terminal interface also supports `/exit` to quit.

## Files

- `llm.py` — Groq language model configuration.
- `tools.py` — Seven GPA and academic tools.
- `agent.py` — System prompt, agent creation, MCP connection, streaming, and terminal conversation loop.
- `app.py` — Streamlit chat interface and conversation management.
- `server/server.py` — FastMCP server and tool registration.
- `requirements.txt` — Python dependencies.
- `.env.example` — Environment variable template.
- `.streamlit/config.toml` — Streamlit theme configuration.
- `checkpoints.sqlite` — Local SQLite conversation database (generated at runtime).

## Available Tools

The MCP server exposes the following tools:

1. `marks_to_grade_points` — Converts marks into grade points.
2. `calculate_semester_gpa` — Calculates semester GPA.
3. `calculate_new_cgpa` — Calculates updated CGPA.
4. `required_gpa_for_target` — Calculates the GPA required to reach a target CGPA.
5. `get_semester_courses` — Retrieves courses for a semester.
6. `get_remaining_credit_hours` — Calculates remaining credit hours.
7. `save_report` — Saves a GPA report.
8. `test_connection` — Checks MCP server connectivity.

## Memory and Persistence

Conversation memory is implemented using LangGraph's SQLite checkpointing.

- Each conversation has a unique thread ID.
- Conversation history is saved in `checkpoints.sqlite`.
- Previous conversations can be retrieved and resumed.
- Conversations can be deleted using `/forget`.

The SQLite database is stored locally and is not part of the remote MCP server deployment.

## Notes

- The agent never calculates numerical results itself; all GPA and CGPA arithmetic is handled by tools.
- The agent asks for missing information instead of inventing marks, credit hours, CGPA, or semester details.
- Math Deficiency courses (MD-001 and MD-002) are excluded from GPA calculations.
- Quran Translation courses count as 0.5 credit hours.
- If a required GPA exceeds 4.0, the agent extends the calculation horizon using remaining semesters instead of reporting an impossible target.
- The MCP server is hosted on Render; the agent and Streamlit GUI currently run locally.
- Do not submit `.env`, `venv/`, `__pycache__/`, or local SQLite database files.

## Testing

The application was tested for:

- MCP server connectivity and tool availability.
- GPA and CGPA calculations.
- Missing-information handling.
- Conversation memory and persistence.
- Conversation history, switching, and deletion.
- Streaming responses.
- Multiple consecutive messages.
- Error handling and MCP reconnection.

## Author

**Muhammad Bilal Khan**

GitHub: [@halogenicc](https://github.com/halogenicc)

Repository: [pucit-bscs-gpa-agent](https://github.com/halogenicc/pucit-bscs-gpa-agent)

---

Developed as part of the PUCIT BSCS Agent Development Assignment.