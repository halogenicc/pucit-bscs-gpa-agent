from langchain.agents import create_agent
from langchain.messages import HumanMessage

from llm import llm
from tools import TOOLS

SYSTEM_PROMPT = """You are a GPA and CGPA assistant for a PUCIT BS(CS) student.

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
   Never call save_report unless the student asks you to save.
"""

def create_gpa_agent():
    return create_agent(llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT)

def run_turn(agent, messages, user_text):
    messages = messages + [HumanMessage(user_text)]
    try:
        result = agent.invoke({"messages": messages})
    except Exception as error:
        print(f"Something went wrong: {error}")
        return messages
    return result["messages"]


if __name__ == "__main__":
    agent = create_gpa_agent()
    messages = []
    print("GPA Agent ready. Type 'exit' to quit.")
    while True:
        user_text = input("You: ")
        if user_text.strip().lower() == "exit":
            break
        messages = run_turn(agent, messages, user_text)
        print("Agent:", messages[-1].text)