import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

if not os.environ.get("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY not set. Add it to your .env file.")

llm = init_chat_model("groq:openai/gpt-oss-20b")


if __name__ == "__main__":
    response = llm.invoke("how are you")
    print(response.content)