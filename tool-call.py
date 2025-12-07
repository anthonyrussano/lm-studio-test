import os
from openai import OpenAI
# Try to load from .env file if it exists
try:
    from dotenv import load_dotenv
    if os.path.exists('.env'):
        load_dotenv()
except ImportError:
    # dotenv is not installed, that's fine for GitHub Actions
    pass

BASE_URL = os.getenv("LMSTUDIO_BASE_URL")
MODEL = os.getenv("LMSTUDIO_MODEL")
API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")

# Connect to LM Studio
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Define a simple function
def say_hello(name: str) -> str:
    print(f"Hello, {name}!")

# Tell the AI about our function
tools = [
    {
        "type": "function",
        "function": {
            "name": "say_hello",
            "description": "Says hello to someone",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The person's name"
                    }
                },
                "required": ["name"]
            }
        }
    }
]

# Ask the AI to use our function
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Can you say hello to Bob the Builder?"}],
    tools=tools
)

# Get the name the AI wants to use a tool to say hello to
# (Assumes the AI has requested a tool call and that tool call is say_hello)
tool_call = response.choices[0].message.tool_calls[0]
name = eval(tool_call.function.arguments)["name"]

# Actually call the say_hello function
say_hello(name) # Prints: Hello, Bob the Builder!

