#!/usr/bin/env python3
import os
from openai import OpenAI

BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://10.32.25.175:1234/v1")
MODEL = os.getenv("LMSTUDIO_MODEL", "openai/gpt-oss-120b")
API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")  # LM Studio usually ignores it

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Start with a single system message and keep chat history in memory.
history = [{"role": "system", "content": "You are a DevSecOps Master."}]

print(f"Connected to {BASE_URL} using model {MODEL}")
print("Commands: /reset to clear history, /exit or /quit to leave.")

while True:
    try:
        user = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break

    if not user:
        continue
    if user.lower() in {"/exit", "/quit"}:
        break
    if user.lower() == "/reset":
        history = history[:1]
        print("History reset.")
        continue

    history.append({"role": "user", "content": user})
    print("Assistant: ", end="", flush=True)

    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=history,
            stream=True,
            temperature=0.7,
        )
        chunks = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                chunks.append(delta)
                print(delta, end="", flush=True)
        print()
        history.append({"role": "assistant", "content": "".join(chunks)})
    except Exception as exc:
        print(f"\n[error] {exc}")
        history.pop()  # remove last user message on failure
