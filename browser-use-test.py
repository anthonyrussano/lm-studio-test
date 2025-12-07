#!/usr/bin/env python3
import asyncio
import os
from browser_use import Agent, Browser, BrowserConfig, ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LMSTUDIO_MODEL", "openai/gpt-oss-120b"),
    base_url=os.getenv("LMSTUDIO_BASE_URL", "http://pop-os:8630/v1"),
    api_key="lm-studio",
)

async def main():
    browser = Browser(
        config=BrowserConfig(
            browser_binary_path="/usr/bin/brave-browser",
            headless=True,  # Run without display
        )
    )
    
    agent = Agent(
        task="Go to github.com and find the browser-use repo, then tell me how many stars it has",
        llm=llm,
        browser=browser,
        use_vision=False,
    )
    
    result = await agent.run()
    print(f"Result: {result}")
    
    await browser.close()

if __name__ == "__main__":
    asyncio.run(main())