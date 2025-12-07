import os
import json
import asyncio
import aiohttp
from openai import OpenAI

try:
    from dotenv import load_dotenv
    if os.path.exists('.env'):
        load_dotenv()
except ImportError:
    pass

BASE_URL = os.getenv("LMSTUDIO_BASE_URL")
MODEL = os.getenv("LMSTUDIO_MODEL")
API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Async functions for parallel execution
async def fetch_url(url: str) -> dict:
    """Fetch a URL and return status and content length."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                content = await resp.text()
                return {
                    "url": url,
                    "status": resp.status,
                    "content_length": len(content),
                    "content_type": resp.headers.get("content-type", "unknown")
                }
    except Exception as e:
        return {"url": url, "error": str(e)}

async def dns_lookup(hostname: str) -> dict:
    """Perform DNS lookup."""
    import socket
    try:
        ip = socket.gethostbyname(hostname)
        return {"hostname": hostname, "ip": ip}
    except socket.gaierror as e:
        return {"hostname": hostname, "error": str(e)}

async def ping_host(host: str, count: int = 3) -> dict:
    """Ping a host."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ping", "-c", str(count), host,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        return {
            "host": host,
            "output": stdout.decode()[-500:],
            "success": proc.returncode == 0
        }
    except asyncio.TimeoutError:
        return {"host": host, "error": "timeout"}
    except Exception as e:
        return {"host": host, "error": str(e)}

# Sync wrappers for the OpenAI tool interface
def fetch_url_sync(url: str) -> dict:
    return asyncio.run(fetch_url(url))

def dns_lookup_sync(hostname: str) -> dict:
    return asyncio.run(dns_lookup(hostname))

def ping_host_sync(host: str, count: int = 3) -> dict:
    return asyncio.run(ping_host(host, count))

FUNCTIONS = {
    "fetch_url": fetch_url_sync,
    "dns_lookup": dns_lookup_sync,
    "ping_host": ping_host_sync,
}

# Async versions for parallel execution
ASYNC_FUNCTIONS = {
    "fetch_url": fetch_url,
    "dns_lookup": dns_lookup,
    "ping_host": ping_host,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a URL and get its status and content info",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "dns_lookup",
            "description": "Perform DNS lookup for a hostname",
            "parameters": {
                "type": "object",
                "properties": {
                    "hostname": {"type": "string", "description": "The hostname to lookup"}
                },
                "required": ["hostname"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ping_host",
            "description": "Ping a host to check connectivity",
            "parameters": {
                "type": "object",
                "properties": {
                    "host": {"type": "string", "description": "The host to ping"},
                    "count": {"type": "integer", "description": "Number of pings", "default": 3}
                },
                "required": ["host"]
            }
        }
    }
]

async def execute_tool_calls_parallel(tool_calls):
    """Execute multiple tool calls in parallel."""
    tasks = []
    for tc in tool_calls:
        func_name = tc.function.name
        args = json.loads(tc.function.arguments)
        if func_name in ASYNC_FUNCTIONS:
            tasks.append((tc.id, func_name, ASYNC_FUNCTIONS[func_name](**args)))
    
    results = []
    if tasks:
        completed = await asyncio.gather(*[t[2] for t in tasks], return_exceptions=True)
        for (tc_id, func_name, _), result in zip(tasks, completed):
            if isinstance(result, Exception):
                result = {"error": str(result)}
            results.append({
                "tool_call_id": tc_id,
                "role": "tool",
                "content": json.dumps(result)
            })
    return results

def run_network_agent(query: str):
    """Run the network diagnostics agent."""
    messages = [
        {"role": "system", "content": "You are a network diagnostics assistant. Use the tools to check connectivity, DNS, and fetch URLs. You can call multiple tools at once for efficiency."},
        {"role": "user", "content": query}
    ]
    
    for _ in range(5):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )
        
        msg = response.choices[0].message
        messages.append(msg)
        
        if not msg.tool_calls:
            return msg.content
        
        print(f"\nExecuting {len(msg.tool_calls)} tool calls in parallel...")
        for tc in msg.tool_calls:
            print(f"  → {tc.function.name}({tc.function.arguments})")
        
        results = asyncio.run(execute_tool_calls_parallel(msg.tool_calls))
        for r in results:
            print(f"  ← {r['content'][:100]}...")
        messages.extend(results)
    
    return "Max iterations reached"

if __name__ == "__main__":
    print("Network Diagnostics Assistant")
    print("Try: 'Check if google.com, github.com, and cloudflare.com are reachable'")
    print("=" * 60)
    
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ["/quit", "/exit"]:
            break
        if not query:
            continue
        
        result = run_network_agent(query)
        print(f"\nAssistant: {result}")