import asyncio
import aiohttp
import time

VLLM_URL = "http://localhost:8000/v1/completions"
PROMPT = "Explain in one sentence what PagedAttention is in vLLM."
CONCURRENT_REQUESTS = 20
TOTAL_REQUESTS = 50

async def send_request(session, request_id):
    payload = {
        "model": "microsoft/Phi-3-mini-4k-instruct",
        "prompt": PROMPT,
        "max_tokens": 50
    }
    start = time.time()
    try:
        async with session.post(VLLM_URL, json=payload) as resp:
            result = await resp.json()
            latency = time.time() - start
            print(f"Request {request_id:03d} completed in {latency:.2f}s")
            return latency
    except Exception as e:
        print(f"Request {request_id:03d} failed: {e}")
        return None

async def main():
    print(f"Sending {TOTAL_REQUESTS} requests with {CONCURRENT_REQUESTS} concurrent...")
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [send_request(session, i) for i in range(TOTAL_REQUESTS)]
        latencies = await asyncio.gather(*tasks)
    
    valid = [l for l in latencies if l is not None]
    print(f"\nCompleted: {len(valid)}/{TOTAL_REQUESTS}")
    print(f"Avg latency: {sum(valid)/len(valid):.2f}s")
    print(f"Min latency: {min(valid):.2f}s")
    print(f"Max latency: {max(valid):.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
