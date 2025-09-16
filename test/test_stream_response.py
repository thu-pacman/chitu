import requests
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, wait

if len(sys.argv) == 4:
    url = sys.argv[1]
    req_nums = int(sys.argv[2])
    max_tokens = int(sys.argv[3])
else:
    print(
        f"Usage: {sys.argv[0]} <url> <req_nums> <max_tokens>. \n"
        f"Example: python3 {sys.argv[0]} http://localhost:25123/v1/chat/completions 1 256"
    )
    sys.exit(1)

headers = {"Content-Type": "application/json"}
msgs = [
    [{"role": "user", "content": "宫保鸡丁怎么做?"}],
    [{"role": "user", "content": "what is the recipe of Kung Pao chicken?"}],
    [{"role": "user", "content": "怎么写程序?"}],
    [{"role": "user", "content": "飞机在对流层还是平流层飞?"}],
    [{"role": "user", "content": "怎么避免加班?"}],
    [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
]

lock = threading.Lock()
indices_received = []


def send_request(index: int):
    body = {
        "messages": msgs[index],
        "max_tokens": max_tokens,
        "stream": True,
        "min_batch_size": req_nums,
    }
    generated_text = ""
    reasoning_text = ""
    with requests.post(url, json=body, stream=True) as response:
        if response.status_code == 200:
            tokens = 0
            for chunk in response.iter_lines():
                if not chunk:
                    continue

                stem = "data: "
                chunk = chunk[len(stem) :]
                if chunk == b"[DONE]":
                    continue
                data = json.loads(chunk)
                delta = data["choices"][0]["delta"]
                if delta.get("content", None):
                    tokens += 1
                    generated_text += delta["content"]
                if delta.get("reasoning_content", None):
                    tokens += 1
                    generated_text += delta["reasoning_content"]

                with lock:
                    indices_received.append(index)
                print(f"Response received from request {index}", flush=True)

            return (
                index,
                generated_text,
                reasoning_text,
                tokens,
            )
        else:
            print(f"Request failed with status code: {response.status_code}")


all_text = {}
all_reasoning_text = {}
total_tokens = 0
with ThreadPoolExecutor(max_workers=req_nums) as executor:
    futures = []
    for i in range(req_nums):
        futures.append(executor.submit(send_request, i))
    for future in as_completed(futures):
        result = future.result()
        total_tokens += result[3]
        text = result[1].replace("\n", "")
        reasoning_text = result[2].replace("\n", "")
        all_text[result[0]] = text
        all_reasoning_text[result[0]] = reasoning_text
print(
    f"Response received order (interleaved indices are expected for true concurrency): {indices_received}"
)
print("All responses:")
for i in range(req_nums):
    print(f"Response {i}: {all_text[i]}")
    print(f"Reasoning {i}: {all_reasoning_text[i]}")
print(f"Total tokens: {total_tokens}")
