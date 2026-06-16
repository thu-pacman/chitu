import requests
import json
import threading
import hydra
from concurrent.futures import ThreadPoolExecutor, as_completed, wait

from chitu.schemas import ServeConfig
from chitu.utils import get_config_dir_path

req_nums = 2
max_completion_tokens = 256

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


def send_request(url: str, index: int):
    body = {
        "messages": msgs[index],
        "max_completion_tokens": max_completion_tokens,
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
                if len(choices := data["choices"]) > 0:
                    delta = choices[0]["delta"]
                    if delta.get("content", None):
                        tokens += 1
                        generated_text += delta["content"]
                    if delta.get("reasoning_content", None):
                        tokens += 1
                        generated_text += delta["reasoning_content"]

                with lock:
                    indices_received.append(index)
                print(f"Response received from request {index}", flush=True)

            return index, generated_text, reasoning_text, tokens
        else:
            print(f"Request failed with status code: {response.status_code}")


@hydra.main(
    version_base=None, config_path=get_config_dir_path(), config_name="serve_config"
)
def main(args: ServeConfig):
    print("Begin streaming test")
    all_text = {}
    all_reasoning_text = {}
    total_tokens = 0
    with ThreadPoolExecutor(max_workers=req_nums) as executor:
        url = f"http://{args.serve.host}:{args.serve.port}/v1/chat/completions"
        futures = []
        for i in range(req_nums):
            futures.append(executor.submit(send_request, url, i))
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


if __name__ == "__main__":
    main()
