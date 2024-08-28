from concurrent.futures import ThreadPoolExecutor, as_completed, wait
import requests, json, sys, time, random

random.seed(2512)

url = "http://127.0.0.1:2512/v1/chat/completions"  # cinfer

headers = {"Content-Type": "application/json"}


def gen_req_id(len=8):
    random_number = random.getrandbits(len * 4)
    hex_string = f"{random_number:0{len}x}"
    return hex_string


msgs = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "宫保鸡丁怎么做"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "show some Emoji"},
    ],
    # 3
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "北京可以参观哪里"},
        {
            "role": "assistant",
            "content": """北京作为中国的首都，是历史悠久、文化丰富的城市，拥有众多值得参观的景点。以下是一些北京著名的旅游景点和文化地标：
            1. **故宫（紫禁城）**：是中国明清两代的皇家宫殿，也是世界上现存规模最大、保存最为完整的木质结构古建筑之一。
            2. **天安门广场**：位于北京的中心，是世界上最大的城市广场之一，周围有天安门城楼、人民英雄纪念碑等。
            3. **长城**：北京段长城包括八达岭长城、慕田峪长城、司马台长城等，是世界文化遗产，象征着中国的坚韧和不屈。
            这些地方不仅展示了北京的传统文化和历史，也是了解中国和世界文化交流的重要窗口。""",
        },
        {"role": "user", "content": "第3个地方好在哪"},
    ],
    # 4
    [
        {"role": "user", "content": "I am going to Paris, what should I see?"},
        {
            "role": "assistant",
            "content": """\
                Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

                1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
                2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
                3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

                These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
        },
        {"role": "user", "content": "What is so great about #1?"},
    ],
]

stream = False

req_nums = 1
msg_id = None

if len(sys.argv) > 2:
    msg_id = int(sys.argv[2])


if len(sys.argv) > 1:
    stream = True
    # index = int(sys.argv[1]) % len(msgs)
    req_nums = int(sys.argv[1])


def send_request(index: int):
    body = {
        "model": "/home/ss/models/Qwen2-7B-Instruct",
        "messages": msgs[index % len(msgs) if msg_id is None else msg_id],
        "max_tokens": 100,
        "stream": stream,
        "temperature": 1,
    }

    generated_text = ""
    start_time = time.monotonic()
    with requests.post(url, json=body, stream=True) as response:
        if response.status_code == 200:
            tokens = 0
            for chunk in response.iter_lines():
                if not chunk:
                    continue
                # print(f"{index}: {chunk}")
                if stream:
                    stem = "data: "
                    chunk = chunk[len(stem) :]
                    if chunk == b"[DONE]":
                        continue
                    data = json.loads(chunk)
                    delta = data["choices"][0]["delta"]
                    if delta.get("content", None):
                        tokens += 1
                        generated_text += delta["content"]

            end_time = time.monotonic()
            duration = end_time - start_time
            tps = tokens / duration
            # print(generated_text)
            return (index, start_time, end_time, duration, generated_text, tps)
        else:
            print(f"Request failed with status code: {response.status_code}")


with ThreadPoolExecutor(max_workers=req_nums) as executor:
    futures = []
    for i in range(req_nums):
        futures.append(executor.submit(send_request, i))
        time.sleep(0.5)
    for future in as_completed(futures):
        result = future.result()
        text = result[4][:35].replace("\n", "")
        print(
            f"Index:{result[0]:2d}, start:{result[1]:.4f}, duration:{result[3]:.4f}, tps:{result[5]:.4f}, text:'{text}'"
        )
