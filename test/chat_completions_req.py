import requests, json

url = "http://127.0.0.1:2512/v1/chat/completions"  # cinfer

stream = True

headers = {"Content-Type": "application/json"}


message = [
    {
        "role": "system",
        "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair.",
    },
    {"role": "user", "content": "nihaonihaonihao"},
]

body = {
    "model": "/home/ss/models/Qwen2-7B-Instruct",
    "messages": message,
    "stream": stream,
}

generated_text = ""
with requests.post(url, json=body, stream=True) as response:
    if response.status_code == 200:
        for chunk in response.iter_lines():
            if not chunk:
                continue
            print(chunk)
            if stream:
                stem = "data: "
                chunk = chunk[len(stem) :]
                if chunk == b"[DONE]":
                    continue
                data = json.loads(chunk)
                delta = data["choices"][0]["delta"]
                if delta.get("content", None):
                    generated_text += delta["content"]
        print(generated_text)
    else:
        print(f"Request failed with status code: {response.status_code}")
