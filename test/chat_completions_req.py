import requests, json

url = "http://127.0.0.1:21002/v1/chat/completions"

headers = {
    "Content-Type": "application/json"
}
message = [
    {
        "role": "system",
        "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."
    },
    {
        "role": "user",
        "content": "Compose a poem that explains the concept of recursion in programming."
    }
]

body = {
    'message': message
}

generated_text = ""
with requests.post(url, json=body, stream=True) as response:
    if response.status_code == 200:
        for chunk in response.iter_lines():
            if not chunk:
                continue
            print(chunk)
            stem = "data: "
            chunk = chunk[len(stem) :]
            if chunk == b"[DONE]":
                continue
            # tokens_num += 1
            data = json.loads(chunk)
            delta = data["choices"][0]["delta"]
            if delta.get("content", None):
                generated_text += delta["content"]
        print(generated_text)
    else:
        print(f"Request failed with status code: {response.status_code}")