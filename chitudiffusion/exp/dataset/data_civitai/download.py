import requests
import json
import datetime
import time

# url = 'https://civitai.com/api/v1/images'
# for i in range(100):

url = "https://civitai.com/api/v1/images?cursor=3922"
for i in range(38, 100):
    print(i, datetime.datetime.now(), url)
    try:
        r = requests.get(url, stream=True)
        j = json.loads(r.text)
    except Exception:
        print("Response:", r.text)
        raise
    with open(f"231112_p{i}.json", "w") as f:
        f.write(r.content.decode())
    url = j["metadata"]["nextPage"]
    time.sleep(60)
