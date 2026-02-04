import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

data_set = pd.read_parquet("metadata-large.parquet", engine="fastparquet")
plt.style.use("_mpl-gallery-nogrid")
x = data_set["width"]
y = data_set["height"]

X_refine = []
Y_refine = []
X_outbound = []
Y_outbound = []

for i in range(0, len(x)):
    item1 = x[i]
    item2 = y[i]
    if item1 <= 2048 and item2 <= 2048:
        X_refine.append(item1)
        Y_refine.append(item2)

    else:
        X_outbound.append(item1)
        Y_outbound.append(item2)

fig, ax = plt.subplots(figsize=(10, 10), facecolor="lightgray", layout="constrained")

h = ax.hist2d(X_refine, Y_refine, bins=60, cmin=0, norm=LogNorm())
# ax.set_facecolor("lightgray")
fig.colorbar(h[3], ax=ax)
fig.suptitle("Height-width Hist2D")
fig.supylabel("width")
fig.supxlabel("height")
fig.savefig("./many4.png")

plt.clf()

# token length statics
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "/home/wcz112/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/76d28af79639c28a79fa5c6c6468febd3490a37e",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)

token_list = data_set["prompt"]
mx_tokenlen = 0
cnt_list = []
count = 0
print(f"{len(token_list)} images need to be counted.")
for item in token_list:
    X_tmp = len((pipe.tokenizer(item)).input_ids)
    cnt_list.append(X_tmp)
    count += 1
    if count % 10000 == 0:
        print(f"{count} images completed.")
    if mx_tokenlen < X_tmp:
        mx_tokenlen = X_tmp

print(f"Max token length={mx_tokenlen}")
VL = [0] * (mx_tokenlen + 1)

for item in cnt_list:
    VL[item] += 1

Vl = np.array(VL)
TOTQ = len(x)
print("prompt token less than 77 (77 is the latent space length)")
print(np.array(VL[:78]) / TOTQ)

print("prompt token more than 77")
print(np.array(VL[78:]) / TOTQ)
