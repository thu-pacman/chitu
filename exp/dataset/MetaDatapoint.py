import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap

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

h = ax.hist2d(X_refine, Y_refine, bins=60, cmin=0)
# ax.set_facecolor("lightgray")
fig.colorbar(h[3], ax=ax)
fig.suptitle("Height-width Hist2D")
fig.supylabel("width")
fig.supxlabel("height")
fig.savefig("./many5.png")

plt.clf()
