# %%
import json
from dataclasses import dataclass, asdict
from dataclasses_json import dataclass_json
from typing import Optional
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import numpy as np


# %%
@dataclass_json
@dataclass
class MetaItem:
    Size: Optional[str] = None
    steps: Optional[int] = None
    prompt: Optional[str] = None
    negativePrompt: Optional[str] = None


@dataclass_json
@dataclass
class ImageItem:
    id: int
    width: int
    height: int
    nsfw: bool
    meta: Optional[MetaItem] = None


@dataclass_json
@dataclass
class DataSet:
    items: list[ImageItem]


def read_dataset(start=0, end=1):
    ret = []
    for i in range(start, end):
        with open(f"data_civitai/civitai_database/231112_p{i}.json") as f:
            a = "".join(f.readlines())
        data = DataSet.from_json(a)
        ret += data.items
    return ret


for iend in range(10, 101, 10):
    imageItems = read_dataset(0, iend)
    print(len(imageItems))

    # %%
    # for item in dataset.items:
    #     item.to_dict()
    imageItems[0].to_dict()

    # %%
    df = pd.json_normalize([asdict(item) for item in imageItems])
    print("Before filter #:", len(df))
    df = df[df["nsfw"] == False]
    print("After NSFW filter #:", len(df))
    df = df[df["meta.Size"].isna() == False]
    print("After Meta shape filter #:", len(df))
    # df = df.iloc[:1000]
    df

    # %%
    def draw_heatmap(x, y):
        X_refine = []
        Y_refine = []
        X_outbound = []
        Y_outbound = []

        print(f"{x.min()=} {y.min()=} {x.max()=} {y.max()=}")
        xy_min = 256
        xy_max = 1024
        for item1, item2 in zip(x, y):
            # item1 = x[i]
            # item2 = y[i]
            if xy_min <= item1 <= xy_max and xy_min <= item2 <= xy_max:
                X_refine.append(item1)
                Y_refine.append(item2)
            else:
                X_outbound.append(item1)
                Y_outbound.append(item2)

        fig, ax = plt.subplots(figsize=(4, 3), layout="constrained")

        # h=ax.hist2d(X_refine,Y_refine,bins=60,cmin=0) # ,norm=LogNorm())

        h = ax.hist2d(
            # x, y,
            X_refine,
            Y_refine,
            range=[[xy_min, xy_max]] * 2,
            bins=(xy_max - xy_min) // 64,
            cmin=1,
            vmin=1,  #  vmax = 1000/100*iend,
            # cmap="twilight"
            # cmap="GnBu"
            cmap="Blues",
        )  # ,norm=LogNorm())
        max_cnt = np.nanmax(h[0])
        print("max=", max_cnt, max_cnt / len(X_refine))
        # ax.set_facecolor("lightgray")
        # Set colorbar range and labels
        # fig.colorbar(h[3], ax=ax)
        fig.colorbar(
            h[3],
            ax=ax,
            format=matplotlib.ticker.FuncFormatter(
                lambda x, pos: f"{x/len(X_refine)*100:.1f}"
            ),
        )
        # fig.suptitle("Height-width Hist2D")
        fig.supylabel("Width")
        fig.supxlabel("Height")
        # fig.savefig("./many4.png")
        print(f"{X_outbound=}\n{Y_outbound=}")
        plt.show()
        plt.savefig(f"figures/civitai_sfw_{0}_{iend}.png")

    # %%
    meta_shape = df["meta.Size"][~df["meta.Size"].isna()]
    # meta_shape.str.split('x').str[0].astype(int).hist(bins=100)
    # draw_heatmap(meta_shape.str.split('x').str[0].astype(int), meta_shape.str.split('x').str[1].astype(int))
    meta_x = meta_shape.str.split("x").str[0].astype(int)
    meta_y = meta_shape.str.split("x").str[1].astype(int)
    draw_heatmap(meta_x, meta_y)

    # %%
    # draw_heatmap(df["width"], df["height"])
