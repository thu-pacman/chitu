from safetensors import safe_open
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    tensors = {}
    with safe_open(
        # "/home/zly/Works/uniserving/exp/diffusers/watercolor_v1_sdxl.safetensors",
        # '/home/zly/Works/uniserving/exp/ikea_instructions_xl_v1_5.safetensors',
        args.model,
        framework="pt",
        device=0,
    ) as f:
        # print(*f.keys(), sep="\n")
        for k in f.keys():
            print(k, f.get_tensor(k).shape)
        # for k in f.keys():
        #     tensors[k] = f.get_tensor(k) # loads the full tensor given a key
