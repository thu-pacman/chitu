import pandas as pd
import numpy as np

import argparse


data_set = pd.read_parquet("metadata-large.parquet", engine="fastparquet")

token_list = data_set["prompt"]

to_be_add = []
lst_seq = []
child_list = [[]]
child_context = [""]
node_count = [0]
length = [0]
pre_node = [-1]
cnts = 0


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False

    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()
parser.add_argument("calc_conceseq", type=str2bool)
parser.add_argument("build_trie", type=str2bool)


def Compare_ret():
    common_counts = 0
    for item in to_be_add:
        try:
            id = lst_seq.index(item)
            common_counts += 1
        except:
            pass
    if len(to_be_add) == 0:
        return 0
    return 1.0 * common_counts / len(to_be_add)


def Add_trie():
    global cnts
    tmp = 0
    for step in to_be_add:
        flag = -1
        node_count[tmp] += 1
        for id in range(len(child_list[tmp])):
            if step == child_context[child_list[tmp][id]]:
                flag = child_list[tmp][id]
                break
        if flag == -1:
            cnts = cnts + 1
            child_list[tmp].append(cnts)
            # child_context[tmp].append(to_be_add[step])
            node_count.append(0)
            child_list.append([])
            child_context.append(step)
            length.append(length[tmp] + 1)
            pre_node.append(tmp)
            flag = cnts
        tmp = flag


if __name__ == "__main__":
    args = parser.parse_args()
    prompt_count = 0
    sum = 0
    start = 0
    lens = 0
    common_token = 0
    for item in token_list:
        to_be_add = item.split()
        prompt_count += 1
        sum += len(to_be_add)
        if args.build_trie:
            Add_trie()
            if prompt_count % 100000 == 0:
                print(f"have already done {prompt_count}/{14000000} of all prompts")
                print(f"total node counts is {cnts}")
                print(f"total word counts is {sum}")
        if args.calc_conceseq:
            percent = Compare_ret()
            id = 0
            while (
                id < len(to_be_add)
                and id < len(lst_seq)
                and to_be_add[id] == lst_seq[id]
            ):
                id += 1
            common_token += id
            lst_seq = to_be_add.copy()
            if percent < 0.6:
                if lens > 2:
                    print(f"serie prompts lasts for {lens}, in following order:")
                    for id in range(start, start + lens):
                        print(f"No.{id-start+1}:", token_list[id])
                start = start + lens
                lens = 1
            else:
                lens += 1
    print(
        f"Total words count:{sum}, average words count:{sum/prompt_count}, averge common words count:{common_token/prompt_count}"
    )
