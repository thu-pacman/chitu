#!/bin/bash

./script/srun_multi_node.sh 1 8 test/single_req_test.py \
    models=deepseek-r1 \
    infer.tp_size=8 \
    infer.pp_size=1 \
    infer.cache_type=paged \
    infer.attn_type=flash_mla \
    infer.mla_absorb=absorb-without-precomp \
    infer.max_reqs=1 \
    infer.max_seq_len=512 \
    request.max_new_tokens=100
