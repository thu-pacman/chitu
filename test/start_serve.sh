export WORLD_SIZE=8

srun -p debug --job-name qly-serve --nodes 1 --ntasks-per-node 1 --cpus-per-task 32 --gpus-per-task 8 timeout 60m \
    torchrun --nnodes 1 \
    --nproc_per_node 8 \
    --master_port=22525 \
    -m chitu \
    serve.port=21002 \
    infer.cache_type=paged \
    infer.pp_size=1 \
    infer.tp_size=8 \
    models=DeepSeek-R1 \
    models.ckpt_dir=/data/DeepSeek-R1 \
    infer.attn_type=flash_infer \
    keep_dtype_in_checkpoint=True \
    infer.mla_absorb=absorb-without-precomp \
    infer.soft_fp8=True \
    infer.do_load=True \
    infer.max_reqs=8 \
    scheduler.prefill_first.num_tasks=100 \
    infer.max_seq_len=4096 \
    request.max_new_tokens=100 \
    infer.use_cuda_graph=True