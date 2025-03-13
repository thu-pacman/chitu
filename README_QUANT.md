# Quantization
## Qwen2
|量化方法|数据类型|使用方法|占用显存|单任务时间(bs=6)|ppl|
|----|----|----|----|-----------|---|
| bfloat16|         | use example/ser_config.yaml         |17194313728| 5499.25  | |
| llmint8 |  w8a8   | use example/ser_config_llmint8.yaml |10376866304| 15778.34 | |
| awq     |  w4a16  | use example/ser_config_awq.yaml     |7640655872 | 3148.76  | 8.539262771606445|
| gptq    |  w8a16  | use example/ser_config_gptq.yaml    |10803440640| 11179.28 | 8.23493003845214|
| w816    |  w8a16  | use example/ser_config_w8a16.yaml   |10627766272| 3952.82  | |

time (ms) | overall: 2397.86 1 2397.86 | prefill: 57.83 1 57.83 | cache_finalize_cache_all_prefill: 16.83 28 0.60 | get_free_block: 7.39 112 0.07 | decode: 2152.69 63 34.17 | decode-model: 2149.49 63 34.12 | prepare_block_table_for_decode: 16.32 1764 0.01 | get_gpu_block_table: 77.87 1764 0.04 | finalize_cache_all_decode: 0.18 4 0.04 | free_req_cache_blocks: 0.11 4 0.03

time (ms) | overall: 2730.26 1 2730.26 | prefill: 79.39 1 79.39 | cache_finalize_cache_all_prefill: 31.02 28 1.11 | get_free_block: 14.07 224 0.06 | decode: 2405.81 63 38.19 | decode-model: 2402.77 63 38.14 | prepare_block_table_for_decode: 15.74 1764 0.01 | get_gpu_block_table: 84.79 1764 0.05 | finalize_cache_all_decode: 0.41 8 0.05 | free_req_cache_blocks: 0.29 8 0.04

time (ms) | overall: 3165.55 1 3165.55 | prefill: 212.42 1 212.42 | cache_finalize_cache_all_prefill: 115.12 28 4.11 | get_free_block: 54.18 896 0.06 | decode: 2453.07 63 38.94 | decode-model: 2449.77 63 38.89 | prepare_block_table_for_decode: 19.21 1764 0.01 | get_gpu_block_table: 102.49 1764 0.06 | finalize_cache_all_decode: 1.06 32 0.03 | free_req_cache_blocks: 0.61 32 0.02

time (ms) | overall: 2118.94 1 2118.94 | prefill: 49.76 1 49.76 | cache_finalize_cache_all_prefill: 4.95 28 0.18 | cache_prepare: 3.39 63 0.05 | decode: 1891.84 63 30.03 | decode-model: 1888.60 63 29.98

time (ms) | overall: 2239.17 1 2239.17 | prefill: 63.31 1 63.31 | cache_finalize_cache_all_prefill: 9.03 28 0.32 | cache_prepare: 3.40 63 0.05 | decode: 1917.45 63 30.44 | decode-model: 1914.25 63 30.38