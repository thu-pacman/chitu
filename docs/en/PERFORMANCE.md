# Performance Evaluation

>  Performance depends on hardware platform, dependency versions, and task configuration. Performance may vary across multiple runs.

Unless explicitly noted, the following configurations are used for evaluation:

* Sequence lengths: input_len = 128 tokens, output_len = 1024 tokens.
* The data types of the original models are used.
* TPS numbers are output TPS numbers. Output TPS = output_tokens / (prefill_sec + decode_sec)。
* N/A stands for missing test results, but it does not imply Chitu cannot run this case.

## Dense Models
### Qwen3-32B

| BS & TPS | 1xH20(96GB) | 2x910B2(64GB) | 4xDCU(64GB) |
| ---------- | ---------------- | ----- | ----- |
| 1          | 44.79 | 24.39 | 25.04 |
| 2          | 85.69 | 45.57 | 47.73 |
| 4          | 167.76 | 84.54 | 94.50 |
| 8          | 319.07 | 146.98 | 181.96 |
| 16         | 585.13 | 265.45 | 346.90 |
| 32         | 977.31 | 470.50 | 592.70 |
| 64         | 1333.49 | 805.26 | 962.24 |
| 128        | N/A | 1223.77 | N/A |

## 混合专家模型
### DeepSeek-R1-671B
| BS & TPS | 16xH20(96GB) TP8PP2 | 32xH20(96GB) DP32EP32 |
| ---------- | ---------------- | ---- |
| 1          | 57.74 | |
| 2          | 107.76 | |
| 4          | 181.68 | |
| 8          | 264.74 | |
| 16         | 420.13 | |
| 32         | 636.03 | 660.34 |
| 64         | 978.04 |  |
| 128        | 1862.98 | 2374.99 |
| 256        | | |
| 512        | | 7458.43 |
| 1024       | | 11696.37 |
| 2048       | | 16022.60 |

### Kimi-K2-1T
| BS & TPS | 16xH20(96GB) |
| ---------- | ---------------- |
| 1          | 47.51 |
| 2          | 92.02 |
| 4          | 163.94 |
| 8          | 241.52 |
| 16         | 403.08 |
| 32         | 634.12 |
| 64         | 943.05 |
| 128        | 1383.46 |
| 256        | 2571.74 |

### GLM-4.5-Air-106B-A12B
| BS & TPS | 8xH20(96GB) | 8x910B2(64GB) |
| ---------- | ---------------- | --- |
| 1          | 113.28           | 30.81 |
| 2          | 193.84           | 54.51 |
| 4          | 352.84           | 98.48 |
| 8          | 621.75           | 168.50 |
| 16         | 1058.10          | 286.75 |
| 32         | 1774.08          | 477.89 |
| 64         | 2986.52          | 796.72 |
| 128        | 4757.48          | 1317.03 |

## Quantized Models
### Deploy DeepSeek-R1-671B on a single eight-card H20 (96G) server

| Output TPS | chitu 0.3.0, 原版 FP8| chitu 0.3.0, FP4->FP8 | chitu 0.3.0, FP4->BF16 |
|:---|:---|:---|:---|
|bs=1| 24.30 | 20.70 | 19.78 |
|bs=16| 203.71 | 89.56 | 110.68 |
|bs=64| OOM | 237.20 | 232.14 |
|bs=128| OOM | 360.80 | 351.73 |
| **MMLU score** | 89.8 | 88.0 | 88.0 |

- The total memory capacity of the eight-card machine is 768GB, while the weight of the original model needs to be close to 700GB, so the number of concurrent operations that can be supported is not large.
- The weight of the FP4 quantized model only requires less than 400GB of video memory space, so it can support a larger number of concurrent operations; it also makes it easy to deploy the 671B model on a server with a GPU configuration of 8 * 64GB.
- The input and output lengths used in the performance test in the above table are both 512 tokens.
- In the MMLU precision test, the FP4 quantized version scores (88.0) better than the INT8 quantized version (87.2) and the INT4 quantized version (82.1), which is about 2% lower than the original version.
- There is still room for performance improvement in the FP4->FP8/BF16 related operator implementation in the v0.3.0 version, which will be optimized in subsequent updates.

### Heterogeneous deployment of DeepSeek-R1-671B on Xeon 8480P + H20 (96G) servers

| Number of layers fully placed on GPU | Number of GPUs | output token/s (bs=1) | output token/s (bs=16) |
|:---------------|:------|:---------------|:----------------|
| 0    | 1    | 10.61          | 28.16            |
| 24    | 2    | 14.04           | 42.57        |

- The model used is the Q4 quantization version (INT4) of DeepSeek-R1-671B.
- With Chitu v0.2.2.
- The performance bottleneck is on the CPU side. The performance improvement is limited after increasing the number of GPUs. It is recommended to use a higher-end CPU and main memory.
- Suitable for scenarios where GPU video memory is limited and high concurrency support is not required.
- MMLU test score is about 83.

### Deploy DeepSeek-R1-671B on A800 (40GB) cluster

|BS & TPS |6 节点, BF16 |3 节点, FP8|
|:---|:---|:---|
|1| 29.8 | 22.7 |
|4| 78.8 | 70.1 |
|8| 129.8 | 108.9 |
|16| 181.4 | 159.0 |
|32| 244.1 | 214.5 |

- With Chitu v0.1.0, may be out-dated.
- From the test data of different batch sizes, based on the Chitu engine, the output speed of the FP8 model running on 3 nodes is about 75%\~90% of that of the BF16 model running on 6 nodes, that is, the output per unit computing power has been improved by 1.5x\~1.8x.

- This is because the decoding process mainly depends on the memory access bandwidth. Using half of the GPU to access half of the data (the weight size of FP8 is half of that of BF16) will not take longer, and the reduction of GPU computing power will only bring a small impact.
