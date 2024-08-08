# Quantization
## Qwen2
|量化方法|数据类型|使用方法|占用显存|单任务时间(bs=6)|
|----|----|----|----|-----------|
| bfloat16|         | use example/ser_config.yaml         |17194313728| 5499.25  |
| llmint8 |  w8a8   | use example/ser_config_llmint8.yaml |10376866304| 15778.34 |
| awq     |  w4a16  | use example/ser_config_awq.yaml     |7640655872 | 3148.76  |
| gptq    |  w8a16  | use example/ser_config_gptq.yaml    |10803440640| 11179.28 |
| w816    |  w8a16  | use example/ser_config_w8a16.yaml   |10627766272| 3952.82  |

