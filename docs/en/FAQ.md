# Frequently Asked Questions (FAQ)

### Q1: How to make Chitu support our model?
If you're a large model developer seeking Chitu compatibility for your model:
1) Submit a Pull Request - our team will review and merge after confirmation (see [CONTRIBUTING](/docs/en/CONTRIBUTING.md))
2) For technical difficulties, contact our support team at solution@chitu.ai

### Q2: How to make Chitu support our chip?
If you're developing or using an unsupported chip architecture:
1) Submit a Pull Request for review (see [CONTRIBUTING](/docs/en/CONTRIBUTING.md))
2) For adaptation challenges, email solution@chitu.ai

### Q3: How to run FP4/FP8 models without native FP4/FP8 compute units?
Solution: Store weights in FP8 format but execute computations in BF16 (similar to w8a16 quantization where "8" refers to float8).  
Note: Floating-point conversion involves greater technical complexity than integer conversion. Technical details are explained in this [Zhihu article](https://www.zhihu.com/question/14928372981/answer/124606559367?utm_psn=1884175276604384926).

### Q4: Why does FP4/FP8 sometimes accelerate performance beyond just compute savings?  
While typically improving cost-performance ratios rather than raw performance, **exceptional cases** may show both compute savings and speedups. This [Zhihu analysis](https://www.zhihu.com/question/14928372981/answer/124606559367?utm_psn=1884175276604384926) explains when such **exceptional cases** occur.

### Q5: How does Chitu differ from vLLM/SGLang/llama.cpp?  
Chitu complements rather than replicates existing solutions by focusing on:
1) Native support for non-nvidia chips (e.g., Ascend/Muxi/Hygon)  
2) Seamless scalability from minimal to large-scale deployments  

### Q6: Ideal use cases for Chitu  
Consider Chitu if you:
1) Use non-nvidia chips (Ascend/Muxi/Hygon/etc.)  
2) Employ heterogeneous computing (mixed chips)  
3) Require high-performance inference  
4) Seek cost-efficient deployment  
5) Engaged in research on inference framework

### Q7: Does Chitu support CPU-only or CPU+GPU inference?  
Since v0.2.2: Supports CPU+GPU heterogeneous inference  
CPU-only support: Planned feature  