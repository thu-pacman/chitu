# FAQ (Frequently Asked Questions)

### Q1: If a chip itself does not have an FP8 unit, how can I run an FP8 model directly?
We can use the FP8 format to store weights and the BF16 format to perform operations, which is equivalent to somekind of w8a16, but the 8 here is float8.
However, since the floating point conversion operation is more complicated than the integer conversion, there are some technical challenges.
[This post](https://www.zhihu.com/question/14928372981/answer/124606559367?utm_psn=1884175276604384926) on Zhihu explains key optimization points in format conversion.

### Q2: It is easy to understand that FP8 saves more computing power than BF16, but why is there still speedup?
In short, saving half of the computing power while achieving several times speedup is a **relatively special case**. More often, Chitu brings cost-effectiveness rather than absolute performance improvement.
Regarding when this **relatively special case** will occur, [this post](https://www.zhihu.com/question/14928372981/answer/124606559367?utm_psn=1884175276604384926) on Zhihu makes explanations.

### Q3: When will chitu support non-nvidia GPUs?
The released version v0.1.0 can run on certain non-nvidia GPUs, but the high-performance operator implementation is not yet included. 

### Q4: Why do we need Chitu? There are already open source projects such as vllm, sglang, and llama.cpp.
Although there are many excellent projects, building Chitu is not reinventing the wheel.
Chitu is more focused on aspects that existing open source projects do not take good care of, such as diversed non-nvidia GPU support.
We think it is a useful supplement to the open source ecosystem of large models.

### Q5: Which scenarios are suitable for Chitu and which are not
As for the version v0.1.0, it is for users who have non-Hopper GPUs and want to run FP8 directly, such as users with A800 clusters.
On the H20 platform, since the hardware already supports FP8 very well and there are already many excellent open source implementations, using Chitu will not immediately bring significant improvements.
Of course, Chitu will continue to optimize its performance on platforms such as H20.

### Q6: Will Chitu support CPU serving or CPU+GPU serving?
From the perspective of supporting the smooth expansion of serving system from small to large, YES. Please stay tuned.
