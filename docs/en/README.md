<img src="../logo.png" width="20%">

# Chitu「赤兔」

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/thu-pacman/chitu)

[中文](/README.md) | English

**Some documents in this repo are originally written in Chinese and then translated into English by LLM.**

Chitu is a high-performance large model inference framework focused on efficiency, flexibility, and usability.

> Chitu refers to a famous fast horse in Chinese history.

## Milestones
* [2025/08/01] Released v0.4.0, significantly improving performance and stability for all-in-one inference deployment scenarios, supporting Ascend, NVIDIA, Muxi, Hygon, and compatible with DeepSeek, Qwen, GLM, Kimi models.
* [2025/07/28] Released v0.3.9, first to support Huawei Ascend 910B inference deployment for GLM-4.5 MoE model.
* [2025/06/12] Released v0.3.5, providing complete native support for Ascend 910B and high-performance inference solutions for Qwen3 series models.
* [2025/04/29] Released v0.3.0, added efficient operator implementations for FP4→FP8/BF16 online conversion, supporting [FP4 quantized version](https://huggingface.co/nvidia/DeepSeek-R1-FP4) of DeepSeek-R1 671B.
* [2025/04/18] Released v0.2.2, added CPU+GPU heterogeneous hybrid inference support, enabling single-card inference for DeepSeek-R1 671B.
* [2025/03/14] Released v0.1.0, supporting DeepSeek-R1 671B with efficient operator implementations for FP8→BF16 online conversion.

## Introduction
Positioned as an "enterprise-grade large model inference engine", Chitu thoroughly considers the progressive needs from small-scale trials to large-scale deployments in enterprise AI implementation, focusing on delivering these key features:
- **Multi-hardware compatibility**: Supports not only nvidia's latest flagship to legacy product lines but also provides optimized support for non-nvidia chips.
- **Full-scenario scalability**: From pure CPU deployment to single GPU deployment and large-scale cluster deployment, Chitu offers extensible solutions.
- **Long-term stable operation**: Suitable for production environments with stability capable of handling concurrent business traffic.

The project team appreciates valuable feedback from users and the open-source community and will continue improving the Chitu inference engine. However, limited by team capacity, we cannot guarantee timely resolution of all issues encountered by users. For professional technical services, please email solution@chitu.ai.

## Benchmark Data
Please refer to our self-tested [performance data](/docs/en/PERFORMANCE.md). Results may vary based on your hardware configuration, software versions, and test workloads, with possible fluctuations across multiple tests. Welcome to [share your test results](https://github.com/thu-pacman/chitu/discussions/104).

## Installation & Usage
Refer to the [Developer Manual](DEVELOPMENT.md) for complete installation instructions. For quick validation in standalone environments, we recommend using official images currently available for:
* Ascend: qingcheng-ai-cn-beijing.cr.volces.com/public/chitu-ascend:latest
* NVIDIA: qingcheng-ai-cn-beijing.cr.volces.com/public/chitu-nvidia:latest
* Muxi: qingcheng-ai-cn-beijing.cr.volces.com/public/chitu-muxi:latest

### Supported Models

Please refer to [Supported Models](/docs/en/SUPPORTED_MODELS.md).

## Contribution Guidelines
Chitu welcomes all forms of contributions! See [CONTRIBUTING](/docs/en/CONTRIBUTING.md).

## Discussion
For questions or concerns, please submit issues.

## License
Apache License v2.0 - see [LICENSE](/LICENSE).

This repository contains code snippets from other open-source projects, and their license information is annotated in the code with the SPDX format. The associated license information can be found in the `LICENSES/` directory.

This repository contains third-party submodules under other open-source licenses found in `third_party/` with their respective license files.

## FAQ
[中文](/docs/zh/FAQ.md) | [English](/docs/en/FAQ.md)

## Acknowledgments
Special thanks to Huawei, Muxi, Hygon, Enflame, Zhipu AI, China Telecom, and Paratera for their support.

During Chitu's development, we've learned from these projects (alphabetical order) and reused some functions:
- [DeepSeek](https://github.com/deepseek-ai)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [KTransformers](https://github.com/kvcache-ai/ktransformers)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [SGLang](https://github.com/sgl-project/sglang)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [vLLM](https://github.com/vllm-project/vllm)

We'll continue contributing more efficient, flexible, compatible, and stable large model inference deployment solutions to the open-source community.
