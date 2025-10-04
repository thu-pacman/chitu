<img src="docs/logo.png" width="20%">

# Chitu「赤兔」

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/thu-pacman/chitu)

中文 | [English](/docs/en/README.md)

Chitu「赤兔」是一个专注于效率、灵活性和可用性的高性能大模型推理框架。

## 里程碑

* [2025/08/01] 发布 v0.4.0，大幅提升了一体机推理部署场景的性能和稳定性，适配昇腾、英伟达、沐曦、海光，支持 DeepSeek、Qwen、GLM、Kimi 等模型。
* [2025/07/28] 发布 v0.3.9，首发支持华为昇腾 910B 推理部署智谱 GLM-4.5 MoE 模型。
* [2025/06/12] 发布 v0.3.5，提供昇腾 910B 完整原生支持，提供 Qwen3 系列模型高性能推理方案。
* [2025/04/29] 发布 v0.3.0，新增 FP4 在线转 FP8、BF16 的高效算子实现，支持 DeepSeek-R1 671B 的 [FP4 量化版](https://huggingface.co/nvidia/DeepSeek-R1-FP4)。
* [2025/04/18] 发布 v0.2.2，新增 CPU+GPU 异构混合推理支持，实现单卡推理 DeepSeek-R1 671B。
* [2025/03/14] 发布 v0.1.0，支持 DeepSeek-R1 671B，提供 FP8 在线转 BF16 的高效算子实现。

## 简介

赤兔定位于「生产级大模型推理引擎」，充分考虑企业 AI 落地从小规模试验到大规模部署的渐进式需求，专注于提供以下重要特性：

- **多元算力适配**：不仅支持 NVIDIA 最新旗舰到旧款的多系列产品，也为国产芯片提供优化支持。
- **全场景可伸缩**：从纯 CPU 部署、单 GPU 部署到大规模集群部署，赤兔引擎提供可扩展的解决方案。
- **长期稳定运行**：可应用于实际生产环境，稳定性足以承载并发业务流量。

项目团队感谢广大用户及开源社区提出的宝贵意见和建议，并将持续改进赤兔推理引擎。
然而，受制于团队成员的精力，无法保证及时解决所有用户在使用中遇到问题。
如需专业技术服务，欢迎致信 solution@chitu.ai

## 测试数据

请参阅赤兔开发团队测试的[性能数据](docs/zh/PERFORMANCE.md)，也欢迎分享您的[自测数据](https://github.com/thu-pacman/chitu/discussions/104)。

性能数据与您的硬件配置、软件版本、测试负载相关，多次测试结果可能存在波动。

## 安装使用

请参阅[开发手册](/docs/zh/DEVELOPMENT.md)获取完整的安装使用说明。

对于在单机环境上快速验证的场景，建议使用官方镜像进行部署。目前提供适用于以下平台的镜像：
* 昇腾：qingcheng-ai-cn-beijing.cr.volces.com/public/chitu-ascend:latest
* 英伟达：qingcheng-ai-cn-beijing.cr.volces.com/public/chitu-nvidia:latest
* 沐曦：qingcheng-ai-cn-beijing.cr.volces.com/public/chitu-muxi:latest

### 查看支持的模型

更多模型请参见 [支持的模型](/docs/zh/SUPPORTED_MODELS.md)。

## 参与开发

赤兔项目欢迎开源社区的朋友们参与项目共建，请参阅[贡献指南](/docs/zh/CONTRIBUTING.md)。

## 交流讨论

如果您有任何问题或疑虑，欢迎提交issue。

您也可以扫码加入赤兔交流微信群：

<img src="docs/WeChatGroup.png" width="20%">

## 许可证

本项目采用 Apache License v2.0 许可证 - 详见 [LICENSE](/LICENSE) 文件。

本代码仓库还引用了一些来自其他开源项目的代码片段，相关版权信息已在代码中以 SPDX 格式标注。这些代码片段的许可证信息可以在 `LICENSES/` 目录下找到。

本代码仓库还包含遵循其他开源许可证的第三方子模块。您可以在 `third_party/` 目录下找到这些子模块，该目录中包含了它们各自的许可证文件。

## 常见问题

[中文](/docs/zh/FAQ.md) | [English](/docs/en/FAQ.md)

## 致谢

非常感谢来自华为、沐曦、海光、燧原、智谱、中国电信、并行科技等各方的帮助。

在构建 Chitu 的过程中，我们从以下项目（按字母排序）中学到了很多，并复用了一些函数：

- [DeepSeek](https://github.com/deepseek-ai)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [KTransformers](https://github.com/kvcache-ai/ktransformers)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [SGLang](https://github.com/sgl-project/sglang)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [vLLM](https://github.com/vllm-project/vllm)

我们将持续为开源社区贡献更高效、更灵活、更兼容、更稳定的大模型推理部署解决方案。
