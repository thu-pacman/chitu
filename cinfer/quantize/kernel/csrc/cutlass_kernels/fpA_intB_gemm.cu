#include "fpA_intB_gemm.h"
#include "fpA_intB_gemm/fpA_intB_gemm_template.h"

namespace fastertransformer
{

  ActivationType get_activation(const std::string &activation_name)
  {
    if (activation_name == "identity")
      return ActivationType::Identity;
    if (activation_name == "relu")
      return ActivationType::Relu;
    if (activation_name == "silu")
      return ActivationType::Silu;
    if (activation_name == "gelu")
      return ActivationType::Gelu;
    // todo: more
    return ActivationType::InvalidType;
  }

  void gemm_fp16_int(const half *A,
                     const uint8_t *B,
                     const half *weight_scales,
                     half *C,
                     int m, int n, int k,
                     char *workspace_ptr,
                     size_t workspace_bytes,
                     cudaStream_t stream)
  {
    CutlassFpAIntBGemmRunner<half, uint8_t> runner;
    runner.gemm(A, B, weight_scales,
                C, m, n, k, workspace_ptr, workspace_bytes, stream);
  }

  template <typename WeightType>
  void gemm_fp16_int_bias_act(const half *A,
                              const WeightType *B,
                              const half *weight_scales,
                              const half *bias,
                              half *C,
                              std::optional<std::string> activation,
                              int m, int n, int k, int bias_stride, char *workspace_ptr,
                              size_t workspace_bytes, cudaStream_t stream)
  {
    CutlassFpAIntBGemmRunner<half, WeightType> runner;

    if (!activation && bias == nullptr)
    {
      runner.gemm(A, B, weight_scales,
                  C, m, n, k, workspace_ptr, workspace_bytes, stream);
    }
    else if (!activation)
    {
      runner.gemm_bias_act(A, B, weight_scales, bias,
                           C, m, n, k, bias_stride, ActivationType::Identity, workspace_ptr, workspace_bytes, stream);
    }
    else
    {
      runner.gemm_bias_act(A, B, weight_scales, bias,
                           C, m, n, k, bias_stride, get_activation(*activation), workspace_ptr, workspace_bytes, stream);
    }
  }

  template <typename WeightType>
  void gemm_fp16_int_bias_act_residual(
      const half *A, const WeightType *B, const half *weight_scales,
      const half *bias, const half *residual, half *C, const std::string &activation, const std::string &binary_op,
      const std::string &unary_op, int m, int n,
      int k, char *workspace_ptr, size_t workspace_bytes, cudaStream_t stream)
  {
    CutlassFpAIntBGemmRunner<half, WeightType> runner;

    runner.gemm_bias_act_residual(A, B, weight_scales, bias, residual,
                                  C, m, n, k, activation, binary_op, unary_op, workspace_ptr, workspace_bytes, stream);
  }

  template void gemm_fp16_int_bias_act<uint4b_t>(const half *A, const uint4b_t *B,
                                                 const half *weight_scales, const half *bias,
                                                 half *C, std::optional<std::string> activation, int m,
                                                 int n, int k, int bias_stride, char *workspace_ptr,
                                                 size_t workspace_bytes, cudaStream_t stream);

  template void gemm_fp16_int_bias_act_residual<uint4b_t>(
      const half *A, const uint4b_t *B, const half *weight_scales,
      const half *bias, const half *residual, half *C, const std::string &activation, const std::string &binary_op,
      const std::string &unary_op, int m, int n, int k, char *workspace_ptr, size_t workspace_bytes, cudaStream_t stream);

  template void gemm_fp16_int_bias_act<uint8_t>(const half *A, const uint8_t *B,
                                                const half *weight_scales, const half *bias,
                                                half *C, std::optional<std::string> activation, int m,
                                                int n, int k, int bias_stride, char *workspace_ptr,
                                                size_t workspace_bytes, cudaStream_t stream);

  template void gemm_fp16_int_bias_act_residual<uint8_t>(
      const half *A, const uint8_t *B, const half *weight_scales,
      const half *bias, const half *residual, half *C, const std::string &activation, const std::string &binary_op,
      const std::string &unary_op, int m, int n, int k, char *workspace_ptr, size_t workspace_bytes, cudaStream_t stream);

} // namespace fastertransformer
