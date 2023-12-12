#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/im2col_shape_check.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>
#include <vector>

#include <c10/macros/Macros.h>

using namespace at;
using namespace at::native;
using namespace at::cuda::detail;
using itype = int64_t;

// borrowed from PyTorch:
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/im2col.cuh
// CUDA_NUM_THREADS = 1024

template <typename dt>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void ragged_nhwc_im2col_kernel(
    const int64_t n, const dt *x_ptr, const int64_t batch_size,
    const int64_t channels, const int64_t *heights, const int64_t *widths,
    const int64_t kernel_height, const int64_t kernel_width,
    const int64_t pad_height, const int64_t pad_width,
    const int64_t stride_height, const int64_t stride_width,
    const int64_t dilation_height, const int64_t dilation_width,
    const int64_t *heights_col, const int64_t *widths_col, dt *y_ptr) {
    CUDA_KERNEL_LOOP(global_index, n) {
        int64_t index = global_index;
        int64_t batch = 0, in_offset = 0, out_offset = 0;
        for (; batch < batch_size; ++batch) {
            auto length = channels * heights_col[batch] * widths_col[batch];
            if (index < length) {
                break;
            } else {
                index -= length;
            }
            in_offset += channels * heights[batch] * widths[batch];
            out_offset += length;
        }
        const int64_t width_col = widths_col[batch];
        const int64_t height = heights[batch];
        const int64_t width = widths[batch];

        // TODO: check how is idx decieded and what does each thread block do?
        int64_t channel_in = index % channels;
        int64_t idx = index / channels;
        int64_t w_out = idx % width_col;
        int64_t h_out = idx / width_col;

        int64_t channel_out = channel_in;
        int64_t h_in = h_out * stride_height - pad_height;
        int64_t w_in = w_out * stride_width - pad_width;

        dt *col = y_ptr // [nhwrsc]
                        // previous batch offset
                  + out_offset * kernel_height * kernel_width
                  // h,w offset in the current batch
                  + (h_out * width_col + w_out) * kernel_height * kernel_width *
                        channels
                  // channel offset
                  + channel_out;
        const dt *im = x_ptr // [nhwc]
                       + in_offset
                       // h,w offset in the current batch
                       + (h_in * width + w_in) * channels + channel_in;

        // printf("%lld %lld %lld %lld %lld %lld %lld %lld %lld %lld\n", index,
        //        in_offset, out_offset, channel_in, h_out, w_out, h_in, w_in,
        //        col - y_ptr, x_ptr - im);
        for (int64_t i = 0; i < kernel_height; ++i) {
            for (int64_t j = 0; j < kernel_width; ++j) {
                int64_t h = h_in + i * dilation_height;
                int64_t w = w_in + j * dilation_width;
                *col = (h >= 0 && w >= 0 && h < height && w < width)
                           ? im[(i * dilation_height * width +
                                 j * dilation_width) *
                                channels]
                           : static_cast<dt>(0);
                // printf("== %lld %lld %lld\n", index, col - y_ptr,
                //        x_ptr - im);
                col += channels; // Step in for next (h,w)
            }
        }
    }
}

template <typename dt>
void ragged_nhwc_im2col_wrapper(
    cudaStream_t stream, const dt *data_im, const int64_t batch_size,
    const int64_t channels, const int64_t *heights, const int64_t *widths,
    const int64_t *heights_col, const int64_t *widths_col,
    const int64_t *heights_col_cpu, const int64_t *widths_col_cpu,
    const int64_t kernel_height, const int64_t kernel_width,
    const int64_t pad_height, const int64_t pad_width,
    const int64_t stride_height, const int64_t stride_width,
    const int64_t dilation_height, const int64_t dilation_width, dt *data_col) {
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int64_t num_kernels = 0;
    for (int i = 0; i < batch_size; ++i)
        num_kernels += channels * heights_col_cpu[i] * widths_col_cpu[i];
    // Launch CUDA_NUM_THREADS = 1024
    ragged_nhwc_im2col_kernel<<<GET_BLOCKS(num_kernels), 1024, 0, stream>>>(
        num_kernels, data_im, batch_size, channels, heights, widths,
        kernel_height, kernel_width, pad_height, pad_width, stride_height,
        stride_width, dilation_height, dilation_width, heights_col, widths_col,
        data_col);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

Tensor ragged_nhwc_im2col(const Tensor &input, const Tensor &idx_cuda,
                          const Tensor &idx_cpu, const Tensor &idx_out_cuda,
                          const Tensor &idx_out_cpu, IntArrayRef kernel_size,
                          IntArrayRef dilation, IntArrayRef padding,
                          IntArrayRef stride) {
    TORCH_CHECK(kernel_size.size() == 2,
                "It is expected kernel_size equals to 2, but got size ",
                kernel_size.size());

    TORCH_CHECK(dilation.size() == 2,
                "It is expected dilation equals to 2, but got size ",
                dilation.size());

    TORCH_CHECK(padding.size() == 2,
                "It is expected padding equals to 2, but got size ",
                padding.size());

    TORCH_CHECK(stride.size() == 2,
                "It is expected stride equals to 2, but got size ",
                stride.size());

    int64_t kernel_height = kernel_size[0];
    int64_t kernel_width = kernel_size[1];
    int64_t dilation_height = dilation[0];
    int64_t dilation_width = dilation[1];
    int64_t pad_height = padding[0];
    int64_t pad_width = padding[1];
    int64_t stride_height = stride[0];
    int64_t stride_width = stride[1];

    // TensorArg input_arg{input_, "input", 1};
    // TensorArg output_arg{output, "output", 2};
    // checkAllSameGPU(__func__, {input_arg, output_arg});

    // im2col_shape_check(input_, Tensor(), kernel_height, kernel_width,
    //                    dilation_height, dilation_width, pad_height,
    //                    pad_width, stride_height, stride_width);
    TORCH_CHECK(input.dim() == 2);

    // Tensor input = input.contiguous();

    bool batched_input = true;

    // if (input.dim() == 3) {
    //     batched_input = false;
    //     input = input.view({1, input.size(0), input.size(1), input.size(2)});
    // }

    int64_t batch_size = idx_cuda.size(1);
    int64_t n_input_plane = input.size(1); // in_channels
    int64_t out_nhw_length = 0;
    auto hs_tensor = idx_cpu.index({0});
    auto hs = hs_tensor.accessor<itype, 1>();
    auto ws_tensor = idx_cpu.index({1});
    auto ws = ws_tensor.accessor<itype, 1>();
    for (int i = 0; i < batch_size; ++i) {
        int64_t input_height = hs[i];
        int64_t input_width = ws[i];

        int64_t output_height = (input_height + 2 * pad_height -
                                 (dilation_height * (kernel_height - 1) + 1)) /
                                    stride_height +
                                1;
        int64_t output_width = (input_width + 2 * pad_width -
                                (dilation_width * (kernel_width - 1) + 1)) /
                                   stride_width +
                               1;
        TORCH_CHECK(output_height == idx_out_cpu[0][i].item<itype>());
        TORCH_CHECK(output_width == idx_out_cpu[1][i].item<itype>());
        //  int64_t output_length = output_height * output_width;
        out_nhw_length += output_height * output_width;
    }
    int64_t n_output_plane = n_input_plane * kernel_width * kernel_height;

    // output.resize_({batch_size, n_output_plane, output_length});
    auto output =
        torch::empty({out_nhw_length, n_output_plane}, input.options());

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "ragged_nhwc_im2col_cuda", [&] {
            for (int64_t elt = 0; elt < batch_size; elt++) {
            }
            ragged_nhwc_im2col_wrapper<scalar_t>(
                at::cuda::getCurrentCUDAStream(),
                input.const_data_ptr<scalar_t>(), batch_size, n_input_plane,
                // input_heights, input_widths, output_height, output_width,
                idx_cuda.index({0}).const_data_ptr<itype>(),
                idx_cuda.index({1}).const_data_ptr<itype>(),
                idx_out_cuda.index({0}).const_data_ptr<itype>(),
                idx_out_cuda.index({1}).const_data_ptr<itype>(),
                idx_out_cpu.index({0}).const_data_ptr<itype>(),
                idx_out_cpu.index({1}).const_data_ptr<itype>(), kernel_height,
                kernel_width, pad_height, pad_width, stride_height,
                stride_width, dilation_height, dilation_width,
                output.mutable_data_ptr<scalar_t>());
        });
    return output;
}
