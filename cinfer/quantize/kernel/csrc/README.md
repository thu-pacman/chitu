Extracted fp16 A and int8/4 B CUTLASS GEMM kernels from FasterTransformer for easier integration in third-party projects. See the original code below.
* https://github.com/NVIDIA/FasterTransformer/tree/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions
* https://github.com/NVIDIA/FasterTransformer/tree/main/src/fastertransformer/kernels/cutlass_kernels/fpA_intB_gemm

Build with
```
mkdir build && cd build
cmake ..
make
```
