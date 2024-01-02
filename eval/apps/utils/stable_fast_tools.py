from sfast.compilers.diffusion_pipeline_compiler import (
    compile,
    compile_unet,
    CompilationConfig,
)


def get_default_sfast_config() -> CompilationConfig:
    config = CompilationConfig.Default()
    config.enable_xformers = True
    config.enable_triton = True
    config.enable_cuda_graph = True
    return config
