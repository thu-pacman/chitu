import tensorrt as trt
import numpy as np
import os
import time

import pycuda.driver as cuda
import pycuda.autoinit

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("batch_size", type=int)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TrtModel:
    def __init__(self, engine_path, max_batch_size=1, dtype=np.float32):
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        (
            self.inputs,
            self.outputs,
            self.bindings,
            self.stream,
            self.input_shapes,
        ) = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        input_shapes = []

        print(f"{self.engine=} {self.engine.__class__.__name__=}")
        for binding in self.engine:
            print(f"{binding=} shape {self.engine.get_tensor_shape(binding)}")
            size = (
                trt.volume(self.engine.get_tensor_shape(binding)) * self.max_batch_size
            )
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
                input_shapes.append(self.engine.get_tensor_shape(binding))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        print(f"{inputs=}, {outputs=}, {bindings=}, {stream=}")
        return inputs, outputs, bindings, stream, input_shapes

    def __call__(self):
        # self.context.execute_async(
        #     batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle
        )
        # return [out.host.reshape(batch_size, -1) for out in self.outputs]

    def copy_input(self, xs):
        for i, x in enumerate(xs):
            x = x.astype(self.dtype)
            np.copyto(self.inputs[i].host, x.ravel())
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

    def copy_output(self):
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

    def sync(self):
        self.stream.synchronize()


if __name__ == "__main__":
    args = parser.parse_args()
    warmup = 1
    iter = 1

    batch_size = args.batch_size * 2
    # trt_engine_path = os.path.join("..", "models", "main.trt")
    trt_engine_path = f"/home/zly/Works/ModelZoo/onnx/diffusion/models/unet/1/log/unet.{batch_size}.fp16.engine"
    model = TrtModel(trt_engine_path)
    shapes = model.input_shapes
    print(shapes)

    data = []
    for shape in shapes:
        data.append(np.random.randint(0, 255, shape) / 255)
    model.copy_input(data)

    # Warmup
    for i in range(warmup):
        model()

    model.sync()
    start = time.time()
    cuda.start_profiler()
    for i in range(iter):
        model()
    model.sync()
    end = time.time()
    model.copy_output()
    model.sync()
    print(
        f"Time {end-start:.2f} s for {iter} iterations. {(end-start)/iter:.2f} s/iter"
    )
    result = model.outputs[0].host
    print(result)
