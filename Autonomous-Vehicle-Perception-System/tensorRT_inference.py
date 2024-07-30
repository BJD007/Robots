import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

def load_tensorrt_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def infer_tensorrt(engine, input_data):
    h_input = cuda.pagelocked_empty(input_data.shape, dtype=np.float32)
    h_output = cuda.pagelocked_empty(output_shape, dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    stream = cuda.Stream()
    
    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, h_input, stream)
    
    # Run inference
    context = engine.create_execution_context()
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    
    # Transfer predictions back
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    
    # Synchronize the stream
    stream.synchronize()
    
    return h_output

# Usage
engine = load_tensorrt_engine("model_tensorrt.engine")
input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)  # Example input
output = infer_tensorrt(engine, input_data)
