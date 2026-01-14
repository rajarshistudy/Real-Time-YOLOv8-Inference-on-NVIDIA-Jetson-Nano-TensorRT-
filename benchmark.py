import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from yolo_postprocess import post_process  # initializes CUDA context


with open("coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
assert len(class_names) == 80

# Small percentile function to help calculate latencies
def p(x, q): return float(np.percentile(x, q))

# Engine details (from your trt_engine_info.py)
ENGINE_PATH = "yolov8n.engine"
INPUT_NAME = "images"
OUTPUT_NAME = "output0"
INPUT_SHAPE = (1, 3, 640, 640)      # NCHW
OUTPUT_SHAPE = (1, 84, 8400)        # YOLOv8 raw output
WARMUP = 50                         # Warmup frames for Benchmark tests
LOG_FRAMES = 300                    # Frame cutoff
DTYPE = np.float32

# Time events to calculate CPU timline and GPU timeline
start_evt= cuda.Event()
end_evt = cuda.Event()

# Arrays to log latencies
lat_e2e = []
lat_gpu = []
lat_pre = []
lat_post = []

# Load TensorRT engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()     # Makes the TensorRT engine available
stream = cuda.Stream()  # Enables asynchronous copies and execution

print("TensorRT version:", trt.__version__)
print("Engine loaded. Running webcam inference...")


# Allocate GPU buffers
# Size of input and output elements 
input_size = int(np.prod(INPUT_SHAPE))
output_size = int(np.prod(OUTPUT_SHAPE))

# Allocate memory within the GPU for the TensorRT engine
d_input = cuda.mem_alloc(input_size * DTYPE().nbytes)
d_output = cuda.mem_alloc(output_size * DTYPE().nbytes)

h_output = np.empty(output_size, dtype=DTYPE)  # empty variable to store Engine output

# Specify the input and output addresses for the TensorRT engine
# Makes sure the Engine knows where to read FROM and where to write TO
context.set_tensor_address(INPUT_NAME, int(d_input))
context.set_tensor_address(OUTPUT_NAME, int(d_output))


# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

prev_time = time.time()     # To calculate FPS
frame_count = 0
bench_start = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    t0 = time.time()

    # Preprocessing data
    # OpenCV frame is HWC BGR uint8 (0-255)
    # Engine wants NCHW RGB float32 normalized (0-1)
    resized = cv2.resize(frame, (640, 640))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img = rgb.astype(np.float32) / 255.0          # float32 normalize
    img = np.transpose(img, (2, 0, 1))            # HWC -> CHW
    input_tensor = np.expand_dims(img, axis=0)    # CHW -> NCHW

    # Verification
    assert input_tensor.shape == INPUT_SHAPE
    assert input_tensor.dtype == np.float32

    t1 = time.time()
    start_evt.record(stream)

    # Copy input to GPU using PyCUDA (host to device)
    # Note that this is asynchronous to allow computation overlap and better performance
    cuda.memcpy_htod_async(d_input, input_tensor.ravel(), stream)
    # Execute inference
    ok = context.execute_async_v3(stream_handle=stream.handle)
    if not ok:
        raise RuntimeError("TensorRT execute_async_v3 failed")
    # Copy output back to CPU again using PyCUDA this time using devide to host
    cuda.memcpy_dtoh_async(h_output, d_output, stream)

    end_evt.record(stream)
    end_evt.synchronize() # Wait until all GPU operations in this stream finish (prevents partial output)
    t2 = time.time()
    gpu_ms = start_evt.time_till(end_evt)
    # Basic output check
    # reshape to (1, 84, 8400) to make data interpretable
    out = h_output.reshape(OUTPUT_SHAPE)

    t_postprocess_start = time.time()
    # Post-Process the data to complete creating the boxes
    final_boxes, final_scores, final_class_ids = post_process(out, 640, 640)
    t_postprocess_end = time.time()
    # Take max over everything 
    out_max = float(out.max())
    out_mean = float(out.mean())

    # FPS calculation
    curr_time = time.time()
    fps = 1.0 / max(curr_time - prev_time, 1e-6)
    prev_time = curr_time

    postprocess_ms = (t_postprocess_end - t_postprocess_start) * 1000
    preprocess_ms = (t1 - t0) * 1000
    infer_total_ms = (t2 - t1) * 1000  # includes H->D, inference, D->H

    # Logging Latencies for benchmark testing
    frame_count+=1
    # Warmup Frames:
    if frame_count==WARMUP+1:
        bench_start = time.time()

    if frame_count > WARMUP:
        
        lat_pre.append(preprocess_ms)
        lat_gpu.append(gpu_ms)
        lat_post.append(postprocess_ms)
        lat_e2e.append(preprocess_ms + gpu_ms + postprocess_ms)

    if frame_count == WARMUP + LOG_FRAMES:
        bench_end = time.time()
        fps_avg = LOG_FRAMES / (bench_end - bench_start)
        print("FPS avg:", fps_avg)
        print("E2E ms avg:", np.mean(lat_e2e), "p50:", p(lat_e2e, 50), "p95:", p(lat_e2e, 95))
        print("GPU ms avg:", np.mean(lat_gpu), "p50:", p(lat_gpu, 50), "p95:", p(lat_gpu, 95))
        print("Pre ms avg:", np.mean(lat_pre))
        print("Post ms avg:", np.mean(lat_post))
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
