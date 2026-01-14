import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from ultralytics import YOLO
from yolo_postprocess import post_process  # initializes CUDA context


with open("coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
assert len(class_names) == 80


# Engine details (from your trt_engine_info.py)
ENGINE_PATH = "yolov8n.engine"
INPUT_NAME = "images"
OUTPUT_NAME = "output0"
INPUT_SHAPE = (1, 3, 640, 640)      # NCHW
OUTPUT_SHAPE = (1, 84, 8400)        # YOLOv8 raw output
DTYPE = np.float32


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

    # Copy input to GPU using PyCUDA (host to device)
    # Note that this is asynchronous to allow computation overlap and better performance
    cuda.memcpy_htod_async(d_input, input_tensor.ravel(), stream)
    # Execute inference
    ok = context.execute_async_v3(stream_handle=stream.handle)
    if not ok:
        raise RuntimeError("TensorRT execute_async_v3 failed")
    # Copy output back to CPU again using PyCUDA this time using devide to host
    cuda.memcpy_dtoh_async(h_output, d_output, stream)

    stream.synchronize() # Wait until all GPU operations in this stream finish (prevents partial output)
    t2 = time.time()

    # Basic output check
    # reshape to (1, 84, 8400) to make data interpretable
    out = h_output.reshape(OUTPUT_SHAPE)

    # Post-Process the data to complete creating the boxes
    final_boxes, final_scores, final_class_ids = post_process(out, 640, 640)

    # Take max over everything 
    out_max = float(out.max())
    out_mean = float(out.mean())

    # FPS calculation
    curr_time = time.time()
    fps = 1.0 / max(curr_time - prev_time, 1e-6)
    prev_time = curr_time


    preprocess_ms = (t1 - t0) * 1000
    infer_total_ms = (t2 - t1) * 1000  # includes H->D, inference, D->H


    cv2.putText(resized, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(resized, f"Preprocess: {preprocess_ms:.1f} ms", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(resized, f"Infer(total): {infer_total_ms:.1f} ms", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(resized, f"out_max: {out_max:.3f}  out_mean: {out_mean:.3f}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if final_boxes:
        print(final_boxes[0])
        for box, score, cls_id in zip(final_boxes, final_scores, final_class_ids):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = class_names[int(cls_id)]
            text = f"{label}: {score:.2f}"
            cv2.putText(resized, text, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("TensorRT YOLOv8 (Raw Output)", resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
