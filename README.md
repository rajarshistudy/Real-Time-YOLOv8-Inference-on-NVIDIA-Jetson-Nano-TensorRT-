# Real-Time-YOLOv8-Inference-on-NVIDIA-Jetson-Nano-TensorRT-
Real-time object detection pipeline using a TensorRT-optimized YOLOv8 model on NVIDIA Jetson Nano, featuring raw-output decoding, CUDA-based profiling, and performance analysis.

## Demo

<img width="612" height="729" alt="image" src="https://github.com/user-attachments/assets/5542ea3e-0ea0-492d-8a48-be4411797f4a" />

The demo script performs live webcam inference, drawing bounding boxes and class labels in real time.
The script classifies objects from the COCO 80 class labels for YOLOv8 engines. 
Along with the classification, the model also labels the confidence score for each bounding box it generates.

python3 trt_webcam_infer_raw.py


## System Architecture

The inference pipeline follows the sequence below:

1. Webcam frame capture (OpenCV)
2. CPU preprocessing (reshape, color conversion, normalization)
3. Host-to-device memory copy (Asynchronous) (CUDA)
4. TensorRT inference (YOLOv8 engine)
5. Device-to-host memory copy (CUDA)
6. CPU post-processing (YOLO decoding, confidence filtering, NMS)
7. Visualization and display


## Challenged Faced

Initial detections produced bounding boxes that were highly inaccurate despite the model running successfully. 
The issue was traced to a mismatch between model output assumptions and post-processing logic. 
(The bouding boxes were too big resulting in the bounds exceeded the size of the image)

Specifically:
- The TensorRT-exported YOLOv8 engine outputs bounding box coordinates in pixel space rather than normalized [0,1] coordinates.
- The original post-processing logic incorrectly scaled these values by the input resolution (in an attempt to convert normalized coordinates to pixel coordinates when coordinates were already in pixel scale), causing boxes to be projected far outside the image.
- Additionally, class confidence outputs required sigmoid activation before thresholding.

Correcting the coordinate interpretation and applying proper activation resolved the issue, resulting in accurate and stable detections.


## Performance Benchmarking

Benchmarking was performed using a dedicated script with warmup frames and CUDA event timing to isolate GPU execution from CPU overhead.

Configuration:
- Model: YOLOv8n (TensorRT engine)
- Input resolution: 640×640
- Precision: FP32
- Warmup frames: 50
- Measured frames: 300
- Platform: NVIDIA Jetson Nano


### Latency Breakdown (Average)

| Stage           | Latency (ms) |
|-----------------|--------------|
| Preprocessing   | ~5.7         |
| GPU Inference   | ~21.2        |
| Post-processing | ~4.8         |
| **End-to-End**  | **~31.7**    |

<img width="821" height="164" alt="image" src="https://github.com/user-attachments/assets/d82f4993-30a3-4ed9-8f7a-89d15f04bf31" />



### Latency Distribution

- End-to-end latency p50: ~31.9 ms
- End-to-end latency p95: ~33.6 ms

The narrow gap between average and p95 latency indicates stable real-time performance with minimal jitter.


### Performance Analysis

While TensorRT inference accounts for the majority of execution time, approximately one-third of the total latency originates from CPU-side preprocessing and post-processing. This indicates that the system is CPU-bound rather than GPU-bound, suggesting further optimization opportunities outside the inference engine itself.

## Repository Structure

- `trt_webcam_infer_raw.py` – Live webcam demo
- `benchmark.py` – Headless performance benchmarking script
- `yolo_postprocess.py` – YOLOv8 raw-output decoding and NMS
- `yolov8n.engine` – TensorRT engine (not included in repository, engine must be generated locally)
- `coco.names` – Class labels

## Possible Improvements

- Use pinned host memory to reduce host–device copy latency
- Move preprocessing to GPU for reduced CPU overhead
- INT8 quantization with calibration for higher throughput
- Multi-threaded post-processing
