---
comments: true
description: Discover the technical differences, performance metrics, and applications of YOLO11 and YOLOv9. Choose the best object detection model for your needs.
keywords: YOLO11, YOLOv9, object detection, Ultralytics, computer vision, deep learning, model comparison, accuracy, efficiency, real-time AI
---

# YOLO11 vs YOLOv9: A Technical Comparison for Object Detection

Ultralytics continuously advances the field of computer vision with state-of-the-art YOLO models. This page provides a detailed technical comparison between two of our cutting-edge object detection models: [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) and [YOLOv9](https://docs.ultralytics.com/models/yolov9/). We will analyze their architectures, performance metrics, and ideal applications to help you choose the best model for your specific needs.

<script async src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO11", "YOLOv9"]'></canvas>

## Ultralytics YOLO11: Redefining Accuracy and Efficiency

[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) is the latest iteration in the YOLO series, building upon the successes of previous versions like [YOLOv8](https://docs.ultralytics.com/models/yolov8/). YOLO11 is designed to achieve higher accuracy and efficiency for various computer vision tasks, including object detection, [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation), [image classification](https://docs.ultralytics.com/tasks/classify/), and [pose estimation](https://docs.ultralytics.com/tasks/pose/).

**Architecture and Key Features:**
YOLO11 boasts an innovative architecture focused on enhanced feature extraction and optimized processing speeds. It achieves greater accuracy with fewer parameters compared to its predecessors, making it more efficient for real-time applications and deployment on diverse platforms, from edge devices like [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/) and [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/) to cloud systems. YOLO11 supports the same tasks as YOLOv8, ensuring a seamless transition for existing users.

**Performance Metrics:**
As shown in the comparison table, YOLO11 models offer a range of sizes and performance levels. For instance, YOLO11m achieves a mAP<sup>val</sup><sub>50-95</sub> of 51.5 at 640 size, with a CPU ONNX speed of 183.2ms and a T4 TensorRT10 speed of 4.7ms. The model size is 20.1M parameters and 68.0B FLOPs. This balance of accuracy and speed makes YOLO11 a versatile choice for many applications.

**Use Cases:**
YOLO11 excels in scenarios demanding high accuracy and real-time performance, such as:

- **Smart Cities:** [Traffic management](https://www.ultralytics.com/blog/optimizingtraffic-management-with-ultralytics-yolo11), [security systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8), and [crowd management](https://www.ultralytics.com/blog/vision-ai-in-crowd-management).
- **Healthcare:** [Medical image analysis](https://www.ultralytics.com/glossary/medical-image-analysis) for tumor detection and diagnostics.
- **Manufacturing:** [Quality control](https://www.ultralytics.com/solutions/ai-in-manufacturing) and automation in production lines.
- **Agriculture:** [Crop health monitoring](https://www.ultralytics.com/blog/real-time-crop-health-monitoring-with-ultralytics-yolo11) and precision farming.

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

## YOLOv9: Pushing the Boundaries of Real-time Object Detection

[YOLOv9](https://docs.ultralytics.com/models/yolov9/) represents another significant advancement in real-time object detection. It introduces innovations like Programmable Gradient Information (PGI) and Generalized Efficient Layer Aggregation Network (GELAN) to achieve state-of-the-art efficiency and accuracy.

**Architecture and Key Features:**
YOLOv9's architecture focuses on maintaining high accuracy while being incredibly efficient. The PGI mechanism ensures that gradient information is reliable, leading to better learning. GELAN helps in designing efficient network architectures that reduce parameter count and computational cost without compromising accuracy.

**Performance Metrics:**
YOLOv9 also offers various model sizes to cater to different computational constraints. YOLOv9m, for example, achieves a mAP<sup>val</sup><sub>50-95</sub> of 51.4 at 640 size, with a T4 TensorRT10 speed of 6.43ms and a model size of 20.0M parameters. While specific CPU speeds are not listed in the provided table, YOLOv9 is designed for efficient inference, particularly on GPUs and accelerators like TensorRT.

**Use Cases:**
YOLOv9 is ideally suited for applications where speed and efficiency are paramount, such as:

- **Edge AI Applications:** Deployment on resource-constrained devices for real-time processing.
- **Robotics:** [Robot vision](https://www.ultralytics.com/glossary/robotics) for navigation and interaction in dynamic environments.
- **Surveillance:** High-speed object detection in [security systems](https://www.ultralytics.com/blog/computer-vision-for-theft-prevention-enhancing-security) requiring rapid response.
- **Autonomous Vehicles:** [Self-driving car](https://www.ultralytics.com/solutions/ai-in-self-driving) perception systems needing low latency.

## Model Comparison Table

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLO11n | 640                   | 39.5                 | 56.1                           | 1.5                                 | 2.6                | 6.5               |
| YOLO11s | 640                   | 47.0                 | 90.0                           | 2.5                                 | 9.4                | 21.5              |
| YOLO11m | 640                   | 51.5                 | 183.2                          | 4.7                                 | 20.1               | 68.0              |
| YOLO11l | 640                   | 53.4                 | 238.6                          | 6.2                                 | 25.3               | 86.9              |
| YOLO11x | 640                   | 54.7                 | 462.8                          | 11.3                                | 56.9               | 194.9             |
|         |                       |                      |                                |                                     |                    |                   |
| YOLOv9t | 640                   | 38.3                 | -                              | 2.3                                 | 2.0                | 7.7               |
| YOLOv9s | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e | 640                   | 55.6                 | -                              | 16.77                               | 57.3               | 189.0             |

## Strengths and Weaknesses

**YOLO11 Strengths:**

- **High Accuracy:** Achieves excellent mAP scores, making it suitable for applications requiring precise object detection.
- **Versatility:** Supports multiple vision tasks beyond object detection.
- **Optimized Architecture:** Fewer parameters than previous models, leading to better efficiency.

**YOLO11 Weaknesses:**

- **Inference Speed:** While efficient, the larger YOLO11 models might be slower on less powerful hardware compared to the fastest YOLOv9 variants.

**YOLOv9 Strengths:**

- **Real-time Performance:** Designed for exceptional speed, making it ideal for latency-sensitive applications.
- **Efficiency:** GELAN architecture ensures fewer parameters and lower computational cost.
- **Cutting-edge Innovations:** PGI and GELAN contribute to improved training and performance.

**YOLOv9 Weaknesses:**

- **Limited Task Support:** Primarily focused on object detection; broader task support like YOLO11 might require further development.
- **CPU Speed Data:** CPU speed metrics are not available in the provided table, which could be relevant for CPU-based deployments.

## Conclusion

Both YOLO11 and YOLOv9 represent significant advancements in object detection. YOLO11 prioritizes accuracy and versatility, making it a robust choice for a wide range of applications where precision is crucial. YOLOv9, on the other hand, excels in real-time performance and efficiency, making it perfect for edge deployment and high-speed processing needs.

For users seeking a balance of accuracy and speed with multi-task capabilities, YOLO11 is an excellent choice. For applications where real-time inference and computational efficiency are the primary concerns, YOLOv9 offers superior performance. Consider exploring other models like [YOLOv10](https://docs.ultralytics.com/models/yolov10/) and [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/) in the Ultralytics [model zoo](https://docs.ultralytics.com/models/) to find the perfect fit for your specific project requirements.
