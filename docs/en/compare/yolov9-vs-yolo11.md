---
comments: true
description: Compare YOLO11 and YOLOv9 for object detection. Explore innovations, benchmarks, and use cases to select the best model for your tasks.
keywords: YOLO11, YOLOv9, object detection, model comparison, benchmarks, Ultralytics, real-time processing, machine learning, computer vision
---

# YOLO11 vs YOLOv9: A Technical Comparison for Object Detection

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv9", "YOLO11"]'></canvas>

Ultralytics consistently delivers state-of-the-art YOLO models, pushing the boundaries of real-time object detection. This page provides a technical comparison between two advanced models: [YOLOv9](https://docs.ultralytics.com/models/yolov9/) and [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/). We analyze their architectural innovations, performance benchmarks, and suitable applications to guide you in selecting the optimal model for your computer vision tasks.

## YOLO11: Redefining Accuracy and Efficiency

[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/), the newest iteration in the YOLO series, builds upon previous successes like [YOLOv8](https://docs.ultralytics.com/models/yolov8/). YOLO11 is engineered for enhanced accuracy and efficiency across various computer vision tasks, including object detection, [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation), [image classification](https://docs.ultralytics.com/tasks/classify/), and [pose estimation](https://docs.ultralytics.com/tasks/pose/).

**Architecture and Key Features:**
YOLO11 features an architecture designed for improved feature extraction and faster processing. It achieves higher accuracy with fewer parameters than predecessors, enhancing real-time performance and enabling deployment across diverse platforms, from edge devices like [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/) and [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/) to cloud infrastructure. YOLO11 maintains task compatibility with YOLOv8, ensuring a smooth transition for existing users.

**Performance Metrics:**
YOLO11 models offer a spectrum of sizes and performance capabilities. For example, YOLO11m achieves a mAPval50-95 of 51.5 at a 640 image size, with a CPU ONNX speed of 183.2ms and a T4 TensorRT10 speed of 4.7ms. The model size is 20.1M parameters and 68.0B FLOPs. This balance positions YOLO11 as a versatile choice for numerous applications requiring both precision and speed. For detailed metrics, refer to the [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/).

**Use Cases:**
YOLO11 is ideal for applications demanding high accuracy and real-time processing:

- **Smart Cities**: For [traffic management](https://www.ultralytics.com/blog/optimizingtraffic-management-with-ultralytics-yolo11) and [security systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8).
- **Healthcare**: In [medical image analysis](https://www.ultralytics.com/glossary/medical-image-analysis) for diagnostic support.
- **Manufacturing**: For [quality control](https://www.ultralytics.com/solutions/ai-in-manufacturing) in automated production lines.
- **Agriculture**: In [crop health monitoring](https://www.ultralytics.com/blog/real-time-crop-health-monitoring-with-ultralytics-yolo11) for precision agriculture.

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

## YOLOv9: Programmable Gradient Information

[YOLOv9](https://docs.ultralytics.com/models/yolov9/), introduced in early 2024, is authored by Chien-Yao Wang and Hong-Yuan Mark Liao from the Institute of Information Science, Academia Sinica, Taiwan. This model introduces innovations to address information loss in deep networks, enhancing both efficiency and accuracy.

**Architecture and Key Features:**
YOLOv9 is built upon the Generalized Efficient Layer Aggregation Network (GELAN) and Programmable Gradient Information (PGI). PGI is designed to preserve complete information for gradient computation, preventing information loss during deep network propagation. GELAN serves as an efficient network architecture that reduces parameter count while maintaining accuracy. These innovations allow YOLOv9 to achieve state-of-the-art performance with fewer parameters and computations. The original paper is available on [arXiv](https://arxiv.org/abs/2402.13616) and the implementation on [GitHub](https://github.com/WongKinYiu/yolov9).

**Performance Metrics:**
YOLOv9 demonstrates superior performance on the COCO dataset compared to other real-time object detectors. For instance, YOLOv9c achieves a mAPval50-95 of 53.0. The model variants range in size and complexity, with YOLOv9t being a tiny model at 2.0M parameters and YOLOv9e being a larger model at 57.3M parameters. The model's efficiency is highlighted by its reduced FLOPs, making it suitable for resource-constrained environments. For detailed performance metrics, see the [YOLOv9 documentation](https://docs.ultralytics.com/models/yolov9/).

**Use Cases:**
YOLOv9 is particularly suited for scenarios where computational efficiency is critical without sacrificing accuracy:

- **Edge Computing**: Deployment on edge devices with limited resources.
- **Real-time Applications**: Scenarios requiring fast inference times such as autonomous systems.
- **High-Resolution Imagery**: Effective in processing high-resolution images due to efficient information preservation.
- **Mobile Applications**: Ideal for mobile applications due to its lightweight nature and efficiency.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

## Model Comparison Table

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv9t | 640                   | 38.3                 | -                              | 2.3                                 | 2.0                | 7.7               |
| YOLOv9s | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e | 640                   | 55.6                 | -                              | 16.77                               | 57.3               | 189.0             |
|         |                       |                      |                                |                                     |                    |                   |
| YOLO11n | 640                   | 39.5                 | 56.1                           | 1.5                                 | 2.6                | 6.5               |
| YOLO11s | 640                   | 47.0                 | 90.0                           | 2.5                                 | 9.4                | 21.5              |
| YOLO11m | 640                   | 51.5                 | 183.2                          | 4.7                                 | 20.1               | 68.0              |
| YOLO11l | 640                   | 53.4                 | 238.6                          | 6.2                                 | 25.3               | 86.9              |
| YOLO11x | 640                   | 54.7                 | 462.8                          | 11.3                                | 56.9               | 194.9             |

Both YOLOv9 and YOLO11 represent significant advancements in object detection. YOLOv9 excels in scenarios prioritizing computational efficiency and reduced parameters, while YOLO11 is optimized for high accuracy and versatile task support within the Ultralytics ecosystem. Your choice should align with the specific demands of your application, whether it be resource constraints or accuracy requirements.

Users might also be interested in exploring other models such as [YOLOv8](https://docs.ultralytics.com/models/yolov8/) and [YOLOv7](https://docs.ultralytics.com/models/yolov7/) for different performance and architectural trade-offs.
