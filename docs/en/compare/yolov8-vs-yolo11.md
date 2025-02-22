---
comments: true
description: Compare YOLOv8 and YOLO11 for object detection. Explore their performance, architecture, and best-use cases to find the right model for your needs.
keywords: YOLOv8, YOLO11, object detection, Ultralytics, YOLO comparison, machine learning, computer vision, inference speed, model accuracy
---

# YOLOv8 vs YOLO11: A Technical Comparison

Comparing Ultralytics YOLOv8 and YOLO11 for object detection reveals advancements in real-time computer vision. Both models, developed by Ultralytics, are designed for speed and accuracy, but cater to slightly different needs and build upon distinct architectural choices. This page provides a detailed technical comparison to help users understand their key differences and ideal applications.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv8", "YOLO11"]'></canvas>

## YOLOv8 Overview

**YOLOv8**, introduced by Ultralytics on 2023-01-10, represents a significant iteration in the YOLO series. Authored by Glenn Jocher, Ayush Chaurasia, and Jing Qiu, YOLOv8 focuses on versatility and ease of use across a range of vision tasks including object detection, segmentation, classification, and pose estimation. It is built upon previous YOLO architectures but incorporates enhancements for improved performance and flexibility. The [YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/) emphasizes its user-friendliness, making it accessible for both beginners and experienced practitioners in the field of [object detection](https://www.ultralytics.com/glossary/object-detection).

**Architecture and Key Features:**

YOLOv8 maintains a single-stage, anchor-free detection paradigm, streamlining the model architecture and simplifying the training process. Key architectural features include:

- **Backbone:** Utilizes a refined CSPDarknet backbone, optimized for feature extraction efficiency.
- **Neck:** Employs a C2f cross-stage partial network in the neck for enhanced feature fusion, improving information flow and gradient propagation.
- **Head:** A decoupled detection head separates classification and regression tasks, contributing to improved accuracy and faster convergence.

**Performance Metrics:**

YOLOv8 achieves state-of-the-art performance across various model sizes. For object detection on the COCO dataset, YOLOv8x, the largest variant, reaches 53.9 mAP<sup>val</sup><sub>50-95</sub>, while the nano version, YOLOv8n, achieves 37.3 mAP<sup>val</sup><sub>50-95</sub>, balancing accuracy and speed. Inference speeds range from 80.4 ms on CPU ONNX for YOLOv8n to 479.1 ms for YOLOv8x, providing options for different computational constraints. Explore detailed [YOLOv8 performance metrics](https://docs.ultralytics.com/models/yolov8/#performance-metrics).

**Use Cases:**

YOLOv8’s versatility makes it suitable for a broad spectrum of applications, from [security alarm systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8) and [smart city](https://www.ultralytics.com/blog/computer-vision-ai-in-smart-cities) deployments to advanced applications in [healthcare](https://www.ultralytics.com/solutions/ai-in-healthcare) and [manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing). Its balanced performance makes it a strong choice for projects requiring a blend of accuracy and speed.

**Strengths:**

- **Versatile Task Support:** Handles detection, segmentation, classification, and pose estimation.
- **High Accuracy and Speed:** Offers a good balance between mAP and inference speed.
- **User-Friendly:** Well-documented and easy to use with Ultralytics [Python](https://docs.ultralytics.com/usage/python/) and [CLI](https://docs.ultralytics.com/usage/cli/) interfaces.
- **Active Community:** Benefits from a large community and continuous updates from Ultralytics.

**Weaknesses:**

- **Resource Intensive:** Larger models require significant computational resources for training and deployment.
- **Optimization Needs:** May require further optimization for extremely resource-constrained environments.

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## YOLO11 Overview

**YOLO11**, the latest model from Ultralytics, released on 2024-09-27, and authored by Glenn Jocher and Jing Qiu, builds upon the YOLO series, aiming for further advancements in efficiency and performance. While sharing the core philosophy of speed and accuracy, YOLO11 introduces architectural refinements designed to optimize inference speed without significantly compromising accuracy. The [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/) highlights its cutting-edge nature and suitability for real-time applications.

**Architecture and Key Features:**

YOLO11 also adopts a single-stage, anchor-free approach, focusing on streamlined design and efficient computation. Key architectural aspects include:

- **Efficient Backbone:** Employs an optimized backbone architecture that reduces computational overhead while maintaining feature extraction capabilities.
- **Neck Design:** Features a refined neck structure that enhances feature aggregation with fewer parameters, contributing to faster inference.
- **Optimized Head:** The detection head is engineered for minimal latency, prioritizing speed in the final prediction layers.

**Performance Metrics:**

YOLO11 demonstrates competitive performance with a focus on speed improvements. For object detection on the COCO dataset, YOLO11x achieves a slightly higher mAP<sup>val</sup><sub>50-95</sub> of 54.7 compared to YOLOv8x, while maintaining faster inference speeds on CPU and achieving comparable speeds on GPU. The YOLO11n model achieves 39.5 mAP<sup>val</sup><sub>50-95</sub>, showing improvements over YOLOv8n. Inference speeds are notably faster on CPU, with YOLO11n at 56.1 ms and YOLO11x at 462.8 ms, making it suitable for CPU-bound applications. Refer to [YOLO11 performance metrics](https://docs.ultralytics.com/models/yolo11/#performance-metrics) for detailed benchmarks.

**Use Cases:**

YOLO11 is particularly well-suited for applications where inference speed is paramount, such as real-time video analysis, robotics, and edge devices with limited computational resources. Its efficiency makes it ideal for deployment in scenarios requiring rapid object detection without sacrificing accuracy. Applications include [waste management](https://www.ultralytics.com/blog/enhancing-waste-management-with-ultralytics-yolo11), [environmental conservation](https://www.ultralytics.com/blog/ultralytics-yolo11-and-computer-vision-for-environmental-conservation), and [automotive solutions](https://www.ultralytics.com/blog/ultralytics-yolo11-and-computer-vision-for-automotive-solutions).

**Strengths:**

- **Superior Inference Speed:** Designed for faster inference, especially on CPU.
- **Competitive Accuracy:** Maintains high accuracy, often exceeding YOLOv8 in smaller model sizes.
- **Efficient Architecture:** Optimized for resource-constrained environments and edge deployment.
- **Latest Ultralytics Model:** Benefits from the most recent advancements and support from Ultralytics.

**Weaknesses:**

- **Marginal Accuracy Gains in Larger Models:** Larger YOLO11 models show only slight accuracy improvements over YOLOv8 while still being computationally intensive.
- **Newer Model:** Being newer, it may have a smaller community and fewer third-party integrations compared to YOLOv8.

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

## Model Comparison Table

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
|---------|-----------------------|----------------------|--------------------------------|-------------------------------------|--------------------|-------------------|
| YOLOv8n | 640                   | 37.3                 | 80.4                           | 1.47                                | 3.2                | 8.7               |
| YOLOv8s | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |
|         |                       |                      |                                |                                     |                    |                   |
| YOLO11n | 640                   | 39.5                 | 56.1                           | 1.5                                 | 2.6                | 6.5               |
| YOLO11s | 640                   | 47.0                 | 90.0                           | 2.5                                 | 9.4                | 21.5              |
| YOLO11m | 640                   | 51.5                 | 183.2                          | 4.7                                 | 20.1               | 68.0              |
| YOLO11l | 640                   | 53.4                 | 238.6                          | 6.2                                 | 25.3               | 86.9              |
| YOLO11x | 640                   | 54.7                 | 462.8                          | 11.3                                | 56.9               | 194.9             |

## Conclusion

Choosing between YOLOv8 and YOLO11 depends on the specific application requirements. YOLOv8 offers a robust and versatile solution suitable for a wide array of tasks, balancing accuracy and speed effectively. It’s a mature and well-supported model ideal for general-purpose object detection needs. YOLO11, on the other hand, is engineered for speed optimization, making it a superior choice when inference time is critical, especially in CPU-intensive or edge computing scenarios. For applications demanding the fastest possible real-time performance with competitive accuracy, YOLO11 is the preferred option.

Users interested in exploring other models may also consider:

- **YOLOv5**: For a well-established and widely-used model with a large community. [YOLOv5 vs YOLOv8 comparison](https://docs.ultralytics.com/compare/yolov5-vs-yolov8/).
- **YOLOv9**: For models focusing on accuracy improvements and architectural innovations. [YOLOv9 documentation](https://docs.ultralytics.com/models/yolov9/).
- **FastSAM**: For extremely fast segmentation tasks. [FastSAM documentation](https://docs.ultralytics.com/models/fast-sam/).
