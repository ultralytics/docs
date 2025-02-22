---
comments: true
description: Compare YOLOv10 and YOLOv9 object detection models. Explore architectures, metrics, and use cases to choose the best model for your application.
keywords: YOLOv10,YOLOv9,Ultralytics,object detection,real-time AI,computer vision,model comparison,AI deployment,deep learning
---

# YOLOv10 vs YOLOv9: A Technical Comparison for Object Detection

Choosing the right object detection model is crucial for computer vision projects. Ultralytics offers state-of-the-art YOLO models, and this page provides a detailed technical comparison between two cutting-edge options: [Ultralytics YOLOv10](https://docs.ultralytics.com/models/yolov10/) and [YOLOv9](https://docs.ultralytics.com/models/yolov9/). We will analyze their architectures, performance metrics, and ideal applications to help you make an informed decision.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv10", "YOLOv9"]'></canvas>

## Ultralytics YOLOv10: Real-Time End-to-End Efficiency

[Ultralytics YOLOv10](https://docs.ultralytics.com/models/yolov10/) is the latest evolution in the YOLO series, developed by researchers at Tsinghua University. Released in May 2024, YOLOv10 focuses on maximizing efficiency and speed for real-time object detection without sacrificing accuracy. This model is particularly designed for end-to-end deployment, minimizing latency by employing NMS-free training techniques.

**Architecture and Key Features:**

YOLOv10 introduces several architectural innovations to enhance both efficiency and accuracy:

- **Consistent Dual Assignments**: This method combines one-to-many and one-to-one label assignment strategies during training, ensuring rich supervision and NMS-free inference, reducing post-processing overhead and latency.
- **Holistic Efficiency-Accuracy Driven Design**: This strategy involves several key components:
    - **Lightweight Classification Head**: Reduces computational cost using depth-wise separable convolutions.
    - **Spatial-Channel Decoupled Downsampling**: Minimizes information loss and computational cost during downsampling.
    - **Rank-Guided Block Design**: Optimizes parameter utilization based on stage redundancy.
    - **Large-Kernel Convolution**: Enhances feature extraction with larger receptive fields.
    - **Partial Self-Attention (PSA)**: Improves global representation learning with minimal overhead.

**Performance Metrics:**

YOLOv10 demonstrates superior performance and efficiency compared to previous YOLO versions and other state-of-the-art models. For instance, YOLOv10-S is reported to be 1.8x faster than RT-DETR-R18 with similar Average Precision (AP) on the COCO dataset. Compared to YOLOv9-C, YOLOv10-B achieves 46% less latency and 25% fewer parameters for comparable performance. Detailed metrics can be found in the comparison table below and the [official YOLOv10 documentation](https://docs.ultralytics.com/models/yolov10/).

**Use Cases:**

YOLOv10 is ideally suited for applications that demand high speed and efficiency, especially on resource-constrained devices:

- **Edge Computing**: Real-time object detection on edge devices like [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/) and mobile platforms.
- **High-Speed Applications**: Scenarios requiring minimal latency, such as autonomous driving and robotics.
- **Efficient Deployment**: Applications where model size and computational cost are critical, like mobile and embedded systems.

**Strengths:**

- **High Efficiency and Speed**: Optimized for real-time performance and low latency.
- **NMS-Free Training**: Enables end-to-end deployment and reduces inference time.
- **State-of-the-art Performance**: Outperforms previous YOLO versions in speed and parameter efficiency while maintaining competitive accuracy.

**Weaknesses:**

- **Relatively New Model**: Being a recent model, it may have a smaller community and fewer deployment examples compared to more mature models like [YOLOv8](https://docs.ultralytics.com/models/yolov8/).
- **Performance Trade-offs**: While highly efficient, the focus on speed might lead to slightly lower accuracy in certain complex scenarios compared to larger, more computationally intensive models.

[Learn more about YOLOv10](https://docs.ultralytics.com/models/yolov10/){ .md-button }

## YOLOv9: Programmable Gradient Information for Enhanced Learning

[YOLOv9](https://docs.ultralytics.com/models/yolov9/), introduced in February 2024 by researchers at the Institute of Information Science, Academia Sinica, Taiwan, focuses on improving information learning and handling information loss during network propagation. YOLOv9 introduces the concept of Programmable Gradient Information (PGI) to address information degradation, leading to more accurate and reliable object detection.

**Architecture and Key Features:**

YOLOv9's architecture is built upon the foundation of previous YOLO models but incorporates significant innovations:

- **Programmable Gradient Information (PGI)**: This key innovation consists of two main components:

    - **Generalized Efficient Layer Aggregation Network (GELAN)**: A highly efficient network structure that reduces parameter count and computational load while maintaining accuracy. GELAN serves as the backbone of YOLOv9, enabling faster training and inference.
    - **Auxiliary Reversible Branch (ARB)**: Designed to preserve complete information and provide reliable gradients to the main branch, ARB helps the model learn more effectively from the data, especially deep within the network.

- **Comprehensive Data Learning**: By preserving gradient information, YOLOv9 can learn more effectively from the training data, leading to improved accuracy and robustness.

**Performance Metrics:**

YOLOv9 achieves state-of-the-art performance, particularly in balancing accuracy and computational cost. As shown in the comparison table, YOLOv9 models like YOLOv9-C achieve a mAPval50-95 of 53.0% at 640 size with a relatively low parameter count. Detailed performance metrics are available in the [YOLOv9 GitHub repository](https://github.com/WongKinYiu/yolov9) and the table below.

**Use Cases:**

YOLOv9 is well-suited for applications requiring high accuracy and robustness, particularly in scenarios where information preservation is critical:

- **High-Accuracy Detection**: Applications that demand precise object detection, such as security and surveillance systems.
- **Complex Scenes**: Environments with dense objects or occlusions where information retention is crucial for accurate detection.
- **Research and Development**: A strong baseline model for further research in object detection and network architecture design.

**Strengths:**

- **High Accuracy**: Achieves excellent mAP scores, particularly due to the PGI and GELAN architectures that enhance information learning and preservation.
- **Efficient Architecture**: GELAN reduces computational overhead, making YOLOv9 efficient in terms of parameters and FLOPs compared to its performance.
- **Innovative PGI**: Addresses the problem of information loss, leading to more robust and reliable learning.

**Weaknesses:**

- **Inference Speed**: While efficient, YOLOv9 might not reach the same inference speeds as YOLOv10, especially in NMS-free scenarios, as YOLOv9 still relies on NMS post-processing.
- **Complexity**: The PGI and ARB mechanisms add architectural complexity, potentially making implementation and optimization more challenging compared to simpler models.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

<br>

| Model    | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| -------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv10n | 640                   | 39.5                 | -                              | 1.56                                | 2.3                | 6.7               |
| YOLOv10s | 640                   | 46.7                 | -                              | 2.66                                | 7.2                | 21.6              |
| YOLOv10m | 640                   | 51.3                 | -                              | 5.48                                | 15.4               | 59.1              |
| YOLOv10b | 640                   | 52.7                 | -                              | 6.54                                | 24.4               | 92.0              |
| YOLOv10l | 640                   | 53.3                 | -                              | 8.33                                | 29.5               | 120.3             |
| YOLOv10x | 640                   | 54.4                 | -                              | 12.2                                | 56.9               | 160.4             |
|          |                       |                      |                                |                                     |                    |                   |
| YOLOv9t  | 640                   | 38.3                 | -                              | 2.3                                 | 2.0                | 7.7               |
| YOLOv9s  | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m  | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c  | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e  | 640                   | 55.6                 | -                              | 16.77                               | 57.3               | 189.0             |

## Choosing Between YOLOv10 and YOLOv9

- **For Real-time and Edge Applications**: If your priority is speed and efficiency, especially for deployment on edge devices or in real-time systems, [YOLOv10](https://docs.ultralytics.com/models/yolov10/) is the better choice. Its NMS-free design and efficiency-focused architecture provide significant advantages in latency and computational cost.
- **For High Accuracy and Complex Scenarios**: If accuracy and robustness are paramount, particularly in complex detection scenarios or when information preservation is critical, [YOLOv9](https://docs.ultralytics.com/models/yolov9/) is more suitable. Its PGI and GELAN innovations enhance learning and lead to better performance in demanding tasks.

Both YOLOv10 and YOLOv9 represent significant advancements in object detection. Your choice should be guided by the specific requirements of your application, balancing the need for speed and efficiency against the demand for accuracy and robustness.

For users interested in other cutting-edge models, [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) offers a blend of accuracy and efficiency, while [YOLOv8](https://docs.ultralytics.com/models/yolov8/) remains a versatile and widely-adopted choice for a broad range of applications.
