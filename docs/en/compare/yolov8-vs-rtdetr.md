---
comments: true
description: Explore the detailed comparison of YOLOv8 and RTDETRv2 models for object detection. Discover their architecture, performance, and best use cases.
keywords: YOLOv8,RTDETRv2,object detection,model comparison,performance metrics,real-time detection,transformer-based models,computer vision,Ultralytics
---

# YOLOv8 vs RTDETRv2: A Technical Comparison

Choosing the right object detection model involves a trade-off between accuracy, speed, and computational cost. This page provides a detailed technical comparison between two powerful models: [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/), a state-of-the-art model from the YOLO family, and RTDETRv2, a real-time detection transformer from Baidu. While both models offer excellent performance, they are built on fundamentally different architectural principles, making them suitable for different applications.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv8", "RTDETRv2"]'></canvas>

## Ultralytics YOLOv8: The Versatile and Efficient Standard

[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) is the latest long-term support (LTS) model in the highly successful YOLO series. It builds upon the innovations of its predecessors, delivering exceptional performance while prioritizing ease of use, speed, and versatility.

**Technical Details:**

- **Authors:** Glenn Jocher, Ayush Chaurasia, and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com/)
- **Date:** 2023-01-10
- **GitHub:** <https://github.com/ultralytics/ultralytics>
- **Docs:** <https://docs.ultralytics.com/models/yolov8/>

### Architecture and Key Features

YOLOv8 features a highly optimized, anchor-free, single-stage architecture. It uses a CSPDarknet53-inspired backbone for efficient feature extraction and a C2f (Cross Stage Partial Bottlebeck with 2 convolutions) module in the neck to enhance feature fusion. This design results in a model that is not only fast and accurate but also computationally efficient.

A key advantage of YOLOv8 is its integration into the comprehensive [Ultralytics ecosystem](https://www.ultralytics.com/blog/ultralytics-hub-making-ml-accessible). This provides a **streamlined user experience** with a simple [Python API](https://docs.ultralytics.com/usage/python/) and [CLI](https://docs.ultralytics.com/usage/cli/), extensive [documentation](https://docs.ultralytics.com/), and active community support.

### Strengths

- **Performance Balance:** YOLOv8 achieves an outstanding trade-off between speed and accuracy, making it suitable for a wide range of [real-world deployment scenarios](https://www.ultralytics.com/solutions), from high-performance cloud servers to resource-constrained [edge devices](https://docs.ultralytics.com/guides/nvidia-jetson/).
- **Ease of Use:** The model is incredibly user-friendly, with straightforward workflows for [training](https://docs.ultralytics.com/modes/train/), validation, and deployment. The well-maintained ecosystem includes tools like [Ultralytics HUB](https://www.ultralytics.com/hub) for no-code training and MLOps management.
- **Versatility:** Unlike RTDETRv2, which is primarily an object detector, YOLOv8 is a multi-task model supporting [object detection](https://docs.ultralytics.com/tasks/detect/), [instance segmentation](https://docs.ultralytics.com/tasks/segment/), [image classification](https://docs.ultralytics.com/tasks/classify/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), and [oriented object detection (OBB)](https://docs.ultralytics.com/tasks/obb/) within a single, unified framework.
- **Training and Memory Efficiency:** YOLOv8 is designed for efficient training processes with readily available pre-trained weights. Compared to transformer-based models like RTDETRv2, YOLOv8 typically requires significantly less CUDA memory and converges faster, reducing computational costs and development time.

### Weaknesses

- While highly accurate, the largest transformer-based models may achieve slightly higher mAP on certain complex datasets with dense objects, though this often comes at the cost of much higher latency and resource requirements.

### Ideal Use Cases

YOLOv8's balance of speed, accuracy, and versatility makes it ideal for:

- **Real-Time Applications:** Video surveillance, [security systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8), and [traffic management](https://www.ultralytics.com/blog/ai-in-traffic-management-from-congestion-to-coordination).
- **Edge Computing:** Deployment on devices like [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/) and [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/) where computational resources are limited.
- **Industrial Automation:** For tasks like [quality control](https://www.ultralytics.com/solutions/ai-in-manufacturing) and inventory management.
- **Multi-Task Solutions:** Projects that require more than just object detection, such as combining detection with pose estimation for [fitness applications](https://docs.ultralytics.com/guides/workouts-monitoring/).

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## RTDETRv2: Real-Time Detection with Transformers

RTDETRv2 (Real-Time Detection Transformer v2) is a state-of-the-art object detector from [Baidu](https://www.baidu.com/) that leverages the power of Vision Transformers to achieve high accuracy while maintaining real-time performance on powerful hardware.

**Technical Details:**

- **Authors:** Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu
- **Organization:** Baidu
- **Date:** 2023-04-17 (Initial RT-DETR), 2024-07-24 (RT-DETRv2 improvements)
- **Arxiv:** <https://arxiv.org/abs/2304.08069>
- **GitHub:** <https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch>
- **Docs:** <https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme>

### Architecture and Key Features

RTDETRv2 employs a hybrid architecture, combining a [CNN](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn) backbone (e.g., ResNet) for initial feature extraction with a [Transformer](https://www.ultralytics.com/glossary/transformer)-based encoder-decoder. The transformer's [self-attention mechanism](https://www.ultralytics.com/glossary/self-attention) allows the model to capture global context and long-range dependencies between objects in an image, which can be beneficial for detecting objects in complex or cluttered scenes.

### Strengths

- **High Accuracy:** The transformer architecture enables RTDETRv2 to achieve excellent mAP scores, particularly on complex datasets with many small or occluded objects.
- **Robust Feature Extraction:** Its ability to process global image context leads to strong performance in challenging detection scenarios.
- **Real-Time on GPU:** The model is optimized to deliver competitive inference speeds when accelerated on high-end GPUs using tools like [NVIDIA TensorRT](https://docs.ultralytics.com/integrations/tensorrt/).

### Weaknesses

- **Computational Cost:** RTDETRv2 generally has a higher parameter count and more FLOPs than comparable YOLOv8 models, demanding more significant computational resources, especially GPU memory.
- **Training Complexity:** Training transformer-based models is notoriously resource-intensive and can be significantly slower and require more memory than training CNN-based models like YOLOv8.
- **Inference Speed:** While fast on powerful GPUs, its performance can degrade significantly on CPUs or less powerful edge devices, making it less suitable for a broad range of hardware.
- **Limited Versatility:** RTDETRv2 is primarily designed for object detection and lacks the native multi-task support for segmentation, classification, and pose estimation found in YOLOv8.
- **Ecosystem:** It does not benefit from a unified, user-friendly ecosystem like Ultralytics, which can make training, deployment, and maintenance more complex for developers.

### Ideal Use Cases

RTDETRv2 is best suited for:

- **High-Accuracy Scenarios:** Applications where achieving the highest possible mAP on complex datasets is the primary goal, and ample GPU resources are available.
- **Academic Research:** Exploring the capabilities of transformer-based architectures for object detection.
- **Cloud-Based Deployment:** Systems where inference is performed on powerful cloud servers with dedicated GPU acceleration.

[Learn more about RTDETRv2](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme){ .md-button }

## Performance Analysis: Speed, Accuracy, and Efficiency

When comparing YOLOv8 and RTDETRv2, it's clear that each model has its own strengths. The table below shows that while the largest RTDETRv2 model slightly edges out YOLOv8x in mAP, YOLOv8 models consistently offer a better balance of speed, accuracy, and efficiency.

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv8n    | 640                   | 37.3                 | **80.4**                       | **1.47**                            | **3.2**            | **8.7**           |
| YOLOv8s    | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m    | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l    | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x    | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |
|            |                       |                      |                                |                                     |                    |                   |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | **54.3**             | -                              | 15.03                               | 76                 | 259               |

YOLOv8 demonstrates superior speed across all model sizes, especially on CPU, where official benchmarks for RTDETRv2 are not provided. For example, YOLOv8l achieves 52.9 mAP with a latency of just 9.06 ms on a T4 GPU, while the slightly more accurate RTDETRv2-l (53.4 mAP) is slower at 9.76 ms. This efficiency makes YOLOv8 a more practical choice for applications requiring [real-time inference](https://www.ultralytics.com/glossary/real-time-inference).

## Conclusion: Which Model Should You Choose?

RTDETRv2 is an impressive model that showcases the potential of transformers for high-accuracy object detection, making it a strong choice for research and specialized applications with abundant computational resources.

However, for the vast majority of developers, researchers, and businesses, **[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) is the superior choice**. It offers an exceptional balance of speed and accuracy, is far more computationally efficient, and is significantly easier to use. Its versatility across multiple computer vision tasks, combined with a robust and well-maintained ecosystem, makes it a more practical, cost-effective, and powerful solution for building and deploying real-world AI systems. For those looking for the latest advancements, newer models like [YOLO11](https://docs.ultralytics.com/models/yolo11/) push these advantages even further.

## Explore Other Models

For further exploration, consider these comparisons involving YOLOv8, RTDETRv2, and other relevant models:

- [RT-DETR vs YOLOv7](https://docs.ultralytics.com/compare/rtdetr-vs-yolov7/)
- [YOLOv8 vs YOLOv7](https://docs.ultralytics.com/compare/yolov7-vs-yolov8/)
- [YOLOv8 vs YOLOv9](https://docs.ultralytics.com/compare/yolov8-vs-yolov9/)
- [EfficientDet vs YOLOv8](https://docs.ultralytics.com/compare/efficientdet-vs-yolov8/)
- Explore the latest models like [YOLOv10](https://docs.ultralytics.com/models/yolov10/) and [YOLO11](https://docs.ultralytics.com/models/yolo11/).
