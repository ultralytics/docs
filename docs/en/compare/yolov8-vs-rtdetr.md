---
comments: true
description: Explore the detailed comparison of YOLOv8 and RTDETRv2 models for object detection. Discover their architecture, performance, and best use cases.
keywords: YOLOv8,RTDETRv2,object detection,model comparison,performance metrics,real-time detection,transformer-based models,computer vision,Ultralytics
---

# Model Comparison: YOLOv8 vs RTDETRv2 for Object Detection

When selecting a computer vision model for object detection tasks, understanding the nuances between different architectures is crucial. This page provides a detailed technical comparison between Ultralytics YOLOv8 and RTDETRv2, two state-of-the-art models in the field. We will delve into their architectural differences, performance metrics, ideal use cases, and discuss their respective strengths and weaknesses to guide you in choosing the right model for your project.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv8", "RTDETRv2"]'></canvas>

## Ultralytics YOLOv8: Streamlined Efficiency and Versatility

Ultralytics YOLOv8 is the latest iteration of the YOLO (You Only Look Once) series, renowned for its speed, efficiency, and ease of use in object detection and other vision AI tasks.

- **Authors:** Glenn Jocher, Ayush Chaurasia, and Jing Qiu
- **Organization:** Ultralytics
- **Date:** 2023-01-10
- **GitHub:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Docs:** [https://docs.ultralytics.com/models/yolov8/](https://docs.ultralytics.com/models/yolov8/)

### Architecture and Key Features

YOLOv8 maintains a single-stage detector architecture, prioritizing **speed and efficiency**. It adopts an **anchor-free** approach, simplifying the design and improving generalization. Key features include a flexible backbone and optimized layers for feature extraction. A significant advantage of YOLOv8 is its **versatility**, supporting not only [object detection](https://docs.ultralytics.com/tasks/detect/) but also [instance segmentation](https://docs.ultralytics.com/tasks/segment/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), [classification](https://docs.ultralytics.com/tasks/classify/), and [oriented bounding boxes](https://docs.ultralytics.com/tasks/obb/) within a unified framework.

### Performance Metrics

YOLOv8 offers a scalable range of models, from YOLOv8n (Nano) to YOLOv8x (Extra Large), catering to different computational needs and performance expectations. It achieves an excellent **balance between speed and accuracy**, making it suitable for numerous real-time applications. Detailed performance metrics are available in the [YOLO Performance Metrics guide](https://docs.ultralytics.com/guides/yolo-performance-metrics/).

### Use Cases

Thanks to its speed and versatility, YOLOv8 is applicable across numerous domains. Its efficiency makes it ideal for real-time surveillance in [security alarm systems](https://docs.ultralytics.com/guides/security-alarm-system/) and industrial automation in [manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing). It is also well-suited for mobile and edge deployments on devices like [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/). Explore more applications on the [Ultralytics Solutions](https://www.ultralytics.com/solutions) page.

### Strengths and Weaknesses

**Strengths:**

- **Exceptional Speed:** Optimized for rapid inference, crucial for real-time processing needs.
- **Efficiency:** Operates with high efficiency and **lower memory requirements** during training and inference compared to transformer models, making it suitable for various hardware platforms, including edge devices.
- **Ease of Use:** Ultralytics provides comprehensive [documentation](https://docs.ultralytics.com/models/yolov8/), a user-friendly [Python package](https://docs.ultralytics.com/usage/python/), and readily available pre-trained weights, simplifying implementation, training, and deployment.
- **Versatility:** Supports multiple vision tasks beyond object detection.
- **Well-Maintained Ecosystem:** Benefits from active development, strong community support, frequent updates, extensive resources, and integration with [Ultralytics HUB](https://www.ultralytics.com/hub) for seamless MLOps.
- **Training Efficiency:** Efficient training processes and readily available pre-trained weights accelerate development cycles.

**Weaknesses:**

- **Accuracy Trade-off:** In highly complex scenarios, especially with very small objects, YOLOv8 might experience a slight decrease in accuracy compared to more computationally intensive models like RTDETRv2, although this gap is often minimal.
- **Hyperparameter Tuning:** Optimal performance might require careful [hyperparameter tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning/).

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## RTDETRv2: High Accuracy with Real-Time Transformer

RTDETRv2 (Real-Time Detection Transformer version 2) is an object detection model developed by Baidu, leveraging the power of Vision Transformers (ViT) to achieve high accuracy while aiming for real-time performance.

- **Authors:** Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu
- **Organization:** Baidu
- **Date:** 2023-04-17 (Original RT-DETR), 2024-07-24 (RTDETRv2)
- **Arxiv:** [https://arxiv.org/abs/2304.08069](https://arxiv.org/abs/2304.08069) (RT-DETR), [https://arxiv.org/abs/2407.17140](https://arxiv.org/abs/2407.17140) (RTDETRv2)
- **GitHub:** [https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)
- **Docs:** [https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme)

### Architecture and Key Features

RTDETRv2 employs a **transformer-based architecture**, utilizing [self-attention mechanisms](https://www.ultralytics.com/glossary/self-attention) to capture global context within images. This approach, combined with CNN features, aims for higher accuracy, particularly in complex scenes with intricate object relationships. Like YOLOv8, it is also an **anchor-free detector**.

### Performance Metrics

RTDETRv2 models generally prioritize accuracy. The larger variants (e.g., RTDETRv2-x) achieve high mAP scores but come with increased computational demands. While optimized for real-time speeds, especially with hardware acceleration like [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/), they may not match the raw speed of the fastest YOLOv8 models on all hardware.

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| :--------- | :-------------------- | :------------------- | :----------------------------- | :---------------------------------- | :----------------- | :---------------- |
| YOLOv8n    | 640                   | 37.3                 | 80.4                           | **1.47**                            | **3.2**            | **8.7**           |
| YOLOv8s    | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m    | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l    | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x    | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |
|            |                       |                      |                                |                                     |                    |                   |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | **54.3**             | -                              | 15.03                               | 76                 | 259               |

### Use Cases

RTDETRv2 is well-suited for applications where achieving the highest possible accuracy is paramount, and sufficient computational resources are available. Examples include [autonomous vehicles](https://www.ultralytics.com/solutions/ai-in-automotive), detailed [medical image analysis](https://www.ultralytics.com/glossary/medical-image-analysis), and high-resolution [satellite image analysis](https://www.ultralytics.com/blog/using-computer-vision-to-analyse-satellite-imagery).

### Strengths and Weaknesses

**Strengths:**

- **High Accuracy:** Transformer architecture enables potentially superior object detection accuracy, especially in complex scenarios.
- **Real-time Capability:** Achieves competitive inference speeds, particularly with hardware acceleration.
- **Robust Feature Extraction:** Vision Transformers effectively capture global context.

**Weaknesses:**

- **Computational Cost:** Generally more computationally intensive and requires **significantly more CUDA memory** for training and inference compared to YOLOv8, especially larger variants.
- **Inference Speed:** While real-time capable, inference speed might be slower than the fastest YOLO models, especially on CPU or resource-constrained devices.
- **Complexity:** Transformer architectures can be more complex to understand, train, and optimize.
- **Limited Task Support:** Primarily focused on object detection, lacking the broad multi-task support of YOLOv8.

[Learn more about RTDETRv2](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme){ .md-button }

## Conclusion

Both Ultralytics YOLOv8 and RTDETRv2 are powerful object detection models, but they cater to different priorities.

**Ultralytics YOLOv8** stands out for its exceptional **balance of speed, accuracy, and efficiency**. Its versatility across multiple vision tasks (detection, segmentation, pose, etc.), combined with its **ease of use**, efficient training, lower memory footprint, and the robust Ultralytics ecosystem (including extensive documentation, active community, and [Ultralytics HUB](https://www.ultralytics.com/hub)), makes it an ideal choice for a wide range of applications, particularly those requiring real-time performance, rapid development, and deployment on diverse hardware, including edge devices.

**RTDETRv2** is a strong contender when **maximum accuracy** is the absolute priority, and computational resources (including significant memory) are less constrained. Its transformer architecture excels in capturing complex scene details but comes at the cost of higher resource usage and complexity compared to YOLOv8.

For users exploring other options, Ultralytics offers a variety of models like [YOLOv5](https://docs.ultralytics.com/models/yolov5/), [YOLOv9](https://docs.ultralytics.com/models/yolov9/), [YOLOv10](https://docs.ultralytics.com/models/yolov10/), and the latest [YOLO11](https://docs.ultralytics.com/models/yolo11/). Further comparisons, such as [YOLOv8 vs DAMO-YOLO](https://docs.ultralytics.com/compare/damo-yolo-vs-yolov8/) or [YOLOv9 vs PP-YOLOE](https://docs.ultralytics.com/compare/pp-yoloe-vs-yolov9/), are available in the [Ultralytics documentation](https://docs.ultralytics.com/compare/).
