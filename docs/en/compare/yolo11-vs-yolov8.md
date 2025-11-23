---
comments: true
description: Compare YOLO11 and YOLOv8 architectures, performance, use cases, and benchmarks. Discover which YOLO model fits your object detection needs.
keywords: YOLO11, YOLOv8, object detection, model comparison, performance benchmarks, YOLO series, computer vision, Ultralytics YOLO, YOLO architecture
---

# YOLO11 vs YOLOv8: Architectural Evolution and Performance Analysis

Selecting the optimal computer vision model is a critical decision for developers and researchers aiming to balance accuracy, speed, and resource efficiency. This page provides a comprehensive technical comparison between [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) and [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/), two industry-leading architectures designed for [object detection](https://docs.ultralytics.com/tasks/detect/) and advanced vision tasks. We analyze their architectural innovations, benchmark metrics, and ideal deployment scenarios to help you determine the best fit for your [artificial intelligence](https://www.ultralytics.com/glossary/artificial-intelligence-ai) applications.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO11", "YOLOv8"]'></canvas>

## Ultralytics YOLO11

**Authors:** Glenn Jocher, Jing Qiu  
**Organization:** [Ultralytics](https://www.ultralytics.com/)  
**Date:** 2024-09-27  
**GitHub:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)  
**Docs:** [https://docs.ultralytics.com/models/yolo11/](https://docs.ultralytics.com/models/yolo11/)

YOLO11 represents the latest evolution in the renowned YOLO series, engineering significant improvements in feature extraction and processing efficiency. By refining the backbone and neck architectures, YOLO11 achieves higher [mean Average Precision (mAP)](https://www.ultralytics.com/glossary/mean-average-precision-map) while utilizing fewer parameters than its predecessors. It natively supports a broad spectrum of tasks, including [instance segmentation](https://docs.ultralytics.com/tasks/segment/), [image classification](https://docs.ultralytics.com/tasks/classify/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), and oriented bounding boxes (OBB).

### Architecture and Key Features

The YOLO11 architecture introduces the **C3k2** block, an optimized version of the CSP (Cross Stage Partial) bottleneck, and the **C2PSA** (Cross Stage Partial with Spatial Attention) module. These components enhance the model's ability to capture complex visual patterns and spatial relationships while minimizing computational overhead. This design philosophy ensures that YOLO11 excels in [real-time inference](https://www.ultralytics.com/glossary/real-time-inference) scenarios, particularly on edge devices where computational resources are limited.

### Strengths

- **State-of-the-Art Accuracy:** Delivers superior detection performance across all model scales, consistently outperforming previous iterations on the [COCO dataset](https://docs.ultralytics.com/datasets/detect/coco/).
- **CPU Efficiency:** Optimized architectural choices result in significantly faster inference speeds on CPUs, making it a top choice for serverless or edge deployments.
- **Parameter Efficiency:** Achieves high accuracy with fewer parameters and FLOPs, reducing [model storage](https://docs.ultralytics.com/guides/model-deployment-practices/) requirements.
- **Unified Framework:** Seamlessly handles multiple vision tasks within a single, easy-to-use API.

### Weaknesses

- **Ecosystem Maturity:** As a newer release, the volume of third-party tutorials and community-generated content is rapidly growing but may be less extensive than the established YOLOv8.
- **Resource Intensity for Large Models:** While efficient, the largest variants (e.g., YOLO11x) still demand significant GPU resources for training and high-throughput inference.

### Use Cases

YOLO11 is the premier choice for applications requiring the highest possible accuracy-to-speed ratio:

- **Edge AI:** Deploying high-performance detection on [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/) or Raspberry Pi devices.
- **Real-Time Robotics:** Enabling autonomous navigation and object interaction with minimal latency.
- **Medical Imaging:** Assisting in precise [medical image analysis](https://www.ultralytics.com/glossary/medical-image-analysis) for diagnostics where accuracy is paramount.

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

## Ultralytics YOLOv8

**Authors:** Glenn Jocher, Ayush Chaurasia, Jing Qiu  
**Organization:** [Ultralytics](https://www.ultralytics.com/)  
**Date:** 2023-01-10  
**GitHub:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)  
**Docs:** [https://docs.ultralytics.com/models/yolov8/](https://docs.ultralytics.com/models/yolov8/)

Released in early 2023, YOLOv8 redefined the standard for real-time object detection. It introduced an anchor-free detection head and the **C2f** backbone module, marking a significant departure from anchor-based approaches. YOLOv8 is renowned for its stability, versatility, and the massive ecosystem that has developed around it, making it one of the most widely adopted vision models globally.

### Architecture and Key Features

YOLOv8 utilizes a modification of the CSPDarknet53 backbone, incorporating C2f modules that allow for richer gradient flow. Its anchor-free design simplifies the [non-maximum suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) process and reduces the complexity of hyperparameter tuning related to anchor boxes. The model is highly scalable, offering variants from Nano (n) to Extra Large (x) to suit various computational budgets.

### Strengths

- **Proven Reliability:** extensively tested in production environments worldwide, ensuring high stability.
- **Rich Ecosystem:** supported by thousands of tutorials, integrations, and community projects.
- **Versatility:** Like YOLO11, it supports detection, segmentation, classification, and pose estimation.
- **Strong Baseline:** continues to offer competitive performance that exceeds many non-YOLO architectures.

### Weaknesses

- **Performance Gap:** Generally surpassed by YOLO11 in both accuracy (mAP) and inference speed, particularly on CPU hardware.
- **Higher Computational Cost:** Requires slightly more parameters and FLOPs to achieve comparable accuracy to YOLO11.

### Use Cases

YOLOv8 remains an excellent option for:

- **Legacy Systems:** Projects already integrated with YOLOv8 workflows that require stability over bleeding-edge performance.
- **Educational Tools:** Learning computer vision concepts using a model with vast documentation and community examples.
- **General Purpose Detection:** Reliable performance for standard security and monitoring applications.

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## Performance Head-to-Head

The most significant distinction between these two models lies in their efficiency. YOLO11 achieves a "Pareto improvement" over YOLOv8—offering higher accuracy with lower computational cost.

### Efficiency and Speed Analysis

The architectural optimizations in YOLO11 (C3k2, C2PSA) allow it to process images faster while retaining more fine-grained features. This is most evident in **CPU inference**, where YOLO11 models show substantial speedups. For example, the YOLO11n model is approximately **30% faster** on CPU than YOLOv8n while also achieving a higher mAP.

In terms of [GPU inference](https://docs.ultralytics.com/guides/model-deployment-options/), YOLO11 models also demonstrate lower latency across most sizes, making them highly effective for real-time video processing pipelines.

!!! tip "Memory Efficiency"
Both Ultralytics YOLO11 and YOLOv8 are designed for low memory consumption during training and inference compared to transformer-based models like [RT-DETR](https://docs.ultralytics.com/models/rtdetr/). This makes them far more accessible for developers using consumer-grade hardware or cloud environments with limited CUDA memory.

### Comparative Metrics

The table below illustrates the performance improvements. Note the reduction in parameters and FLOPs for YOLO11 alongside the increase in mAP.

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLO11n | 640                   | **39.5**             | **56.1**                       | 1.5                                 | **2.6**            | **6.5**           |
| YOLO11s | 640                   | **47.0**             | **90.0**                       | **2.5**                             | **9.4**            | **21.5**          |
| YOLO11m | 640                   | **51.5**             | **183.2**                      | **4.7**                             | **20.1**           | **68.0**          |
| YOLO11l | 640                   | **53.4**             | **238.6**                      | **6.2**                             | **25.3**           | **86.9**          |
| YOLO11x | 640                   | **54.7**             | **462.8**                      | **11.3**                            | **56.9**           | **194.9**         |
|         |                       |                      |                                |                                     |                    |                   |
| YOLOv8n | 640                   | 37.3                 | 80.4                           | **1.47**                            | 3.2                | 8.7               |
| YOLOv8s | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |

## The Ultralytics Ecosystem Advantage

Choosing an Ultralytics model means gaining access to a comprehensive ecosystem designed to streamline the entire [MLOps](https://www.ultralytics.com/glossary/machine-learning-operations-mlops) lifecycle.

- **Ease of Use:** Both models share the same [Python API](https://docs.ultralytics.com/usage/python/) and Command Line Interface (CLI). Switching from YOLOv8 to YOLO11 often requires changing only a single character in your code string (e.g., `"yolov8n.pt"` to `"yolo11n.pt"`).
- **Training Efficiency:** Ultralytics models utilize advanced training routines including mosaic augmentation and hyperparameter evolution. Pre-trained weights are readily available, allowing for efficient [transfer learning](https://www.ultralytics.com/glossary/transfer-learning) on custom datasets.
- **Versatility:** Unlike many competitors limited to specific tasks, Ultralytics models offer native support for detection, segmentation, classification, pose, and OBB within a unified package.
- **Deployment:** Export models easily to formats like [ONNX](https://docs.ultralytics.com/integrations/onnx/), [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/), CoreML, and OpenVINO for optimized deployment on diverse hardware.

### Unified Usage Example

The shared API design allows for effortless experimentation. Here is how you can load and run prediction with either model:

```python
from ultralytics import YOLO

# Load YOLO11 or YOLOv8 by simply changing the model name
model = YOLO("yolo11n.pt")  # or "yolov8n.pt"

# Train on a custom dataset
# model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference on an image
results = model("https://ultralytics.com/images/bus.jpg")

# Display results
results[0].show()
```

## Conclusion: Which Model Should You Choose?

For the vast majority of new projects, **YOLO11 is the recommended choice**. Its architectural advancements provide a clear advantage in both accuracy and speed, particularly for [edge computing](https://www.ultralytics.com/glossary/edge-computing) applications where efficiency is critical. The reduced parameter count also implies lighter storage requirements and faster download times for mobile deployments.

**YOLOv8** remains a powerful and relevant tool, especially for teams with existing pipelines deeply integrated with specific YOLOv8 versions or for those who rely on the absolute maturity of its documentation ecosystem. However, migrating to YOLO11 is generally straightforward and yields immediate performance benefits.

Both models are released under the **AGPL-3.0** license, promoting open-source collaboration, with [Enterprise Licenses](https://www.ultralytics.com/license) available for commercial products requiring proprietary capabilities.

## Explore Other Models

While YOLO11 and YOLOv8 are excellent general-purpose detectors, specific requirements might benefit from other architectures in the Ultralytics family:

- **[YOLOv10](https://docs.ultralytics.com/models/yolov10/):** Focuses on NMS-free training for lower latency.
- **[YOLOv9](https://docs.ultralytics.com/models/yolov9/):** Emphasizes programmable gradient information for deep model training.
- **[RT-DETR](https://docs.ultralytics.com/models/rtdetr/):** A transformer-based detector offering high accuracy, though with higher memory and compute requirements.

Explore our full range of [model comparisons](https://docs.ultralytics.com/compare/) to find the perfect fit for your project.
