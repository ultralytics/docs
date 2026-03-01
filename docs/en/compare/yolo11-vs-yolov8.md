---
comments: true
description: Compare YOLO11 and YOLOv8 architectures, performance, use cases, and benchmarks. Discover which YOLO model fits your object detection needs.
keywords: YOLO11, YOLOv8, object detection, model comparison, performance benchmarks, YOLO series, computer vision, Ultralytics YOLO, YOLO architecture
---

# YOLO11 vs YOLOv8: A Comprehensive Technical Comparison of Real-Time Vision Models

The field of computer vision has witnessed remarkable advancements with the continuous evolution of object detection architectures. When evaluating models for real-world deployment, developers often compare the strengths of [Ultralytics YOLO11](https://platform.ultralytics.com/ultralytics/yolo11) and its highly successful predecessor, [Ultralytics YOLOv8](https://platform.ultralytics.com/ultralytics/yolov8). Both models have set industry standards for speed, accuracy, and developer experience, but they cater to slightly different project lifecycles and performance thresholds.

This guide provides an in-depth analysis of their architectures, training methodologies, and ideal use cases to help you select the best solution for your [artificial intelligence](https://en.wikipedia.org/wiki/Artificial_intelligence) initiatives.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO11", "YOLOv8"]'></canvas>

## Architectural Innovations

The transition from YOLOv8 to YOLO11 introduced several key architectural refinements aimed at maximizing feature extraction efficiency while minimizing computational overhead.

### YOLO11 Architecture

YOLO11 represents a significant leap forward in optimizing parameter usage. It replaces the traditional C2f modules with advanced C3k2 blocks, which enhance spatial feature processing without ballooning the parameter count. Additionally, YOLO11 introduces the C2PSA (Cross-Stage Partial Spatial Attention) module within its backbone. This attention mechanism allows the model to focus on critical regions of interest, drastically improving [small object detection](https://www.ultralytics.com/blog/exploring-small-object-detection-with-ultralytics-yolo11) and handling complex occlusions.

- **Authors:** Glenn Jocher and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com/)
- **Date:** 2024-09-27
- **GitHub:** [Ultralytics Repository](https://github.com/ultralytics/ultralytics)
- **Documentation:** [YOLO11 Docs](https://docs.ultralytics.com/models/yolo11/)

[Learn more about YOLO11](https://platform.ultralytics.com/ultralytics/yolo11){ .md-button }

### YOLOv8 Architecture

Launched a year earlier, YOLOv8 pioneered the transition to an anchor-free detection head, which eliminated the need to manually tune anchor boxes and simplified the loss formulation. Its architecture relies heavily on the C2f block, a design that successfully balanced network depth and gradient flow, making it incredibly robust across a wide range of [computer vision applications](https://viso.ai/applications/computer-vision-applications/).

- **Authors:** Glenn Jocher, Ayush Chaurasia, and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com/)
- **Date:** 2023-01-10
- **GitHub:** [Ultralytics Repository](https://github.com/ultralytics/ultralytics)
- **Documentation:** [YOLOv8 Docs](https://docs.ultralytics.com/models/yolov8/)

[Learn more about YOLOv8](https://platform.ultralytics.com/ultralytics/yolov8){ .md-button }

!!! tip "Design Philosophy"

    While YOLOv8 laid the foundation for anchor-free detection in the Ultralytics ecosystem, YOLO11 refined this approach with spatial attention mechanisms, achieving higher accuracy with fewer computational resources.

## Performance and Benchmarks

When deploying models to edge devices like the [Raspberry Pi](https://www.raspberrypi.org/) or high-performance servers running [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt), understanding the trade-off between speed and accuracy is paramount. The table below illustrates how YOLO11 consistently outperforms YOLOv8 across all size variants.

| Model   | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| YOLO11n | 640                         | **39.5**                   | **56.1**                             | 1.5                                       | **2.6**                  | **6.5**                 |
| YOLO11s | 640                         | **47.0**                   | **90.0**                             | **2.5**                                   | **9.4**                  | **21.5**                |
| YOLO11m | 640                         | **51.5**                   | **183.2**                            | **4.7**                                   | **20.1**                 | **68.0**                |
| YOLO11l | 640                         | **53.4**                   | **238.6**                            | **6.2**                                   | **25.3**                 | **86.9**                |
| YOLO11x | 640                         | **54.7**                   | **462.8**                            | **11.3**                                  | **56.9**                 | **194.9**               |
|         |                             |                            |                                      |                                           |                          |                         |
| YOLOv8n | 640                         | 37.3                       | 80.4                                 | **1.47**                                  | 3.2                      | 8.7                     |
| YOLOv8s | 640                         | 44.9                       | 128.4                                | 2.66                                      | 11.2                     | 28.6                    |
| YOLOv8m | 640                         | 50.2                       | 234.7                                | 5.86                                      | 25.9                     | 78.9                    |
| YOLOv8l | 640                         | 52.9                       | 375.2                                | 9.06                                      | 43.7                     | 165.2                   |
| YOLOv8x | 640                         | 53.9                       | 479.1                                | 14.37                                     | 68.2                     | 257.8                   |

### Analyzing the Metrics

YOLO11 achieves a notably higher Mean Average Precision (mAP) while simultaneously reducing both parameter count and Floating Point Operations (FLOPs). For instance, the YOLO11m model requires 22% fewer parameters than YOLOv8m but delivers a 1.3% higher mAP on the [COCO dataset](https://cocodataset.org/). Furthermore, CPU inference speeds when exported to [ONNX format](https://onnx.ai/) show that YOLO11 is substantially faster, making it an excellent candidate for deployments lacking dedicated [GPU acceleration](https://en.wikipedia.org/wiki/Hardware_acceleration).

## The Ultralytics Ecosystem Advantage

Regardless of whether you choose YOLO11 or YOLOv8, both models benefit from the comprehensive Ultralytics ecosystem, which dramatically simplifies the machine learning lifecycle.

### Ease of Use and Simple API

The `ultralytics` Python package provides a streamlined API that allows engineers and researchers to train, validate, and export models with just a few lines of code. This abstracts away the typical complexities associated with setting up deep learning environments in [PyTorch](https://pytorch.org/).

### Training Efficiency and Memory Requirements

Unlike heavy [Vision Transformers](https://arxiv.org/abs/2010.11929) (like [RT-DETR](https://docs.ultralytics.com/models/rtdetr/)), Ultralytics YOLO models are renowned for their low memory usage during training. This memory efficiency enables developers to train state-of-the-art networks on consumer-grade GPUs or cloud environments like [Google Colab](https://colab.research.google.com/) without facing out-of-memory errors.

### Versatility Across Vision Tasks

Both YOLO11 and YOLOv8 are true multi-task learners. Beyond standard bounding box [object detection](https://docs.ultralytics.com/tasks/detect/), they natively support [instance segmentation](https://docs.ultralytics.com/tasks/segment/), [image classification](https://docs.ultralytics.com/tasks/classify/), human [pose estimation](https://docs.ultralytics.com/tasks/pose/), and [Oriented Bounding Boxes (OBB)](https://docs.ultralytics.com/tasks/obb/) for aerial imagery.

## Code Example: Getting Started

Deploying and training an Ultralytics model is incredibly intuitive. The following example demonstrates how to load a pre-trained YOLO11 model, fine-tune it on a custom dataset, and export it for edge deployment using [Apple CoreML](https://developer.apple.com/machine-learning/core-ml/):

```python
from ultralytics import YOLO

# Initialize the YOLO11 small model
model = YOLO("yolo11s.pt")

# Train the model efficiently with optimized memory requirements
results = model.train(data="coco8.yaml", epochs=50, imgsz=640)

# Evaluate the validation performance
metrics = model.val()

# Run real-time inference on a test image
predictions = model("https://ultralytics.com/images/bus.jpg")

# Export to CoreML for fast mobile deployment
export_path = model.export(format="coreml")
```

!!! info "Seamless Upgrades"

    Because the Ultralytics API is standardized, upgrading a legacy pipeline from YOLOv8 to YOLO11 usually requires only changing the weights string from `"yolov8n.pt"` to `"yolo11n.pt"`.

## Looking Forward: The Pinnacle of Edge AI with YOLO26

While YOLO11 represents a mature and highly capable architecture, the rapid pace of AI innovation continues. For developers initiating new projects who require the absolute cutting edge in performance, [Ultralytics YOLO26](https://platform.ultralytics.com/ultralytics/yolo26) (released January 2026) is the ultimate recommendation.

YOLO26 pushes the boundaries of computer vision with several groundbreaking features:

- **End-to-End NMS-Free Design:** Building on concepts explored in [YOLOv10](https://docs.ultralytics.com/models/yolov10/), YOLO26 natively eliminates Non-Maximum Suppression (NMS) post-processing, resulting in lower, more predictable latency across all deployment hardware.
- **Up to 43% Faster CPU Inference:** By completely removing the Distribution Focal Loss (DFL) branch, YOLO26 is specifically optimized for [edge computing devices](https://www.ibm.com/topics) that lack powerful GPUs.
- **MuSGD Optimizer:** Inspired by large language model (LLM) training techniques, YOLO26 utilizes a hybrid MuSGD optimizer, ensuring remarkably stable and rapid training convergence.
- **ProgLoss + STAL:** These advanced loss functions yield notable improvements in tiny and heavily occluded object recognition, essential for autonomous robotics and drone-based analytics.

Whether you rely on the proven reliability of YOLOv8, the optimized architecture of YOLO11, or the next-generation capabilities of YOLO26, the [Ultralytics Platform](https://platform.ultralytics.com) ensures you have the tools necessary to bring your vision AI applications from concept to production seamlessly. Ensure you explore the extensive [integrations](https://docs.ultralytics.com/integrations/) available to connect your models with enterprise workflows and analytics dashboards.
