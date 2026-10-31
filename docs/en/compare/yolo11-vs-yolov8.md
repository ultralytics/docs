---
title: YOLO11 vs YOLOv8 Comparison
comments: true
description: Compare Ultralytics YOLO11 and YOLOv8, covering C3k2 and C2PSA vs C2f architecture, COCO mAP, CPU and TensorRT speed, parameters, and when to upgrade.
keywords: YOLO11, YOLOv8, object detection, model comparison, performance benchmarks, YOLO series, computer vision, Ultralytics YOLO, YOLO architecture
---

# YOLO11 vs YOLOv8

[Object detection](https://www.ultralytics.com/glossary/object-detection) architectures continue to evolve. When evaluating models for real-world deployment, developers often compare [Ultralytics YOLO11](https://platform.ultralytics.com/ultralytics/yolo11) and its widely adopted predecessor, [Ultralytics YOLOv8](https://platform.ultralytics.com/ultralytics/yolov8). Both models are known for speed, accuracy, and developer experience, but they suit slightly different project lifecycles and performance targets.

This guide analyzes their architectures, training methodologies, and ideal use cases to help you select the right model for your [artificial intelligence](https://www.ultralytics.com/glossary/artificial-intelligence-ai) projects.

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO11", "YOLOv8"]'></canvas>

## Architectural Innovations

The transition from YOLOv8 to YOLO11 introduced several architectural refinements that improve feature extraction efficiency while reducing computational overhead.

### YOLO11 Architecture

YOLO11 focuses on parameter efficiency. It replaces the C2f modules with C3k2 blocks, which improve spatial feature processing without increasing the parameter count. Additionally, YOLO11 introduces the C2PSA (Cross-Stage Partial Spatial Attention) module within its backbone. This attention mechanism helps the model focus on critical regions of interest, improving [small object detection](https://www.ultralytics.com/blog/exploring-small-object-detection-with-ultralytics-yolo11) and handling complex occlusions.

- **Authors:** Glenn Jocher and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com)
- **Date:** 2024-09-27
- **GitHub:** [Ultralytics Repository](https://github.com/ultralytics/ultralytics)
- **Documentation:** [YOLO11 Docs](https://docs.ultralytics.com/models/yolo11)

[Learn more about YOLO11](https://platform.ultralytics.com/ultralytics/yolo11){ .md-button }

### YOLOv8 Architecture

Launched a year earlier, YOLOv8 brought an [anchor-free](https://www.ultralytics.com/glossary/anchor-free-detectors) detection head to Ultralytics YOLO, which eliminated the need to manually tune anchor boxes and simplified the loss formulation. Its architecture relies on the C2f block, a design that balances network depth and gradient flow, making it reliable across a wide range of [computer vision applications](https://www.ultralytics.com/blog/60-impactful-computer-vision-applications).

- **Authors:** Glenn Jocher, Ayush Chaurasia, and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com)
- **Date:** 2023-01-10
- **GitHub:** [Ultralytics Repository](https://github.com/ultralytics/ultralytics)
- **Documentation:** [YOLOv8 Docs](https://docs.ultralytics.com/models/yolov8)

[Learn more about YOLOv8](https://platform.ultralytics.com/ultralytics/yolov8){ .md-button }

!!! tip "Design Philosophy"

    While YOLOv8 laid the foundation for anchor-free detection in the Ultralytics ecosystem, YOLO11 refined this approach with spatial attention mechanisms, achieving higher accuracy with fewer computational resources.

## Performance and Benchmarks

When deploying models to edge devices like the [Raspberry Pi](https://www.raspberrypi.org/) or high-performance servers running [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt), understanding the trade-off between speed and accuracy is essential. The table below shows that YOLO11 is more accurate than YOLOv8 at every model size.

| Model                                                                  | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ---------------------------------------------------------------------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| [YOLO11n](https://platform.ultralytics.com/ultralytics/yolo11/yolo11n) | 640                         | **39.5**                   | **56.1**                             | 1.5                                       | **2.6**                  | **6.5**                 |
| [YOLO11s](https://platform.ultralytics.com/ultralytics/yolo11/yolo11s) | 640                         | **47.0**                   | **90.0**                             | **2.5**                                   | **9.4**                  | **21.6**                |
| [YOLO11m](https://platform.ultralytics.com/ultralytics/yolo11/yolo11m) | 640                         | **51.5**                   | **183.2**                            | **4.7**                                   | **20.1**                 | **68.1**                |
| [YOLO11l](https://platform.ultralytics.com/ultralytics/yolo11/yolo11l) | 640                         | **53.4**                   | **238.6**                            | **6.2**                                   | **25.3**                 | **87.2**                |
| [YOLO11x](https://platform.ultralytics.com/ultralytics/yolo11/yolo11x) | 640                         | **54.7**                   | **462.8**                            | **11.3**                                  | **56.9**                 | **195.3**               |
|                                                                        |                             |                            |                                      |                                           |                          |                         |
| [YOLOv8n](https://platform.ultralytics.com/ultralytics/yolov8/yolov8n) | 640                         | 37.3                       | 80.4                                 | **1.47**                                  | 3.2                      | 8.7                     |
| [YOLOv8s](https://platform.ultralytics.com/ultralytics/yolov8/yolov8s) | 640                         | 44.9                       | 128.4                                | 2.66                                      | 11.2                     | 28.6                    |
| [YOLOv8m](https://platform.ultralytics.com/ultralytics/yolov8/yolov8m) | 640                         | 50.2                       | 234.7                                | 5.86                                      | 25.9                     | 78.9                    |
| [YOLOv8l](https://platform.ultralytics.com/ultralytics/yolov8/yolov8l) | 640                         | 52.9                       | 375.2                                | 9.06                                      | 43.7                     | 165.1                   |
| [YOLOv8x](https://platform.ultralytics.com/ultralytics/yolov8/yolov8x) | 640                         | 53.9                       | 479.1                                | 14.37                                     | 68.2                     | 257.8                   |

### Analyzing the Metrics

YOLO11 achieves higher [mean Average Precision (mAP)](https://www.ultralytics.com/glossary/mean-average-precision-map) while reducing both parameter count and Floating Point Operations (FLOPs). For instance, the YOLO11m model requires 22% fewer parameters than YOLOv8m but delivers 1.3 points higher mAP on the [COCO dataset](https://cocodataset.org/). Furthermore, CPU inference speeds when exported to [ONNX format](https://onnx.ai/) show that YOLO11 is faster at every size (56.1 ms vs 80.4 ms for the nano models), making it a good candidate for deployments without dedicated [GPU acceleration](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit).

## The Ultralytics Ecosystem Advantage

Regardless of whether you choose YOLO11 or YOLOv8, both models benefit from the Ultralytics ecosystem, which simplifies the machine learning lifecycle.

### Ease of Use and Simple API

The `ultralytics` Python package provides a streamlined API that allows engineers and researchers to train, validate, and export models with just a few lines of code. This abstracts away the typical complexities associated with setting up deep learning environments in [PyTorch](https://pytorch.org/).

### Training Efficiency and Memory Requirements

Unlike heavy [Vision Transformers](https://arxiv.org/abs/2010.11929) (like [RT-DETR](https://docs.ultralytics.com/models/rtdetr)), Ultralytics YOLO models are known for their low memory usage during training. This memory efficiency lets developers train accurate networks on consumer-grade GPUs or cloud environments like [Google Colab](https://colab.research.google.com/) without facing out-of-memory errors.

### Versatility Across Vision Tasks

Both YOLO11 and YOLOv8 are multi-task models. Beyond standard bounding box [object detection](https://docs.ultralytics.com/tasks/detect), they natively support [instance segmentation](https://docs.ultralytics.com/tasks/segment), [image classification](https://docs.ultralytics.com/tasks/classify), human [pose estimation](https://docs.ultralytics.com/tasks/pose), and [Oriented Bounding Boxes (OBB)](https://docs.ultralytics.com/tasks/obb) for aerial imagery.

## Use Cases and Recommendations

[Choosing between YOLO11 and YOLOv8](https://www.ultralytics.com/blog/comparing-ultralytics-yolo11-vs-previous-yolo-models) depends on your specific project requirements, deployment constraints, and ecosystem preferences.

### When to Choose YOLO11

YOLO11 is a strong choice for:

- **Production Edge Deployment:** Commercial applications on devices like [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi) or [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson) where reliability and active maintenance are paramount.
- **Multi-Task Vision Applications:** Projects requiring detection, segmentation, classification, pose estimation, and OBB within a single unified framework.
- **Rapid Prototyping and Deployment:** Teams that need to move quickly from data collection to production using the streamlined [Ultralytics Python API](https://docs.ultralytics.com/usage/python).

### When to Choose YOLOv8

YOLOv8 is recommended for:

- **Versatile Multi-Task Deployment:** Projects requiring a proven model for detection, segmentation, classification, pose estimation, and OBB within the Ultralytics ecosystem.
- **Established Production Systems:** Existing production environments already built on the YOLOv8 architecture with stable, well-tested deployment pipelines.
- **Broad Community and Ecosystem Support:** Applications benefiting from YOLOv8's extensive tutorials, third-party integrations, and active community resources.

### When to Choose Ultralytics (YOLO26)

For most new projects, [Ultralytics YOLO26](https://platform.ultralytics.com/ultralytics/yolo26) offers the best combination of performance and developer experience:

- **NMS-Free Edge Deployment:** Applications requiring consistent, low-latency inference without the complexity of Non-Maximum Suppression post-processing.
- **CPU-Only Environments:** Devices without dedicated GPU acceleration, where YOLO26's up to 43% faster CPU inference provides a decisive advantage.
- **Small Object Detection:** Challenging scenarios like [aerial drone imagery](https://docs.ultralytics.com/datasets/detect/visdrone) or IoT sensor analysis where ProgLoss and STAL improve accuracy on small objects.

## Code Example: Getting Started

Training and deploying an Ultralytics model takes a few lines of code. The following example shows how to load a pre-trained YOLO11 model, fine-tune it on a custom dataset, and export it for edge deployment using [Apple CoreML](https://developer.apple.com/documentation/coreml/):

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

!!! info "Simple Upgrades"

    Because the Ultralytics API is standardized, upgrading a legacy pipeline from YOLOv8 to YOLO11 usually requires only changing the weights string from `"yolov8n.pt"` to `"yolo11n.pt"`.

## Looking Forward: The Pinnacle of Edge AI with YOLO26

YOLO11 is a mature and capable architecture, but for developers starting new projects, Ultralytics YOLO26 (released January 2026) is the [recommended model](https://www.ultralytics.com/blog/ultralytics-yolo26-vs-yolo11-vs-yolov8-which-one-should-you-use).

YOLO26 adds several new features:

- **End-to-End NMS-Free Design:** Building on concepts explored in [YOLOv10](https://docs.ultralytics.com/models/yolov10), YOLO26 offers an optional one-to-one head (`nms=False`) that skips [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) post-processing, resulting in lower, more predictable latency across deployment hardware.
- **Up to 43% Faster CPU Inference:** With the Distribution Focal Loss (DFL) branch removed, YOLO26n runs up to 43% faster than YOLO11n on CPU, which suits [edge computing devices](https://www.ultralytics.com/glossary/edge-ai) that lack powerful GPUs.
- **MuSGD Optimizer:** Inspired by [large language model (LLM)](https://www.ultralytics.com/glossary/large-language-model-llm) training techniques, YOLO26 uses MuSGD, a hybrid of SGD and Muon, for stable and fast training convergence.
- **ProgLoss + STAL:** Progressive Loss and Small-Target-Aware Label Assignment (STAL) improve recognition of small and occluded objects, essential for autonomous robotics and drone-based analytics.

Whether you rely on the proven reliability of YOLOv8, the optimized architecture of YOLO11, or the next-generation capabilities of YOLO26, [Ultralytics Platform](https://platform.ultralytics.com) provides the tools to take your vision AI applications from concept to production. The Ultralytics [integrations](https://docs.ultralytics.com/integrations) connect your models with enterprise workflows and analytics dashboards.
