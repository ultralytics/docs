---
title: YOLOv10 vs YOLOv5 Comparison
comments: true
description: Compare YOLOv10 and YOLOv5 for object detection, including NMS-free design, COCO mAP, TensorRT speed, parameter efficiency, ecosystem support, and use cases.
keywords: YOLOv10, YOLOv5, object detection, real-time models, computer vision, NMS-free, model comparison, YOLO, Ultralytics, machine learning
---

# YOLOv10 vs YOLOv5

Choosing the right neural network architecture is critical for deploying successful [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) pipelines in production. This page provides an in-depth technical analysis comparing **YOLOv10** and **YOLOv5**, two highly influential models in the evolution of real-time object detection. While both models have made significant impacts on the AI community, they represent different eras and philosophies in deep learning architecture design.

This guide evaluates these architectures based on [mean Average Precision (mAP)](https://www.ultralytics.com/glossary/mean-average-precision-map), inference latency, parameter efficiency, and ecosystem support, helping you choose the best model for your deployment needs.

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv10", "YOLOv5"]'></canvas>

## Model Overviews

### YOLOv10: Real-Time End-to-End Object Detection

Developed by researchers at Tsinghua University, YOLOv10 introduced a novel approach to object detection by eliminating the need for NMS post-processing.

- **Authors:** Ao Wang, Hui Chen, Lihao Liu, et al.
- **Organization:** [Tsinghua University](https://www.tsinghua.edu.cn/en/)
- **Date:** 2024-05-23
- **Research Paper:** [arXiv:2405.14458](https://arxiv.org/abs/2405.14458)
- **Source Code:** [YOLOv10 GitHub Repository](https://github.com/THU-MIG/yolov10)

The defining feature of YOLOv10 is its **End-to-End NMS-Free Design**. Historically, YOLO models relied on [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) to filter out redundant bounding boxes. YOLOv10 utilizes consistent dual assignments for NMS-free training, which reduces inference latency variability and simplifies deployment logic. Additionally, the architecture follows a holistic efficiency-accuracy driven design that optimizes various components to reduce computational redundancy.

[Learn more about YOLOv10](https://docs.ultralytics.com/models/yolov10){ .md-button }

### YOLOv5: The Industry Standard for Usability

Released by Ultralytics in June 2020, YOLOv5 raised expectations for usability in open-source vision AI frameworks. It remains one of the most widely deployed architectures globally.

- **Author:** Glenn Jocher
- **Organization:** [Ultralytics](https://www.ultralytics.com)
- **Date:** 2020-06-26
- **Source Code:** [YOLOv5 GitHub Repository](https://github.com/ultralytics/yolov5)

YOLOv5 is known for its **Ease of Use** and **Well-Maintained Ecosystem**. Written entirely in PyTorch, it offered out-of-the-box support for training, validation, and export to formats like [ONNX](https://onnx.ai/) and [TensorRT](https://developer.nvidia.com/tensorrt). Unlike YOLOv10, which focuses on object detection, YOLOv5 also offers **Versatility**: the original YOLOv5 repository added [instance segmentation](https://docs.ultralytics.com/tasks/segment) in v7.0 and [image classification](https://docs.ultralytics.com/tasks/classify) in v6.2.

[Learn more about YOLOv5](https://platform.ultralytics.com/ultralytics/yolov5){ .md-button }

## Performance and Metrics Comparison

Visualizing the relationship between speed and accuracy is essential for identifying the models that offer the best accuracy for a given speed constraint. Understanding these [performance metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics) is fundamental to selecting a model that aligns with your specific hardware constraints.

| Model    | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| -------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| YOLOv10n | 640                         | 38.5                       | -                                    | 1.84                                      | 2.3                      | 6.7                     |
| YOLOv10s | 640                         | 46.3                       | -                                    | 2.49                                      | 7.2                      | 21.6                    |
| YOLOv10m | 640                         | 51.1                       | -                                    | 4.74                                      | 15.4                     | 59.1                    |
| YOLOv10b | 640                         | 52.5                       | -                                    | 5.74                                      | 19.1                     | 92.0                    |
| YOLOv10l | 640                         | 53.2                       | -                                    | 7.28                                      | 24.4                     | 120.3                   |
| YOLOv10x | 640                         | **54.4**                   | -                                    | 10.70                                     | 29.5                     | 160.4                   |
|          |                             |                            |                                      |                                           |                          |                         |
| YOLOv5n  | 640                         | 28.0                       | -                                    | **1.12**                                  | **1.9**                  | **4.5**                 |
| YOLOv5s  | 640                         | 37.4                       | -                                    | 1.92                                      | 7.2                      | 16.5                    |
| YOLOv5m  | 640                         | 45.4                       | -                                    | 4.03                                      | 21.2                     | 49.0                    |
| YOLOv5l  | 640                         | 49.0                       | -                                    | 6.61                                      | 46.5                     | 109.1                   |
| YOLOv5x  | 640                         | 50.7                       | -                                    | 11.89                                     | 86.7                     | 205.7                   |

### Technical Analysis

1. **Accuracy (mAP):** YOLOv10 shows a clear generational advantage in accuracy. For instance, the YOLOv10-X model achieves 54.4% mAP<sup>val</sup>, outperforming YOLOv5x (50.7% mAP). This gain comes largely from YOLOv10's anchor-free head and accuracy-driven refinements such as large-kernel convolutions and partial self-attention.
2. **Inference Latency:** While YOLOv5 models are very fast on raw T4 TensorRT benchmarks (e.g., YOLOv5n at 1.12 ms), YOLOv10 can skip the NMS post-processing step entirely. In practical end-to-end deployments, YOLOv10's NMS-free design provides more consistent latency, which is critical for real-time applications like [autonomous vehicles](https://www.ultralytics.com/glossary/autonomous-vehicles) and robotics.
3. **Parameter Efficiency:** YOLOv10 models offer a strong **Performance Balance**. YOLOv10-S achieves 46.3% mAP with 7.2M parameters, the same count as YOLOv5s, which reaches 37.4% mAP.

!!! tip "Deployment Tip"

    When deploying to [edge AI](https://www.ultralytics.com/glossary/edge-ai) devices like the [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing), models without NMS logic (YOLOv10 or YOLO26 exported with `nms=False`) often compile more cleanly to TensorRT, avoiding fallback operations to the CPU.

## Use Cases and Recommendations

Choosing between YOLOv10 and YOLOv5 depends on your specific project requirements, deployment constraints, and ecosystem preferences.

### When to Choose YOLOv10

YOLOv10 is a strong choice for:

- **NMS-Free Real-Time Detection:** Applications that benefit from end-to-end detection without Non-Maximum Suppression, reducing deployment complexity.
- **Balanced Speed-Accuracy Tradeoffs:** Projects requiring a strong balance between inference speed and detection accuracy across various model scales.
- **Consistent-Latency Applications:** Deployment scenarios where predictable inference times are critical, such as [robotics](https://www.ultralytics.com/glossary/robotics) or autonomous systems.

### When to Choose YOLOv5

YOLOv5 is recommended for:

- **Proven Production Systems:** Existing deployments where YOLOv5's long track record of stability, extensive documentation, and large community are valued.
- **Resource-Constrained Training:** Environments with limited GPU resources where YOLOv5's efficient training pipeline and lower memory requirements are advantageous.
- **Extensive Export Format Support:** Projects requiring deployment across many formats including [ONNX](https://docs.ultralytics.com/integrations/onnx), [TensorRT](https://docs.ultralytics.com/integrations/tensorrt), [CoreML](https://docs.ultralytics.com/integrations/coreml), and [LiteRT](https://docs.ultralytics.com/integrations/litert).

### When to Choose Ultralytics (YOLO26)

For most new projects, [Ultralytics YOLO26](https://platform.ultralytics.com/ultralytics/yolo26) offers the best combination of performance and developer experience:

- **NMS-Free Edge Deployment:** Applications requiring consistent, low-latency inference without the complexity of Non-Maximum Suppression post-processing.
- **CPU-Only Environments:** Devices without dedicated GPU acceleration, where YOLO26n runs up to 43% faster on CPU than YOLO11n.
- **Small Object Detection:** Challenging scenarios like [aerial drone imagery](https://docs.ultralytics.com/datasets/detect/visdrone) or IoT sensor analysis where ProgLoss and STAL improve accuracy on tiny objects.

## The Ultralytics Advantage

While YOLOv10 offers excellent detection capabilities, relying on academic repositories can sometimes complicate production pipelines. By using the official [Ultralytics Python package](https://pypi.org/project/ultralytics/), you gain access to a unified ecosystem that supports both YOLOv5 and YOLOv10, along with advanced features.

- **Training Efficiency:** Ultralytics YOLO architectures are optimized for low memory use during training. Unlike heavier transformer models (such as [RT-DETR](https://docs.ultralytics.com/models/rtdetr)), which require much more CUDA memory, YOLOv5 and YOLOv10 train comfortably on standard consumer GPUs.
- **Ecosystem Integration:** [Ultralytics Platform](https://platform.ultralytics.com) lets developers manage datasets visually and train in the cloud, while the Python package integrates with experiment trackers like [Weights & Biases](https://docs.ultralytics.com/integrations/weights-biases) and supports [hyperparameter tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning).

### Code Example: Seamless Training

Using the Ultralytics library, switching between these architectures is as simple as changing the model string. The training pipeline automatically handles data augmentation, scaling, and optimizer configuration.

```python
from ultralytics import YOLO

# To use YOLOv5:
# model = YOLO("yolov5su.pt")

# To use YOLOv10:
model = YOLO("yolov10s.pt")

# Train the model on a custom dataset
results = model.train(
    data="coco8.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # Use GPU 0
)

# Export the trained model to ONNX format
path = model.export(format="onnx")
```

## The Next Generation: Ultralytics YOLO26

If you are starting a new [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) project, evaluate the latest **Ultralytics YOLO26**. Released in January 2026, it combines key innovations from recent YOLO generations.

YOLO26 incorporates the **[End-to-End NMS-Free Design](https://www.ultralytics.com/blog/why-ultralytics-yolo26-removes-nms-and-how-that-changes-deployment)** pioneered by YOLOv10 as an optional one-to-one head (`nms=False`), which simplifies deployment. YOLO26 also introduces several other improvements:

- **Up to 43% Faster CPU Inference:** With the Distribution Focal Loss (DFL) module removed, YOLO26n runs up to 43% faster than YOLO11n on standard CPUs, making YOLO26 well suited to [mobile deployment](https://docs.ultralytics.com/guides/model-deployment-options) and low-power IoT sensors.
- **MuSGD Optimizer:** Inspired by Large Language Model (LLM) training techniques like Moonshot AI's Kimi K2, YOLO26 uses a hybrid of SGD and Muon. This provides more stable training and faster convergence than the standard SGD and AdamW optimizers used in earlier YOLO models.
- **ProgLoss + STAL:** These loss functions improve small-object recognition, which is critical for drone imagery and aerial security applications.
- **Task-Specific Improvements:** While YOLOv10 is strictly a bounding box detector, YOLO26 supports detection, instance segmentation, semantic segmentation, depth estimation, classification, pose estimation, and OBB, with task-specific improvements such as Residual Log-Likelihood Estimation (RLE) for pose and a specialized angle loss for Oriented Bounding Boxes (OBB).

!!! note "Explore Further"

    To compare these architectures against other detector families, see the [YOLO11 vs EfficientDet](https://docs.ultralytics.com/compare/yolo11-vs-efficientdet) and [RT-DETR vs YOLOv8](https://docs.ultralytics.com/compare/rtdetr-vs-yolov8) comparisons.

Whether you rely on the proven YOLOv5, the NMS-free YOLOv10, or the latest YOLO26, the Ultralytics ecosystem provides the tools to bring your vision AI applications to production quickly.
