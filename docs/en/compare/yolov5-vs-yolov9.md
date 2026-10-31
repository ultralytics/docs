---
title: YOLOv5 vs YOLOv9 Comparison
comments: true
description: Compare YOLOv5 and YOLOv9 for real-time object detection, covering PGI and GELAN vs CSPNet, COCO benchmarks, training workflows, and use cases.
keywords: YOLOv5, YOLOv9, object detection, model comparison, performance metrics, real-time detection, computer vision, Ultralytics, machine learning
---

# YOLOv5 vs YOLOv9

The landscape of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) and real-time object detection has advanced considerably over the past few years. Navigating the choice between established, battle-tested models and newer research architectures is a common challenge for machine learning engineers. This guide provides a comprehensive technical comparison between two highly influential models in the YOLO family: **YOLOv5** and **YOLOv9**.

Whether you are deploying to constrained edge devices, researching high-fidelity feature extraction, or building complex [object detection](https://docs.ultralytics.com/tasks/detect) pipelines, understanding the architectural differences, performance metrics, and ecosystem support of these models is important.

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv5", "YOLOv9"]'></canvas>

## Model Overviews

Before diving into the architectural comparisons, it is helpful to understand the origins and primary objectives of each model.

### Ultralytics YOLOv5

Developed by Glenn Jocher and released by [Ultralytics](https://www.ultralytics.com) on June 26, 2020, YOLOv5 changed how developers interacted with vision models. By fully embracing the [PyTorch](https://pytorch.org/) framework, YOLOv5 traded the complex compilation steps of earlier Darknet-based models for an intuitive, Python-first user experience.

- **Author:** Glenn Jocher
- **Organization:** [Ultralytics](https://www.ultralytics.com)
- **Date:** 2020-06-26
- **GitHub:** [YOLOv5 Repository](https://github.com/ultralytics/yolov5)
- **Docs:** [YOLOv5 Documentation](https://docs.ultralytics.com/models/yolov5)

YOLOv5 is known for its **Ease of Use** and stable performance across diverse hardware environments. It supports not just detection, but also [instance segmentation](https://docs.ultralytics.com/tasks/segment) and [image classification](https://docs.ultralytics.com/tasks/classify).

[Learn more about YOLOv5](https://platform.ultralytics.com/ultralytics/yolov5){ .md-button }

### YOLOv9

Introduced by Chien-Yao Wang and Hong-Yuan Mark Liao from the Institute of Information Science at Academia Sinica, Taiwan, YOLOv9 focuses heavily on architectural theory to mitigate information bottleneck issues in deep neural networks.

- **Authors:** Chien-Yao Wang and Hong-Yuan Mark Liao
- **Organization:** Institute of Information Science, Academia Sinica, Taiwan
- **Date:** 2024-02-21
- **arXiv:** [2402.13616](https://arxiv.org/abs/2402.13616)
- **GitHub:** [YOLOv9 Repository](https://github.com/WongKinYiu/yolov9)
- **Docs:** [YOLOv9 Documentation](https://docs.ultralytics.com/models/yolov9)

The core of YOLOv9 relies on two major theoretical innovations: Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN). These concepts help the model retain critical spatial features through deep network layers.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9){ .md-button }

!!! tip "Future-Proof Your Deployments"

    While YOLOv5 and YOLOv9 are powerful, the newer [YOLO26](https://platform.ultralytics.com/ultralytics/yolo26) offers a strong balance of speed and accuracy. Featuring optional end-to-end NMS-free inference and up to 43% faster CPU inference (YOLO26n vs YOLO11n), YOLO26 is recommended for modern edge computing and production deployments.

## Architectural and Technical Differences

Understanding what powers these vision models under the hood is vital for optimizing [model deployment](https://docs.ultralytics.com/guides/model-deployment-options) strategies.

### Feature Extraction and Information Retention

YOLOv5 utilizes a Cross Stage Partial Network (CSPNet) backbone, which effectively reduces computation overhead while maintaining accurate gradient flow during backpropagation. This design is highly optimized for traditional [GPU operations](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit) and ensures lower memory requirements during training compared to heavy transformer alternatives.

YOLOv9 introduces GELAN, a generic architecture that extends CSPNet principles. Coupled with PGI—an auxiliary reversible branch—YOLOv9 helps deep layers retain the semantic information needed to compute reliable gradients. This allows YOLOv9 to achieve high [accuracy](https://www.ultralytics.com/glossary/accuracy), particularly on smaller objects.

### Memory Requirements and Training Efficiency

When it comes to training efficiency, YOLOv5 remains strong. The well-maintained [Ultralytics ecosystem](https://docs.ultralytics.com) ensures that YOLOv5 models consume significantly less CUDA memory, allowing researchers to maximize [batch sizes](https://www.ultralytics.com/glossary/batch-size) on consumer-grade GPUs. While YOLOv9 achieves excellent parameter efficiency (high accuracy relative to its size), its training process can be more resource-intensive if not utilizing optimized frameworks. Fortunately, integrating YOLOv9 into the Ultralytics API brings it closer to parity with YOLOv5's streamlined resource management.

## Performance and Metrics

The table below compares their performance on the COCO dataset, including mAP (Mean Average Precision), inference speed, and parameter counts.

| Model   | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| YOLOv5n | 640                         | 28.0                       | -                                    | **1.12**                                  | **1.9**                  | **4.5**                 |
| YOLOv5s | 640                         | 37.4                       | -                                    | 1.92                                      | 7.2                      | 16.5                    |
| YOLOv5m | 640                         | 45.4                       | -                                    | 4.03                                      | 21.2                     | 49.0                    |
| YOLOv5l | 640                         | 49.0                       | -                                    | 6.61                                      | 46.5                     | 109.1                   |
| YOLOv5x | 640                         | 50.7                       | -                                    | 11.89                                     | 86.7                     | 205.7                   |
|         |                             |                            |                                      |                                           |                          |                         |
| YOLOv9t | 640                         | 38.3                       | -                                    | 2.3                                       | 2.0                      | 7.7                     |
| YOLOv9s | 640                         | 46.8                       | -                                    | 3.54                                      | 7.2                      | 26.7                    |
| YOLOv9m | 640                         | 51.4                       | -                                    | 6.43                                      | 20.1                     | 76.8                    |
| YOLOv9c | 640                         | 53.0                       | -                                    | 7.16                                      | 25.5                     | 102.8                   |
| YOLOv9e | 640                         | **55.6**                   | -                                    | 16.77                                     | 58.1                     | 192.5                   |

As the table shows, YOLOv9 achieves higher raw accuracy at equivalent tiers, reflecting its newer architecture. However, YOLOv5n keeps a low T4 TensorRT latency of 1.12 ms, highlighting its enduring strength for high-speed, localized [edge computing](https://www.ultralytics.com/glossary/edge-computing) applications.

## Training Methodologies and Ease of Use

Much of the practical value of computer vision today lies in the accessibility of the toolchain.

### The Ultralytics Advantage

While original research repositories for models like YOLOv9 are foundational, they often come with complex dependency matrices and boilerplate scripts. The [Ultralytics Python API](https://docs.ultralytics.com/usage/python) abstracts most of this complexity. With the Ultralytics ecosystem, you can train, evaluate, and export both YOLOv5 and YOLOv9 with an identical, unified syntax.

```python
from ultralytics import YOLO

# Load a pre-trained YOLOv5 model for fast deployment
model_v5 = YOLO("yolov5su.pt")  # YOLOv5u: anchor-free YOLOv5 weights for the ultralytics package

# Or leverage a YOLOv9 model for high-fidelity accuracy
model_v9 = YOLO("yolov9c.pt")

# Train on the COCO8 example dataset
results = model_v9.train(data="coco8.yaml", epochs=50, imgsz=640)

# Export the trained model to ONNX
model_v9.export(format="onnx")
```

This single-API approach provides broad **Versatility**, supporting not just detection but also instance segmentation, [pose estimation](https://docs.ultralytics.com/tasks/pose), and [oriented bounding boxes (OBB)](https://docs.ultralytics.com/tasks/obb), depending on the model chosen. Integrations with tools like [Comet](https://docs.ultralytics.com/integrations/comet) and [Weights & Biases](https://docs.ultralytics.com/integrations/weights-biases) are also built into the training loop.

## Ideal Use Cases and Real-World Applications

Choosing between these architectures depends largely on the constraints of your hardware and the precision required by your application domain.

### When to Choose YOLOv5

YOLOv5 is a mature model that suits deployments prioritizing stability, low memory footprints, and broad export compatibility.

- **Mobile Deployments:** Exporting YOLOv5 to [LiteRT](https://docs.ultralytics.com/integrations/litert) or CoreML for on-device inference on older smartphones is straightforward.
- **Legacy Edge Hardware:** For devices like the Raspberry Pi or early generation NVIDIA Jetson Nanos, the straightforward convolutions of YOLOv5 help maintain consistent frame rates for applications like [smart parking management](https://docs.ultralytics.com/guides/parking-management).
- **Rapid Prototyping:** The extensive availability of community tutorials, custom [pretrained weights](https://www.ultralytics.com/glossary/model-weights), and broad dataset compatibility makes it the fastest way to validate a proof-of-concept.

### When to Choose YOLOv9

YOLOv9 is ideal for scenarios where capturing intricate details and minimizing false negatives is critical, even if it requires slightly more compute overhead.

- **Aerial and Satellite Imagery:** The PGI framework helps maintain the fidelity of small objects, making YOLOv9 excellent for drone-based [agricultural monitoring](https://www.ultralytics.com/solutions/computer-vision-in-agriculture).
- **Medical Imaging Diagnostics:** When detecting minute anomalies or lesions in high-resolution scans, the accurate gradient flow of GELAN provides a necessary edge in recall.
- **High-End Retail Analytics:** Tracking overlapping products on dense shelves benefits from YOLOv9's stronger feature retention.

## Expanding Your Horizons

While comparing YOLOv5 and YOLOv9 offers a clear view of how architectures have evolved from 2020 to 2024, the field continues to move quickly. For developers who want the latest performance, the [YOLO26 models](https://platform.ultralytics.com/ultralytics/yolo26) are the recommended choice. By offering a native **End-to-End NMS-Free Design** (`nms=False`) as an alternative to traditional Non-Maximum Suppression and utilizing the advanced **MuSGD Optimizer**, YOLO26 bridges the gap between research-level accuracy and production-level speed. With **DFL Removal** (Distribution Focal Loss removed for simplified export and better edge/low-power device compatibility), YOLO26n achieves up to **43% faster CPU inference** than YOLO11n, making it well suited to edge computing. Additionally, **ProgLoss + STAL** pairs a progressive loss with small-target-aware label assignment for notable improvements in small-object recognition, critical for IoT, robotics, and aerial imagery.

You might also be interested in comparing these architectures against other models, such as [YOLOv5 vs RT-DETR](https://docs.ultralytics.com/compare/yolov5-vs-rtdetr) or [YOLOv5 vs YOLO11](https://docs.ultralytics.com/compare/yolov5-vs-yolo11). The unified Ultralytics framework keeps your development pipeline consistent whichever model you choose.
