---
title: YOLOv8 vs YOLO11 Comparison
comments: true
description: Compare Ultralytics YOLOv8 and YOLO11 for object detection, from C2f versus C3k2 architectures to COCO mAP, CPU and GPU speed, parameter counts, and use cases.
keywords: YOLOv8, YOLO11, object detection, Ultralytics, YOLO comparison, machine learning, computer vision, inference speed, model accuracy
---

# YOLOv8 vs YOLO11

The rapid evolution of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) has been driven by continuous advances in real-time object detection frameworks. For developers and researchers, choosing the right model is critical to balancing accuracy, speed, and resource efficiency. This technical comparison explores the differences between two foundational models from the [Ultralytics](https://www.ultralytics.com) ecosystem: [Ultralytics YOLOv8](https://platform.ultralytics.com/ultralytics/yolov8) and [Ultralytics YOLO11](https://platform.ultralytics.com/ultralytics/yolo11).

Both models share the hallmark features of Ultralytics architectures: **ease of use**, a **well-maintained ecosystem**, and efficient training with low memory requirements. The sections below cover their architectural designs, performance benchmarks, and ideal deployment scenarios.

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv8", "YOLO11"]'></canvas>

## Model Overviews

Before comparing their specific technical merits, it is helpful to establish the origins and core specifications of both models.

### Ultralytics YOLOv8

Released in early 2023, YOLOv8 introduced [anchor-free detection](https://www.ultralytics.com/glossary/anchor-free-detectors) and significant improvements to the loss functions, quickly becoming widely adopted across many vision tasks.

- **Authors:** Glenn Jocher, Ayush Chaurasia, and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com)
- **Date:** 2023-01-10
- **GitHub:** [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

[Learn more about YOLOv8](https://platform.ultralytics.com/ultralytics/yolov8){ .md-button }

### Ultralytics YOLO11

Building on [its predecessors](https://www.ultralytics.com/blog/comparing-ultralytics-yolo11-vs-previous-yolo-models), YOLO11 refined the core architecture to improve the accuracy-latency trade-off, reducing parameter count while increasing accuracy.

- **Authors:** Glenn Jocher and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com)
- **Date:** 2024-09-27
- **GitHub:** [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

[Learn more about YOLO11](https://platform.ultralytics.com/ultralytics/yolo11){ .md-button }

!!! tip "Other Architectures"

    If you are exploring alternative approaches, Ultralytics also supports transformer-based models like [RT-DETR](https://docs.ultralytics.com/models/rtdetr) and zero-shot open-vocabulary detectors like [YOLO-World](https://docs.ultralytics.com/models/yolo-world). However, for optimal latency and memory efficiency, standard YOLO architectures typically remain the preferred choice.

## Architectural and Methodological Differences

The shift from YOLOv8 to YOLO11 represents a careful evolution in neural network design rather than a complete overhaul, ensuring that the **well-maintained ecosystem** around the models remains stable.

### Backbone and Neck Optimizations

YOLOv8 introduced a streamlined CNN backbone that moved away from traditional anchor boxes, treating object detection purely as a center-point prediction problem. This anchor-free approach significantly reduced the complexity of bounding box regression. YOLO11 took this foundation and introduced an optimized [feature pyramid network (FPN)](https://www.ultralytics.com/glossary/feature-pyramid-network-fpn) and modified the C2f blocks into C3k2 modules. This modification allows YOLO11 to extract richer spatial features, which translates to better accuracy on smaller objects typically found in the [COCO dataset](https://cocodataset.org/).

### Memory Requirements and Training Efficiency

One of the most notable advantages of both YOLOv8 and YOLO11 is their **low memory requirements** during training. Unlike heavy vision transformers that can easily exhaust VRAM on consumer hardware, these models are optimized for accessible [PyTorch](https://pytorch.org/) training on standard GPUs. YOLO11 achieves a substantial reduction in total parameters—up to 42% fewer parameters in the large (L) variant compared to YOLOv8—while increasing its [mean Average Precision (mAP)](https://www.ultralytics.com/glossary/mean-average-precision-map). This means faster epochs and a lower carbon footprint for model training.

## Performance Metrics

Objective benchmarks show the performance balance of these models. The table below compares YOLOv8 and YOLO11 across the standard scaling variants (nano to extra-large).

| Model                                                                  | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ---------------------------------------------------------------------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| [YOLOv8n](https://platform.ultralytics.com/ultralytics/yolov8/yolov8n) | 640                         | 37.3                       | 80.4                                 | **1.47**                                  | 3.2                      | 8.7                     |
| [YOLOv8s](https://platform.ultralytics.com/ultralytics/yolov8/yolov8s) | 640                         | 44.9                       | 128.4                                | 2.66                                      | 11.2                     | 28.6                    |
| [YOLOv8m](https://platform.ultralytics.com/ultralytics/yolov8/yolov8m) | 640                         | 50.2                       | 234.7                                | 5.86                                      | 25.9                     | 78.9                    |
| [YOLOv8l](https://platform.ultralytics.com/ultralytics/yolov8/yolov8l) | 640                         | 52.9                       | 375.2                                | 9.06                                      | 43.7                     | 165.1                   |
| [YOLOv8x](https://platform.ultralytics.com/ultralytics/yolov8/yolov8x) | 640                         | 53.9                       | 479.1                                | 14.37                                     | 68.2                     | 257.8                   |
|                                                                        |                             |                            |                                      |                                           |                          |                         |
| [YOLO11n](https://platform.ultralytics.com/ultralytics/yolo11/yolo11n) | 640                         | **39.5**                   | **56.1**                             | 1.5                                       | **2.6**                  | **6.5**                 |
| [YOLO11s](https://platform.ultralytics.com/ultralytics/yolo11/yolo11s) | 640                         | **47.0**                   | **90.0**                             | **2.5**                                   | **9.4**                  | **21.6**                |
| [YOLO11m](https://platform.ultralytics.com/ultralytics/yolo11/yolo11m) | 640                         | **51.5**                   | **183.2**                            | **4.7**                                   | **20.1**                 | **68.1**                |
| [YOLO11l](https://platform.ultralytics.com/ultralytics/yolo11/yolo11l) | 640                         | **53.4**                   | **238.6**                            | **6.2**                                   | **25.3**                 | **87.2**                |
| [YOLO11x](https://platform.ultralytics.com/ultralytics/yolo11/yolo11x) | 640                         | **54.7**                   | **462.8**                            | **11.3**                                  | **56.9**                 | **195.3**               |

As shown, YOLO11 consistently outperforms YOLOv8 in accuracy while using fewer parameters and FLOPs. The CPU inference speed, measured using [ONNX Runtime](https://onnxruntime.ai/), highlights YOLO11's efficiency for edge deployments. When exported to [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt), both model families stay under 15 ms per image on a T4 GPU, which suits real-world video stream analysis.

## Ecosystem and Ease of Use

Both models share the unified `ultralytics` Python package, so engineers can switch between YOLOv8 and YOLO11 by changing the weights file. Training, validation, and exporting can be achieved in just a few lines of code.

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 model (you can simply swap to "yolov8n.pt")
model = YOLO("yolo11n.pt")

# Train the model efficiently on a local dataset
results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=0)

# Export the optimized model to ONNX
model.export(format="onnx")
```

[Ultralytics Platform](https://platform.ultralytics.com) extends this workflow with cloud-based training, model monitoring, and deployment without requiring advanced DevOps knowledge.

## Versatility and Real-World Applications

A major strength of the Ultralytics framework is its **versatility**. Both YOLOv8 and YOLO11 support a wide range of computer vision tasks beyond standard object detection:

- **[Instance Segmentation](https://docs.ultralytics.com/tasks/segment):** Accurate pixel-level masks useful for medical imaging and autonomous driving.
- **[Image Classification](https://docs.ultralytics.com/tasks/classify):** Lightweight categorization using backbones trained on [ImageNet](https://www.image-net.org/).
- **[Pose Estimation](https://docs.ultralytics.com/tasks/pose):** Keypoint detection for sports analytics and human-computer interaction.
- **[Oriented Bounding Boxes (OBB)](https://docs.ultralytics.com/tasks/obb):** Critical for identifying rotated objects in satellite imagery.

YOLOv8, having been available longer, has a large body of community tutorials and well-tested enterprise deployments. If you are integrating with legacy pipelines that strictly expect YOLOv8 tensor shapes, it remains a dependable choice. For new projects that prioritize efficiency, such as deployment on embedded edge devices like a Raspberry Pi, YOLO11 is the better option thanks to its higher accuracy with fewer parameters and lower CPU latency.

## Use Cases and Recommendations

Choosing between YOLOv8 and YOLO11 depends on your specific project requirements, deployment constraints, and ecosystem preferences.

### When to Choose YOLOv8

YOLOv8 is a strong choice for:

- **Versatile Multi-Task Deployment:** Projects requiring a proven model for [detection](https://docs.ultralytics.com/tasks/detect), segmentation, classification, and pose estimation within the Ultralytics ecosystem.
- **Established Production Systems:** Existing production environments already built on the YOLOv8 architecture with stable, well-tested deployment pipelines.
- **Broad Community and Ecosystem Support:** Applications benefiting from YOLOv8's extensive tutorials, third-party integrations, and active community resources.

### When to Choose YOLO11

YOLO11 is recommended for:

- **Production Edge Deployment:** Commercial applications on devices like [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi) or [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson) where reliability and active maintenance are essential.
- **Multi-Task Vision Applications:** Projects requiring detection, segmentation, pose estimation, and OBB within a single unified framework.
- **Rapid Prototyping and Deployment:** Teams that need to move quickly from data collection to production using the streamlined [Ultralytics Python API](https://docs.ultralytics.com/usage/python).

### When to Choose Ultralytics (YOLO26)

For most new projects, [Ultralytics YOLO26](https://platform.ultralytics.com/ultralytics/yolo26) offers the best combination of performance and developer experience:

- **NMS-Free Edge Deployment:** Applications requiring consistent, low-latency inference without the complexity of [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) post-processing.
- **CPU-Only Environments:** Devices without dedicated GPU acceleration, where YOLO26's up to 43% faster CPU inference (YOLO26n vs YOLO11n) provides a decisive advantage.
- **Small Object Detection:** Challenging scenarios like [aerial drone imagery](https://docs.ultralytics.com/datasets/detect/visdrone) or IoT sensor analysis where ProgLoss and STAL improve accuracy on tiny objects.

## The Cutting Edge: The YOLO26 Advantage

YOLOv8 and YOLO11 are both strong architectures, but for new projects in 2026, Ultralytics YOLO26 is the recommended next step.

YOLO26 simplifies the deployment pipeline. It features an **End-to-End NMS-Free Design**, an approach pioneered in YOLOv10, which (with `nms=False`) eliminates NMS post-processing. **DFL Removal** (Distribution Focal Loss) simplifies export logic and improves compatibility with low-power edge devices, and YOLO26n delivers up to **43% faster CPU inference** than YOLO11n.

Training stability and convergence speed improve with the **MuSGD Optimizer**, a hybrid of SGD and Muon inspired by LLM training techniques. **ProgLoss + STAL** (Progressive Loss and Small-Target-Aware Label Assignment) improve small-object detection, a long-standing challenge for IoT and robotics. Task-specific improvements include RLE for pose estimation and a multi-scale proto module for segmentation.

[Learn more about YOLO26](https://platform.ultralytics.com/ultralytics/yolo26){ .md-button }

!!! info "Choosing the Right Model"

    Choose **YOLOv8** if you rely on its extensive community resources and existing deployments. Choose **YOLO11** for a refined balance of speed and parameter count. Choose **YOLO26**, the [current recommended model](https://www.ultralytics.com/blog/ultralytics-yolo26-vs-yolo11-vs-yolov8-which-one-should-you-use), for edge-optimized, NMS-free inference.

## Conclusion

Choosing between YOLOv8 and YOLO11 ultimately comes down to your project timeline and hardware constraints. YOLOv8 is battle-tested and stable. YOLO11 refines that architecture, delivering higher mAP with fewer parameters, which makes it attractive for resource-constrained edge applications. Either way, the shared Ultralytics Python API keeps your development workflow consistent and well supported, and YOLO26 is available when you need the best edge performance.
