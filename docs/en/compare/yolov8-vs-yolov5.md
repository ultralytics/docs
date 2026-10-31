---
title: YOLOv8 vs YOLOv5 Comparison
comments: true
description: Compare Ultralytics YOLOv8 and YOLOv5 for object detection, from anchor-free C2f versus anchor-based C3 designs to COCO mAP, speed, model size, and use cases.
keywords: YOLOv8, YOLOv5, object detection, YOLO comparison, computer vision, model comparison, speed, accuracy, Ultralytics, deep learning
---

# YOLOv8 vs YOLOv5

Choosing the right computer vision architecture is a critical step in building robust machine learning pipelines. This technical comparison explores the differences between two of the most popular models in the vision AI ecosystem: **YOLOv8** and **YOLOv5**. Both models were developed by Ultralytics and have significantly shaped real-time [object detection](https://docs.ultralytics.com/tasks/detect), setting benchmarks for speed, accuracy, and ease of use.

Whether you are deploying to edge devices or scaling cloud inference, understanding the architectural shifts, performance metrics, and training methodologies of these models will help you make an informed decision for your [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) projects.

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv8", "YOLOv5"]'></canvas>

## Ultralytics YOLOv8: The Versatile Standard

Released in early 2023, YOLOv8 represented a major architectural shift from its predecessors. It was designed from the ground up to serve as a unified framework capable of handling multiple vision tasks natively, including [instance segmentation](https://docs.ultralytics.com/tasks/segment), image classification, and [pose estimation](https://docs.ultralytics.com/tasks/pose).

- **Authors:** Glenn Jocher, Ayush Chaurasia, and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com)
- **Date:** 2023-01-10
- **GitHub:** [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Docs:** [YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8)

### Architecture and Methodologies

YOLOv8 introduced an **anchor-free** detection head, which simplifies the training process by eliminating the need to manually configure anchor boxes based on dataset distribution. This makes the model more robust when generalizing to custom datasets and reduces the number of box predictions, speeding up [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms).

The architecture features a **C2f module** (Cross-Stage Partial bottleneck with two convolutions), which replaces the C3 module found in YOLOv5. The C2f module improves gradient flow and allows the model to learn richer feature representations without a significant increase in computational cost. Furthermore, YOLOv8 utilizes a **decoupled head** structure, separating classification and regression tasks, which has been shown to improve convergence speed and accuracy.

!!! tip "Memory Efficiency"

    Ultralytics YOLO models, including YOLOv8, are optimized for lower CUDA memory usage during training compared to many transformer-based alternatives like [RT-DETR](https://docs.ultralytics.com/models/rtdetr). This allows developers to use larger batch sizes on standard consumer GPUs like the NVIDIA RTX series.

### Strengths and Weaknesses

**Strengths:**

- Versatility across multiple tasks beyond simple bounding box detection.
- Streamlined Python API via the `ultralytics` package, making training and exporting intuitive.
- Higher [mean Average Precision (mAP)](https://www.ultralytics.com/glossary/mean-average-precision-map) across all size variants compared to YOLOv5.

**Weaknesses:**

- The decoupled head and C2f module increase parameter count and FLOPs for the smaller variants compared to their YOLOv5 counterparts.

[Learn more about YOLOv8](https://platform.ultralytics.com/ultralytics/yolov8){ .md-button }

## Ultralytics YOLOv5: The Agile Pioneer

Introduced in 2020, YOLOv5 brought YOLO to the [PyTorch](https://pytorch.org/) ecosystem, making it far more accessible to developers. It quickly became one of the most widely used models for fast, reliable, and easily deployable object detection.

- **Author:** Glenn Jocher
- **Organization:** [Ultralytics](https://www.ultralytics.com)
- **Date:** 2020-06-26
- **GitHub:** [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- **Docs:** [YOLOv5 Documentation](https://docs.ultralytics.com/models/yolov5)

### Architecture and Methodologies

YOLOv5 relies on an **anchor-based** architecture and uses a modified CSPDarknet53 backbone. Anchor-based approaches need anchors that match the dataset's bounding box distribution (YOLOv5 checks and recomputes them automatically with AutoAnchor), and they work well on well-defined datasets.

YOLOv5 incorporates the **C3 module**, which efficiently extracts features while maintaining a low parameter footprint. Its loss function combines objectness loss with classification and bounding box regression losses to guide the network toward accurate predictions.

### Strengths and Weaknesses

**Strengths:**

- Lightweight, making the Nano (YOLOv5n) and Small (YOLOv5s) variants well suited to resource-constrained [edge AI](https://www.ultralytics.com/glossary/edge-ai) deployments.
- Fast inference on both CPUs and GPUs.
- A well-established ecosystem with extensive community tutorials and third-party integrations.

**Weaknesses:**

- Relies on anchor boxes, whose priors can limit generalization on highly varied or custom datasets.
- Lower overall accuracy (mAP) compared to modern anchor-free architectures like YOLOv8 and YOLO26.

[Learn more about YOLOv5](https://platform.ultralytics.com/ultralytics/yolov5){ .md-button }

## Performance Comparison

When evaluating these models, achieving a favorable trade-off between speed and accuracy is essential. The table below outlines the performance metrics of both architectures evaluated on the [COCO dataset](https://docs.ultralytics.com/datasets/detect/coco). CPU speeds were measured using [ONNX](https://onnx.ai/), while GPU speeds were tested using [TensorRT](https://developer.nvidia.com/tensorrt).

| Model                                                                  | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ---------------------------------------------------------------------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| [YOLOv8n](https://platform.ultralytics.com/ultralytics/yolov8/yolov8n) | 640                         | **37.3**                   | 80.4                                 | 1.47                                      | 3.2                      | 8.7                     |
| [YOLOv8s](https://platform.ultralytics.com/ultralytics/yolov8/yolov8s) | 640                         | **44.9**                   | 128.4                                | 2.66                                      | 11.2                     | 28.6                    |
| [YOLOv8m](https://platform.ultralytics.com/ultralytics/yolov8/yolov8m) | 640                         | **50.2**                   | 234.7                                | 5.86                                      | 25.9                     | 78.9                    |
| [YOLOv8l](https://platform.ultralytics.com/ultralytics/yolov8/yolov8l) | 640                         | **52.9**                   | 375.2                                | 9.06                                      | **43.7**                 | 165.1                   |
| [YOLOv8x](https://platform.ultralytics.com/ultralytics/yolov8/yolov8x) | 640                         | **53.9**                   | 479.1                                | 14.37                                     | **68.2**                 | 257.8                   |
|                                                                        |                             |                            |                                      |                                           |                          |                         |
| YOLOv5n                                                                | 640                         | 28.0                       | -                                    | **1.12**                                  | **1.9**                  | **4.5**                 |
| YOLOv5s                                                                | 640                         | 37.4                       | -                                    | **1.92**                                  | **7.2**                  | **16.5**                |
| YOLOv5m                                                                | 640                         | 45.4                       | -                                    | **4.03**                                  | **21.2**                 | **49.0**                |
| YOLOv5l                                                                | 640                         | 49.0                       | -                                    | **6.61**                                  | 46.5                     | **109.1**               |
| YOLOv5x                                                                | 640                         | 50.7                       | -                                    | **11.89**                                 | 86.7                     | **205.7**               |

While YOLOv5 has lower GPU latency and fewer FLOPs at every size in this table, YOLOv8 delivers substantially higher mAP across the board (for example, 37.3 versus 28.0 for the Nano models), providing a stronger accuracy-speed balance for demanding real-world deployments.

## Ease of Use and The Ultralytics Ecosystem

A defining characteristic of modern Ultralytics models is the well-maintained ecosystem surrounding them. The transition from YOLOv5 to YOLOv8 brought the introduction of the unified `ultralytics` pip package, streamlining the user experience.

Developers can handle [model training](https://docs.ultralytics.com/modes/train), validation, prediction, and export with just a few lines of Python code, bypassing the complex boilerplate scripts historically required in deep learning projects.

```python
from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Train the model on custom data efficiently
results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=0)

# Export the model to ONNX for production deployment
path = model.export(format="onnx")
```

[Ultralytics Platform](https://platform.ultralytics.com) further simplifies dataset management, cloud training, and deployment.

## Ideal Use Cases

**When to choose YOLOv5:**
If you are maintaining legacy systems, running inference on severely constrained CPUs like a [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi), or working on a project where saving every fraction of a megabyte in model size is critical, YOLOv5 remains a reliable workhorse.

**When to choose YOLOv8:**
Between the two, YOLOv8 is the better choice for new projects. Its architecture supports tracking, segmentation, and [oriented bounding boxes (OBB)](https://docs.ultralytics.com/tasks/obb), and it suits modern applications ranging from autonomous robotics to medical image analysis and smart city infrastructure.

!!! info "Looking for the Latest State-of-the-Art?"

    While YOLOv8 is capable, Ultralytics recommends **[YOLO26](https://platform.ultralytics.com/ultralytics/yolo26)** for new projects. Released in January 2026, it introduces several advancements:

    *   **End-to-End NMS-Free Design:** An optional one-to-one head (`nms=False`) eliminates NMS post-processing for faster, simpler deployment, an approach introduced to YOLO models by YOLOv10.
    *   **MuSGD Optimizer:** A hybrid of SGD and Muon that brings LLM training innovations to computer vision, enabling more stable training and faster convergence.
    *   **Up to 43% Faster CPU Inference:** YOLO26n runs up to 43% faster than YOLO11n on CPU, which suits edge computing environments without dedicated GPUs.
    *   **DFL Removal:** Distribution Focal Loss has been removed for simplified export and enhanced edge device compatibility.
    *   **ProgLoss + STAL:** Progressive Loss and Small-Target-Aware Label Assignment drive notable improvements in small-object recognition, which is critical for aerial imagery and IoT.

With the Ultralytics documentation and tools, you can deploy YOLOv8 or adopt YOLO26 for demanding vision applications. For further learning, see the guides on [hyperparameter tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning) and [model deployment practices](https://docs.ultralytics.com/guides/model-deployment-practices).
