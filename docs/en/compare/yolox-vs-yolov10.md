---
title: YOLOX vs YOLOv10 Comparison
comments: true
description: Compare Megvii YOLOX and YOLOv10 for object detection, from anchor-free decoupled heads to NMS-free dual assignments, with COCO mAP, T4 speed, and use cases.
keywords: YOLOv10, YOLOX, object detection, Ultralytics, real-time, model comparison, benchmark, computer vision, deep learning, AI
---

# YOLOX vs YOLOv10

The evolution of real-time computer vision models has been marked by significant architectural advances. Two important milestones in this journey are YOLOX and YOLOv10. Released in 2021, YOLOX helped bridge the gap between academic research and industrial application by introducing an effective anchor-free design. Three years later, YOLOv10 eliminated the need for Non-Maximum Suppression (NMS) during post-processing, improving efficiency and speed.

This technical comparison explores the architectures, performance metrics, and ideal use cases for both models to help you choose the right tool for your next object detection project.

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOX", "YOLOv10"]'></canvas>

## Model Origins and Metadata

Understanding the origins of these models provides context for their architectural choices and intended deployment environments.

**YOLOX Details**

- **Authors:** Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun
- **Organization:** [Megvii](https://en.megvii.com/)
- **Date:** 2021-07-18
- **arXiv:** [https://arxiv.org/abs/2107.08430](https://arxiv.org/abs/2107.08430)
- **GitHub:** [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- **Docs:** [https://github.com/Megvii-BaseDetection/YOLOX/tree/main/docs](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/docs)

[Learn more about YOLOX](https://github.com/Megvii-BaseDetection/YOLOX){ .md-button }

**YOLOv10 Details**

- **Authors:** Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, and Guiguang Ding
- **Organization:** [Tsinghua University](https://www.tsinghua.edu.cn/en/)
- **Date:** 2024-05-23
- **arXiv:** [https://arxiv.org/abs/2405.14458](https://arxiv.org/abs/2405.14458)
- **GitHub:** [https://github.com/THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)
- **Docs:** [https://docs.ultralytics.com/models/yolov10](https://docs.ultralytics.com/models/yolov10)

[Learn more about YOLOv10](https://docs.ultralytics.com/models/yolov10){ .md-button }

## Architectural Innovations

The core differences between YOLOX and YOLOv10 lie in how they handle bounding box predictions and post-processing.

### YOLOX: Pioneering Anchor-Free Design

YOLOX moved the YOLO family to an anchor-free architecture. By predicting the center of an object rather than relying on predefined anchor boxes, YOLOX significantly reduced the number of design parameters and heuristic tuning required for custom datasets. It also introduced a decoupled head, separating classification and regression tasks into distinct pathways. This approach reduced the conflict between identifying _what_ an object is and determining _where_ it is, leading to noticeable gains in convergence speed and accuracy.

### YOLOv10: NMS-Free Real-Time Detection

While YOLOX simplified the detection head, it still relied on NMS to filter out redundant bounding box predictions. YOLOv10 tackled this fundamental bottleneck. By utilizing consistent dual assignments during training, YOLOv10 achieves native end-to-end detection. It employs a one-to-many head during training to ensure rich supervisory signals, while utilizing a one-to-one head during inference to output final predictions directly. This holistic efficiency-accuracy driven design eliminates NMS entirely, significantly reducing inference latency on embedded chips.

!!! note "The Impact of Removing NMS"

    Non-Maximum Suppression is often a complex operation to accelerate on Neural Processing Units (NPUs). By removing it, YOLOv10 allows the entire model graph to run on specialized hardware, improving compatibility with optimization frameworks like [OpenVINO](https://docs.ultralytics.com/integrations/openvino) and TensorRT.

## Performance Metrics and Comparison

When evaluating models for production, balancing accuracy with computational overhead is critical. The table below illustrates the trade-offs between various scales of YOLOX and YOLOv10.

| Model     | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| --------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| YOLOXnano | 416                         | 25.8                       | -                                    | -                                         | **0.91**                 | **1.08**                |
| YOLOXtiny | 416                         | 32.8                       | -                                    | -                                         | 5.06                     | 6.45                    |
| YOLOXs    | 640                         | 40.5                       | -                                    | 2.56                                      | 9.0                      | 26.8                    |
| YOLOXm    | 640                         | 46.9                       | -                                    | 5.43                                      | 25.3                     | 73.8                    |
| YOLOXl    | 640                         | 49.7                       | -                                    | 9.04                                      | 54.2                     | 155.6                   |
| YOLOXx    | 640                         | 51.1                       | -                                    | 16.1                                      | 99.1                     | 281.9                   |
|           |                             |                            |                                      |                                           |                          |                         |
| YOLOv10n  | 640                         | 38.5                       | -                                    | **1.84**                                  | 2.3                      | 6.7                     |
| YOLOv10s  | 640                         | 46.3                       | -                                    | 2.49                                      | 7.2                      | 21.6                    |
| YOLOv10m  | 640                         | 51.1                       | -                                    | 4.74                                      | 15.4                     | 59.1                    |
| YOLOv10b  | 640                         | 52.5                       | -                                    | 5.74                                      | 19.1                     | 92.0                    |
| YOLOv10l  | 640                         | 53.2                       | -                                    | 7.28                                      | 24.4                     | 120.3                   |
| YOLOv10x  | 640                         | **54.4**                   | -                                    | 10.70                                     | 29.5                     | 160.4                   |

### Analyzing the Data

The metrics show YOLOv10's generational improvement. For instance, YOLOv10-S achieves a [mean Average Precision](https://docs.ultralytics.com/guides/yolo-performance-metrics) of 46.3% compared to YOLOX-m's 46.9%, but does so using less than a third of the parameters (7.2M vs 25.3M) and significantly fewer FLOPs. Furthermore, the top-tier YOLOv10-X model pushes the mAP to 54.4%, making it highly competitive for demanding accuracy tasks while remaining faster than the older YOLOX-x architecture.

## The Ultralytics Ecosystem Advantage

While YOLOX remains a robust open-source research implementation, adopting YOLOv10 provides immediate access to the well-maintained ecosystem provided by Ultralytics. Choosing an Ultralytics-supported model ensures a streamlined user experience characterized by a simple API and extensive documentation.

Developers benefit from the framework's low memory requirements; training Ultralytics models typically consumes much less CUDA memory than heavy transformer-based alternatives like [RT-DETR](https://docs.ultralytics.com/models/rtdetr). This efficient training footprint allows for larger batch sizes on consumer-grade hardware, accelerating the time from data collection to model deployment. The framework also supports multiple tasks, so users can switch between [object detection](https://docs.ultralytics.com/tasks/detect), [instance segmentation](https://docs.ultralytics.com/tasks/segment), and [pose estimation](https://docs.ultralytics.com/tasks/pose) with minimal code changes.

### Training and Inference Example

The unified API makes validating ideas fast. The following snippet trains and deploys a YOLOv10 model using the [PyTorch](https://pytorch.org/) backend:

```python
from ultralytics import YOLO

# Load a pre-trained YOLOv10 nano model
model = YOLO("yolov10n.pt")

# Train the model on the COCO8 dataset
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference on a sample image
predictions = model.predict("https://ultralytics.com/images/bus.jpg")

# Export the model for edge deployment
model.export(format="engine", quantize=16)
```

By leveraging built-in export routines, converting models to formats like [TensorRT](https://docs.ultralytics.com/integrations/tensorrt) or [ONNX](https://docs.ultralytics.com/integrations/onnx) requires just a single line of code, avoiding complex compilation steps.

## Ideal Use Cases and Deployment Scenarios

Choosing between these architectures depends largely on your hardware constraints and specific domain requirements.

### Real-Time Video Analytics

For applications requiring very low latency, such as autonomous driving or real-time traffic monitoring, YOLOv10 is the stronger choice. Its end-to-end NMS-free design provides predictable execution times, which is critical for safety systems where variable post-processing latency cannot be tolerated. The models achieve high frame rates on devices like the NVIDIA Jetson series.

### Academic Baselines and Edge Microcontrollers

YOLOX still holds value in academic settings where researchers want a clean, decoupled-head baseline for experimenting with label assignment strategies. Additionally, the very small YOLOX-Nano (under 1 million parameters) can fit on highly constrained edge devices where memory is tightly limited, provided the hardware can support standard convolution operations.

## The Recommended Model: Ultralytics YOLO26

YOLOv10 was a major step forward in removing NMS, but the field of computer vision advances rapidly. For new projects, Ultralytics recommends [YOLO26](https://platform.ultralytics.com/ultralytics/yolo26), released in January 2026.

YOLO26 builds on the ideas of its predecessors and offers a strong performance balance, natively supporting detection, instance and semantic segmentation, depth estimation, classification, pose estimation, and oriented bounding boxes.

Here is why YOLO26 is the recommended choice for modern computer vision pipelines:

- **End-to-End NMS-Free Design:** Building on YOLOv10, YOLO26 supports native end-to-end inference (`nms=False`), delivering faster, more predictable inference times without NMS post-processing bottlenecks.
- **Up to 43% Faster CPU Inference:** YOLO26n runs up to 43% faster than YOLO11n on CPU, which suits mobile processors and devices without discrete GPUs.
- **MuSGD Optimizer:** Inspired by Large Language Model training (specifically Moonshot AI's Kimi K2), YOLO26 uses a hybrid of SGD and Muon for stable training and fast convergence.
- **ProgLoss + STAL:** Progressive Loss and Small-Target-Aware Label Assignment (STAL) improve small-object detection, which is critical for demanding domains like aerial imagery and drone navigation.
- **DFL Removal:** By removing Distribution Focal Loss, YOLO26 simplifies the model graph for simpler export to edge and low-power devices.
- **Task-Specific Improvements:** Whether you are using Residual Log-Likelihood Estimation (RLE) for pose estimation or a specialized angle loss for OBB, YOLO26 includes improvements for each major vision task.

For developers upgrading their pipelines, [Ultralytics Platform](https://platform.ultralytics.com) with YOLO26 provides efficient training and deployment tools. Users interested in established architectures may also review [YOLO11](https://platform.ultralytics.com/ultralytics/yolo11) or [YOLOv8](https://platform.ultralytics.com/ultralytics/yolov8) for extensive community support and proven robustness.
