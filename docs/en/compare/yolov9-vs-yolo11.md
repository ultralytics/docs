---
title: YOLOv9 vs YOLO11 Comparison
comments: true
description: Compare YOLOv9 and Ultralytics YOLO11 for object detection, from PGI and GELAN to YOLO11's refined heads, plus COCO mAP, CPU and T4 speed, and use cases.
keywords: YOLO11, YOLOv9, object detection, model comparison, benchmarks, Ultralytics, real-time processing, machine learning, computer vision
---

# YOLOv9 vs YOLO11

The rapid evolution of computer vision has continuously expanded what is possible in real-time [object detection](https://docs.ultralytics.com/tasks/detect). When comparing leading architectures, **YOLOv9** and **[Ultralytics YOLO11](https://platform.ultralytics.com/ultralytics/yolo11)** stand out as major advances, each serving distinct technical needs. YOLOv9 introduced new ways to preserve gradient flow during deep network training, while YOLO11 refined the general-purpose Ultralytics framework for efficiency, versatility, and ease of use.

This technical comparison analyzes their architectures, performance metrics, memory requirements, and ideal deployment scenarios to help you select the right model for your next AI project.

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv9", "YOLO11"]'></canvas>

!!! tip "Future-Proof Your Project with YOLO26"

    While YOLOv9 and YOLO11 are capable models, [YOLO26](https://platform.ultralytics.com/ultralytics/yolo26) is the current recommended Ultralytics model. It features an end-to-end NMS-free design for simplified deployment, up to 43% faster CPU inference (YOLO26n versus YOLO11n), and the MuSGD optimizer for faster convergence, making it the best starting point for new production projects.

## Technical Specifications and Authorship

Understanding the lineage of these models provides essential context for their architectural decisions and framework dependencies.

### YOLOv9

YOLOv9 brought a strong academic focus on deep learning information bottlenecks, prioritizing feature fidelity through custom network blocks.

- **Authors:** Chien-Yao Wang and Hong-Yuan Mark Liao
- **Organization:** [Institute of Information Science, Academia Sinica](https://www.iis.sinica.edu.tw/en/index.html)
- **Date:** February 21, 2024
- **Arxiv:** [https://arxiv.org/abs/2402.13616](https://arxiv.org/abs/2402.13616)
- **GitHub:** [https://github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9){ .md-button }

### Ultralytics YOLO11

YOLO11 was designed from the ground up for production environments, focusing on a balance of high accuracy, real-world deployment speeds, and multi-task versatility.

- **Authors:** Glenn Jocher and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com)
- **Date:** September 27, 2024
- **GitHub:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

[Learn more about YOLO11](https://platform.ultralytics.com/ultralytics/yolo11){ .md-button }

## Architectural Innovations

### Programmable Gradient Information in YOLOv9

YOLOv9 introduces the concept of Programmable Gradient Information (PGI) alongside the Generalized Efficient Layer Aggregation Network (GELAN). As neural networks get deeper, they often suffer from information bottlenecks, where critical details are lost during the feed-forward process. PGI addresses this by providing reliable gradient updates that retain fine-grained spatial information, while GELAN maximizes parameter efficiency. This makes YOLOv9 particularly adept at tasks requiring high feature fidelity, though it relies on standard [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) during post-processing, which can introduce latency on edge devices.

### Streamlined Efficiency in YOLO11

YOLO11 builds on years of YOLO research to deliver an efficient architecture. It improves on [previous iterations](https://www.ultralytics.com/blog/comparing-ultralytics-yolo11-vs-previous-yolo-models) by reducing computational overhead while improving feature extraction. YOLO11 still relies on standard NMS post-processing, but its refined detection heads achieve a strong balance between latency and accuracy. YOLO11 also uses less memory during [model training](https://docs.ultralytics.com/modes/train) and inference than heavy [transformer](https://www.ultralytics.com/glossary/transformer) models, which are often slower to train and require large amounts of CUDA memory.

## Performance Metrics Comparison

When comparing these models on the standard [COCO dataset](https://cocodataset.org/), both perform strongly, but trade-offs emerge between raw parameter count and operational speed.

The table below compares key [YOLO performance metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics).

| Model                                                                  | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ---------------------------------------------------------------------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| YOLOv9t                                                                | 640                         | 38.3                       | -                                    | 2.3                                       | **2.0**                  | 7.7                     |
| YOLOv9s                                                                | 640                         | 46.8                       | -                                    | 3.54                                      | 7.2                      | 26.7                    |
| YOLOv9m                                                                | 640                         | 51.4                       | -                                    | 6.43                                      | 20.1                     | 76.8                    |
| YOLOv9c                                                                | 640                         | 53.0                       | -                                    | 7.16                                      | 25.5                     | 102.8                   |
| YOLOv9e                                                                | 640                         | **55.6**                   | -                                    | 16.77                                     | 58.1                     | 192.5                   |
|                                                                        |                             |                            |                                      |                                           |                          |                         |
| [YOLO11n](https://platform.ultralytics.com/ultralytics/yolo11/yolo11n) | 640                         | 39.5                       | **56.1**                             | **1.5**                                   | 2.6                      | **6.5**                 |
| [YOLO11s](https://platform.ultralytics.com/ultralytics/yolo11/yolo11s) | 640                         | 47.0                       | 90.0                                 | 2.5                                       | 9.4                      | 21.6                    |
| [YOLO11m](https://platform.ultralytics.com/ultralytics/yolo11/yolo11m) | 640                         | 51.5                       | 183.2                                | 4.7                                       | 20.1                     | 68.1                    |
| [YOLO11l](https://platform.ultralytics.com/ultralytics/yolo11/yolo11l) | 640                         | 53.4                       | 238.6                                | 6.2                                       | 25.3                     | 87.2                    |
| [YOLO11x](https://platform.ultralytics.com/ultralytics/yolo11/yolo11x) | 640                         | 54.7                       | 462.8                                | 11.3                                      | 56.9                     | 195.3                   |

### Analysis of the Results

1. **Speed and Hardware Efficiency:** YOLO11 is faster than YOLOv9 at every size in this table. For example, YOLO11n runs in 1.5 ms on an [NVIDIA T4](https://www.nvidia.com/en-us/data-center/tesla-t4/) GPU using [TensorRT](https://developer.nvidia.com/tensorrt), making it well suited to strict real-time pipelines.
2. **Compute Requirements:** YOLO11 models generally require fewer FLOPs (e.g., 68.1B for YOLO11m vs 76.8B for YOLOv9m), translating to lower power draw on battery-operated edge devices like a [Raspberry Pi](https://www.raspberrypi.org/) or mobile hardware.
3. **Accuracy Parity:** While YOLOv9e edges out YOLO11x slightly in absolute mAP (55.6 vs 54.7), YOLO11 reaches its peak accuracy with substantially less latency (11.3ms vs 16.77ms), showing a more favorable performance balance for real-world deployments.

## Ecosystem and Ease of Use

While raw metrics are important, the framework ecosystem often determines project success, and this is where Ultralytics stands out.

The original YOLOv9 repository is highly specialized, offering a research implementation. However, [Ultralytics Platform](https://platform.ultralytics.com) and its corresponding open-source package offer a streamlined user experience, simple API, and extensive documentation that reduce time-to-market.

### Multi-Task Versatility

YOLOv9 focuses predominantly on bounding box detection. In contrast, YOLO11 is a unified multi-task model natively supporting:

- [Instance Segmentation](https://docs.ultralytics.com/tasks/segment)
- [Image Classification](https://docs.ultralytics.com/tasks/classify)
- [Pose Estimation](https://docs.ultralytics.com/tasks/pose)
- [Oriented Bounding Boxes (OBB)](https://docs.ultralytics.com/tasks/obb)

### Seamless Deployment

Using the Ultralytics ecosystem allows developers to [export models](https://docs.ultralytics.com/modes/export) to an array of formats with a single line of [Python](https://www.python.org/) code. Whether targeting [ONNX](https://onnx.ai/), [OpenVINO](https://docs.openvino.ai/), [LiteRT](https://developers.google.com/edge/litert), or [CoreML](https://developer.apple.com/documentation/coreml/), the move from training to production is straightforward.

```python
from ultralytics import YOLO

# Load a highly efficient YOLO11 model
model = YOLO("yolo11n.pt")

# Train rapidly on a custom dataset with minimal memory footprint
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Export the trained model to OpenVINO for Intel CPU acceleration
model.export(format="openvino")
```

## Ideal Use Cases

### When to Utilize YOLOv9

YOLOv9 is a useful tool for research-centric environments or scenarios prioritizing high feature fidelity where hardware latency is not the primary constraint. Its GELAN architecture can help in medical image analysis, where detecting the smallest pixel variations is crucial.

### Why YOLO11 is the Superior Choice

For developers, engineers, and production teams choosing between these two, **YOLO11 is the recommended option**. It excels in environments demanding high-speed, scalable deployment:

- **Smart Retail Analytics:** Tracking products and customers using standard [Intel processors](https://www.intel.com/).
- **Autonomous Drones:** Where low-FLOP architectures preserve battery life while still delivering robust small-object detection.
- **Dynamic Projects:** Workflows that might start as detection but evolve to require pose estimation or segmentation later on.

## Looking Ahead: The Next Evolution

YOLO11 was state-of-the-art for its generation, but the computer vision landscape continues to advance. For new projects, consider **YOLO26**, the current recommended Ultralytics model.

Building on the end-to-end NMS-free design first explored in [YOLOv10](https://docs.ultralytics.com/models/yolov10), YOLO26 introduces the MuSGD optimizer (a hybrid of SGD and Muon) for more stable training. With Distribution Focal Loss (DFL) removed to simplify export, and ProgLoss and STAL (Progressive Loss and Small-Target-Aware Label Assignment) improving small-object detection, YOLO26n achieves up to 43% faster CPU inference than YOLO11n. Teams upgrading from older models like [Ultralytics YOLOv8](https://platform.ultralytics.com/ultralytics/yolov8) can move to YOLO26 or YOLO11 with minimal code changes thanks to the unified Ultralytics API.
