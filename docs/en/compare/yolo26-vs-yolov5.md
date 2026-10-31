---
title: YOLO26 vs YOLOv5 Comparison
comments: true
description: Compare Ultralytics YOLO26 and YOLOv5 — NMS-free vs anchor-based design, COCO mAP and speed benchmarks, MuSGD training, and when to migrate legacy systems.
keywords: YOLO26, YOLOv5, object detection, real-time detection, Ultralytics, NMS-free, MuSGD, edge AI, CPU inference, benchmarks, small object detection, pose estimation, OBB, ONNX, TensorRT, model comparison
---

# YOLO26 vs YOLOv5

Choosing the right architecture is critical to the success of any computer vision project. This guide compares two major Ultralytics releases: the widely adopted [YOLOv5](https://platform.ultralytics.com/ultralytics/yolov5) and the current [YOLO26](https://platform.ultralytics.com/ultralytics/yolo26). Both have shaped real-time [object detection](https://www.ultralytics.com/glossary/object-detection), but their underlying designs reflect more than five years of progress in how neural networks process visual data.

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO26", "YOLOv5"]'></canvas>

## Model Overview

The key details of both models are summarized below.

**YOLO26 Details:**

- **Authors:** Glenn Jocher and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com)
- **Date:** 2026-01-14
- **GitHub:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Docs:** [YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26)

[Learn more about YOLO26](https://platform.ultralytics.com/ultralytics/yolo26){ .md-button }

**YOLOv5 Details:**

- **Authors:** Glenn Jocher
- **Organization:** [Ultralytics](https://www.ultralytics.com)
- **Date:** 2020-06-26
- **GitHub:** [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- **Docs:** [YOLOv5 Documentation](https://docs.ultralytics.com/models/yolov5)

[Learn more about YOLOv5](https://platform.ultralytics.com/ultralytics/yolov5){ .md-button }

!!! tip "Exploring Other Options"

    While this guide focuses on YOLO26 and YOLOv5, developers migrating legacy systems might also be interested in comparing [YOLO11](https://docs.ultralytics.com/models/yolo11) or the pioneering NMS-free architecture of [YOLOv10](https://docs.ultralytics.com/models/yolov10). Both offer excellent stepping stones for specific deployment environments.

## Architectural Innovations

The gap of more than five years between YOLOv5 and YOLO26 spans major advances in deep learning research. YOLOv5 helped popularize [PyTorch](https://pytorch.org/)-based YOLO training, offering an optimized, anchor-based detection mechanism that became an industry standard. However, it relied on [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) during post-processing, which could introduce latency bottlenecks on resource-constrained devices.

YOLO26 redesigns the inference pipeline with an **End-to-End NMS-Free Design**. Its optional one-to-one head (`nms=False`) removes the need for NMS post-processing, delivering faster and simpler deployment logic, a concept first pioneered in YOLOv10. YOLO26 also features **DFL Removal** (Distribution Focal Loss), which simplifies the output head. This makes exporting the model to formats like [ONNX](https://onnx.ai/) and [TensorRT](https://developer.nvidia.com/tensorrt) more straightforward and improves compatibility with edge and low-power devices.

During training, YOLO26 uses the **MuSGD Optimizer**, a hybrid of SGD and Muon inspired by [Moonshot AI's Kimi K2](https://www.moonshot.ai). This brings LLM training ideas into computer vision, providing more stable training and faster convergence than the plain SGD optimizer YOLOv5 uses by default.

## Performance and Metrics

When evaluating models, the balance between mean Average Precision (mAP) and inference speed determines real-world viability. YOLO26 is optimized for both high-end GPUs and edge CPUs.

| Model                                                                  | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ---------------------------------------------------------------------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| [YOLO26n](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n) | 640                         | **40.9**                   | 38.9                                 | 1.7                                       | 2.4                      | 5.5                     |
| [YOLO26s](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s) | 640                         | **48.6**                   | 87.2                                 | 2.5                                       | 9.5                      | 20.9                    |
| [YOLO26m](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m) | 640                         | **53.1**                   | 220.0                                | 4.7                                       | **20.4**                 | 68.4                    |
| [YOLO26l](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l) | 640                         | **55.0**                   | 286.2                                | **6.2**                                   | **24.8**                 | **86.8**                |
| [YOLO26x](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x) | 640                         | **57.5**                   | 525.8                                | **11.8**                                  | **55.7**                 | **194.4**               |
|                                                                        |                             |                            |                                      |                                           |                          |                         |
| YOLOv5n                                                                | 640                         | 28.0                       | -                                    | **1.12**                                  | **1.9**                  | **4.5**                 |
| YOLOv5s                                                                | 640                         | 37.4                       | -                                    | **1.92**                                  | **7.2**                  | **16.5**                |
| YOLOv5m                                                                | 640                         | 45.4                       | -                                    | **4.03**                                  | 21.2                     | **49.0**                |
| YOLOv5l                                                                | 640                         | 49.0                       | -                                    | 6.61                                      | 46.5                     | 109.1                   |
| YOLOv5x                                                                | 640                         | 50.7                       | -                                    | 11.89                                     | 86.7                     | 205.7                   |

The benchmarks show a large accuracy gain. For example, `YOLO26n` achieves an mAP of 40.9 compared to `YOLOv5n`'s 28.0, and YOLO26n offers **up to 43% faster CPU inference than YOLO11n**. This makes YOLO26 a better fit for embedded deployments like [Raspberry Pi](https://www.raspberrypi.org/) or mobile devices. YOLOv5 is faster in T4 TensorRT at the n, s, and m scales, but the accuracy trade-off strongly favors YOLO26 (for example, 53.1 vs 45.4 mAP for the m models).

## Training Ecosystem and Ease of Use

Both models benefit from the well-maintained Ultralytics ecosystem, with a simple Python API, extensive documentation, and active community support. YOLO26 adds further training improvements.

Ultralytics models typically need less [CUDA memory](https://developer.nvidia.com/cuda) during training than transformer-heavy alternatives. YOLO26 adds **ProgLoss + STAL** (Progressive Loss and Small-Target-Aware Label Assignment), which improve small-object recognition without adding memory overhead.

```python
from ultralytics import YOLO

# Initialize the cutting-edge YOLO26 Nano model
model = YOLO("yolo26n.pt")

# Train the model (optimizer='auto' selects MuSGD for longer training runs)
results = model.train(data="coco8.yaml", epochs=100, imgsz=640, batch=16, device=0)

# Run fast, NMS-free inference on a test image
predictions = model("https://ultralytics.com/images/bus.jpg", nms=False)
predictions[0].show()
```

This script lets developers iterate quickly on [custom datasets](https://docs.ultralytics.com/datasets), moving from data ingestion to a production-ready model.

!!! note "Deployment Made Easy"

    Using [Ultralytics Platform](https://platform.ultralytics.com), you can export your trained YOLO26 models to formats like [CoreML](https://developer.apple.com/documentation/coreml/) or [LiteRT](https://developers.google.com/edge/litert) without writing a single line of conversion code.

## Versatility and Ideal Use Cases

### When to Use YOLOv5

YOLOv5 remains a reliable workhorse for legacy systems. If you have an existing industrial pipeline tightly coupled to anchor-based outputs, or if you are running inference on older [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing) devices with mature, frozen TensorRT stacks, YOLOv5 provides a stable, well-documented solution.

### When to Use YOLO26

YOLO26 is the recommended choice for new computer vision projects, and it covers more tasks than its predecessor. While YOLOv5 primarily focuses on detection (with segmentation and classification added later), YOLO26 natively supports [Instance Segmentation](https://docs.ultralytics.com/tasks/segment), semantic segmentation, depth estimation, [Image Classification](https://docs.ultralytics.com/tasks/classify), [Pose Estimation](https://docs.ultralytics.com/tasks/pose), and [Oriented Bounding Boxes (OBB)](https://docs.ultralytics.com/tasks/obb).

YOLO26 introduces **Task-Specific Improvements**, such as a specialized semantic segmentation loss, Residual Log-Likelihood Estimation (RLE) for more precise pose keypoints, and a specialized angle loss for OBB that resolves boundary issues.

- **Edge IoT and Robotics:** Optional NMS-free inference and up to 43% faster CPU inference than YOLO11n make YOLO26 well suited to real-time robotic navigation and smart home cameras.
- **Aerial Imagery:** The ProgLoss + STAL enhancements make detecting tiny objects from drones—like vehicles in parking lots or crops in agricultural fields—more reliable.
- **Real-Time Video Analytics:** Whether tracking athletes in sports broadcasts or monitoring traffic flows, YOLO26 balances accuracy and speed for real-time video.

Because both models share the Ultralytics ecosystem, moving from YOLOv5 to YOLO26 is straightforward and gives researchers and developers access to the latest Ultralytics capabilities.
