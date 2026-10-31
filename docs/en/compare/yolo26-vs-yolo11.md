---
title: YOLO26 vs YOLO11 Comparison
comments: true
description: Compare Ultralytics YOLO26 and YOLO11 — COCO mAP, CPU ONNX and T4 TensorRT speed, NMS-free inference, DFL removal, MuSGD, and when to upgrade or stay.
keywords: YOLO26, YOLO11, Ultralytics, object detection, NMS-free, end-to-end detection, CPU inference, edge AI, MuSGD, ProgLoss, small object detection, model comparison, YOLO comparison, ONNX export, real-time detection
---

# YOLO26 vs YOLO11

When building computer vision systems, selecting the right model means balancing accuracy, latency, and resource efficiency. This technical comparison covers the transition from [Ultralytics](https://www.ultralytics.com) **YOLO11** to the newer **YOLO26**, giving AI engineers and researchers the details needed to make informed architectural decisions.

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO26", "YOLO11"]'></canvas>

## Model Lineage and Metadata

Both models were developed by Ultralytics, but they represent different paradigms in the timeline of object detection and multi-task vision models.

**YOLO26 Details:**

- **Authors:** Glenn Jocher and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com/about)
- **Date:** 2026-01-14
- **GitHub:** [Ultralytics Repository](https://github.com/ultralytics/ultralytics)
- **Docs:** [YOLO26 Official Documentation](https://docs.ultralytics.com/models/yolo26)

[Learn more about YOLO26](https://platform.ultralytics.com/ultralytics/yolo26){ .md-button }

**YOLO11 Details:**

- **Authors:** Glenn Jocher and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com/about)
- **Date:** 2024-09-27
- **GitHub:** [Ultralytics Repository](https://github.com/ultralytics/ultralytics)
- **Docs:** [YOLO11 Official Documentation](https://docs.ultralytics.com/models/yolo11)

[Learn more about YOLO11](https://platform.ultralytics.com/ultralytics/yolo11){ .md-button }

!!! tip "Other Architectures"

    While YOLO26 is the most advanced Ultralytics real-time model, users with specialized hardware or large memory budgets might also explore transformer-based architectures like [RT-DETR](https://docs.ultralytics.com/models/rtdetr) or the first NMS-free YOLO, [YOLOv10](https://docs.ultralytics.com/models/yolov10).

## Architectural Differences and Innovations

The move from YOLO11 to YOLO26 changes both the model architecture and the training recipe. While YOLO11 established a strong baseline for [object detection](https://docs.ultralytics.com/tasks/detect) and multi-task learning, YOLO26 redesigns the deployment path for edge computing.

### End-to-End NMS-Free Design

One of the most significant upgrades in YOLO26 is native end-to-end inference. Unlike YOLO11, which relies on [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) post-processing to filter overlapping bounding boxes, YOLO26 can skip this step entirely with its optional one-to-one head (`nms=False`). This concept, first pioneered in YOLOv10 (see the [YOLOv10 vs YOLO26 comparison](https://docs.ultralytics.com/compare/yolov10-vs-yolo26)), reduces latency variability and simplifies deployment logic across diverse edge devices.

### DFL Removal for Edge Efficiency

YOLO11 uses Distribution Focal Loss (DFL) to refine bounding box estimates. However, DFL relies on softmax operations that are often poorly supported by low-power edge accelerators. YOLO26 removes DFL without sacrificing accuracy. This simplification improves compatibility with embedded systems and helps YOLO26n reach up to **43% faster CPU inference** than YOLO11n.

### The MuSGD Optimizer

YOLO26 introduces the **MuSGD Optimizer**, a hybrid of Stochastic Gradient Descent (SGD) and Muon inspired by large language model (LLM) training innovations from Moonshot AI's Kimi K2. It brings language model training stability to computer vision and speeds up convergence.

### ProgLoss and STAL

For researchers working with [aerial imagery](https://docs.ultralytics.com/datasets/detect/visdrone) or drone applications, detecting small objects is a long-standing challenge. YOLO26 introduces ProgLoss combined with STAL (Small-Target-Aware Label Assignment), which improves small-object recognition over YOLO11.

## Performance and Metrics Comparison

YOLO26 reaches higher mAP than YOLO11 at every size, from +1.4 (n) to +2.8 (x), with similar parameter counts and FLOPs. YOLO26n and YOLO26s are also faster on CPU, while the larger YOLO11 models keep a CPU latency edge.

| Model                                                                  | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ---------------------------------------------------------------------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| [YOLO26n](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n) | 640                         | **40.9**                   | **38.9**                             | 1.7                                       | **2.4**                  | **5.5**                 |
| [YOLO26s](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s) | 640                         | **48.6**                   | **87.2**                             | **2.5**                                   | 9.5                      | **20.9**                |
| [YOLO26m](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m) | 640                         | **53.1**                   | 220.0                                | **4.7**                                   | 20.4                     | 68.4                    |
| [YOLO26l](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l) | 640                         | **55.0**                   | 286.2                                | **6.2**                                   | **24.8**                 | **86.8**                |
| [YOLO26x](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x) | 640                         | **57.5**                   | 525.8                                | 11.8                                      | **55.7**                 | **194.4**               |
|                                                                        |                             |                            |                                      |                                           |                          |                         |
| [YOLO11n](https://platform.ultralytics.com/ultralytics/yolo11/yolo11n) | 640                         | 39.5                       | 56.1                                 | **1.5**                                   | 2.6                      | 6.5                     |
| [YOLO11s](https://platform.ultralytics.com/ultralytics/yolo11/yolo11s) | 640                         | 47.0                       | 90.0                                 | **2.5**                                   | **9.4**                  | 21.6                    |
| [YOLO11m](https://platform.ultralytics.com/ultralytics/yolo11/yolo11m) | 640                         | 51.5                       | **183.2**                            | **4.7**                                   | **20.1**                 | **68.1**                |
| [YOLO11l](https://platform.ultralytics.com/ultralytics/yolo11/yolo11l) | 640                         | 53.4                       | **238.6**                            | **6.2**                                   | 25.3                     | 87.2                    |
| [YOLO11x](https://platform.ultralytics.com/ultralytics/yolo11/yolo11x) | 640                         | 54.7                       | **462.8**                            | **11.3**                                  | 56.9                     | 195.3                   |

_Note: YOLO26n has ~31% lower CPU ONNX latency than YOLO11n (38.9 ms vs 56.1 ms), reflecting its edge-first design._

## Versatility Across Computer Vision Tasks

Both models benefit from the actively maintained Ultralytics ecosystem and its unified Python API, and both support detection, segmentation, classification, pose estimation, and OBB. YOLO26 adds semantic segmentation and depth estimation, plus several task-specific improvements:

- **Instance Segmentation:** YOLO26 uses a semantic segmentation loss and a multi-scale proto module that improve mask quality over YOLO11. Learn more about [segmentation workflows](https://docs.ultralytics.com/tasks/segment).
- **Pose Estimation:** By integrating Residual Log-Likelihood Estimation (RLE), YOLO26 improves keypoint accuracy in complex human poses. Discover [pose estimation capabilities](https://docs.ultralytics.com/tasks/pose).
- **Oriented Bounding Boxes (OBB):** A specialized angle loss function resolves boundary discontinuity issues, making YOLO26 more reliable for detecting rotated objects in satellite imagery. Read about [OBB tasks](https://docs.ultralytics.com/tasks/obb).
- **Image Classification:** Both models handle high-speed [classification](https://docs.ultralytics.com/tasks/classify) efficiently, with YOLO26 delivering modest top-1 accuracy gains on ImageNet (71.4% vs 70.0% for the n models).

## Training and Inference Code Example

Training a model or running inference with Ultralytics takes only a few lines of code. Training YOLO models also requires less [CUDA memory](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit) than large transformer networks.

```python
from ultralytics import YOLO

# Load the cutting-edge YOLO26 nano model
model = YOLO("yolo26n.pt")

# Train the model on the COCO8 dataset efficiently
# optimizer="auto" selects MuSGD for longer training runs
train_results = model.train(
    data="coco8.yaml",
    epochs=100,
    imgsz=640,
    device="0",  # Utilize GPU for accelerated training
)

# Perform NMS-free inference directly on an image
results = model("https://ultralytics.com/images/bus.jpg", nms=False)

# Display the clean, instant predictions
results[0].show()
```

## Ideal Use Cases and Deployment Strategies

Choosing between YOLO26 and YOLO11 depends on the constraints of your production environment.

### When to Deploy YOLO26

YOLO26 is the recommended choice for new projects. It is built for:

- **Edge Computing and IoT:** Its CPU performance and DFL removal make it well suited to devices like Raspberry Pi, Coral Edge TPU, and mobile processors.
- **Drone and Aerial Analytics:** ProgLoss + STAL improve detection and tracking of small, fast-moving objects across wide scenes.
- **Latency-Critical Applications:** In autonomous robotics or [manufacturing quality control](https://www.ultralytics.com/solutions/computer-vision-in-manufacturing), the optional NMS-free head (`nms=False`) keeps latency consistent without post-processing spikes.

### When to Retain YOLO11

YOLO11 remains a capable model. You might stick with YOLO11 if:

- **Legacy Pipelines:** Your existing C++ deployment infrastructure is tightly coupled to the specific output format and NMS logic of YOLO11.
- **Academic Baselines:** You are publishing research and need a widely recognized 2024 baseline to benchmark new algorithms against.

## The Power of the Ultralytics Ecosystem

Whether you deploy YOLO11 or YOLO26, Ultralytics models come with a [well-maintained ecosystem](https://github.com/ultralytics/ultralytics), frequent updates, and a large community.

For enterprise teams, the [Ultralytics Platform](https://platform.ultralytics.com) provides an end-to-end solution for [data annotation](https://docs.ultralytics.com/platform/data/annotation), model training, and cloud deployment. The `ultralytics` package covers the rest of the lifecycle, from [hyperparameter tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning) to exporting trained weights to [CoreML](https://docs.ultralytics.com/integrations/coreml) or [TensorRT](https://docs.ultralytics.com/integrations/tensorrt).
