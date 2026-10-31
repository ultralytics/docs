---
title: YOLOv5 vs YOLO26 Comparison
comments: true
description: Compare Ultralytics YOLOv5 and YOLO26 for object detection, covering NMS-free inference, DFL removal, MuSGD training, COCO benchmarks, and migration tips.
keywords: YOLOv5, YOLO26, object detection, model comparison, computer vision, benchmarks, architecture, real-time detection
---

# YOLOv5 vs YOLO26

The evolution of computer vision has been defined by the continuous push for faster, more accurate, and more accessible models. Comparing [Ultralytics YOLOv5](https://platform.ultralytics.com/ultralytics/yolov5) with [Ultralytics YOLO26](https://platform.ultralytics.com/ultralytics/yolo26) spans that progression, from a widely deployed legacy model to the current recommended Ultralytics model.

This guide provides a comprehensive technical breakdown of both architectures, highlighting their performance metrics, structural differences, and ideal deployment scenarios.

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv5", "YOLO26"]'></canvas>

## Model Overviews

### YOLOv5: The Industry Workhorse

Released in 2020, YOLOv5 made object detection far more accessible. By implementing the architecture natively in the [PyTorch](https://pytorch.org/) framework, it gave developers a simple path from installation to training and deployment.

- **Authors:** Glenn Jocher
- **Organization:** [Ultralytics](https://www.ultralytics.com)
- **Date:** 2020-06-26
- **GitHub:** [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- **Docs:** [YOLOv5 Documentation](https://docs.ultralytics.com/models/yolov5)

YOLOv5 established the foundation for the actively maintained Ultralytics ecosystem. It introduced strong data augmentation techniques, efficient training loops, and highly optimized export paths to edge formats like [CoreML](https://developer.apple.com/documentation/coreml) and [ONNX](https://onnx.ai/). Its ease of use and low memory requirements during training made it a staple for startups and researchers worldwide.

[Learn more about YOLOv5](https://platform.ultralytics.com/ultralytics/yolov5){ .md-button }

### YOLO26: The Next-Gen Vision AI Standard

Released in January 2026, **Ultralytics YOLO26** is the latest generation of Ultralytics YOLO. It integrates lessons learned from intervening generations like [YOLOv8](https://platform.ultralytics.com/ultralytics/yolov8) and [YOLO11](https://platform.ultralytics.com/ultralytics/yolo11), while introducing training techniques inspired by Large Language Model (LLM) training.

- **Authors:** Glenn Jocher and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com)
- **Date:** 2026-01-14
- **GitHub:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Docs:** [YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26)

YOLO26 offers a strong balance of accuracy and speed and was designed specifically for edge computing scenarios.

[Learn more about YOLO26](https://platform.ultralytics.com/ultralytics/yolo26){ .md-button }

!!! tip "Other Ultralytics Models"

    If you are migrating an older codebase, you might also be interested in [comparing YOLOv5 with YOLO11](https://docs.ultralytics.com/compare/yolov5-vs-yolo11), the previous generation model supporting diverse tasks like pose estimation and oriented bounding boxes (OBB).

## Architectural Breakthroughs in YOLO26

While YOLOv5 relies on anchor-based detection heads and standard loss functions, YOLO26 reworks the head, loss functions, and optimizer to remove deployment bottlenecks.

1. **End-to-End NMS-Free Design:** The most significant difference is YOLO26's native end-to-end inference option. Unlike YOLOv5, which requires Non-Maximum Suppression (NMS) to filter redundant bounding boxes, YOLO26's optional one-to-one head (`nms=False`) skips this post-processing step. This keeps inference latency more predictable and simplifies integration into C++ or embedded hardware.
2. **DFL Removal:** YOLO26 removes Distribution Focal Loss (DFL). This architectural choice simplifies model export and improves compatibility with low-power edge devices and accelerators that often struggle with complex operators.
3. **MuSGD Optimizer:** Taking cues from Moonshot AI's Kimi K2, YOLO26 utilizes the **MuSGD Optimizer**, a hybrid of SGD and Muon. This brings the stability and fast convergence seen in LLM training to computer vision.
4. **ProgLoss + STAL:** YOLO26 uses ProgLoss and STAL to improve detection of small and dense objects, a historical challenge for YOLOv5.

## Performance Comparison

When comparing the models on the [COCO dataset](https://cocodataset.org/), YOLO26 shows higher accuracy (mAP) at every scale while using fewer parameters from the medium variant upward.

| Model                                                                  | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ---------------------------------------------------------------------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| YOLOv5n                                                                | 640                         | 28.0                       | -                                    | **1.12**                                  | **1.9**                  | **4.5**                 |
| YOLOv5s                                                                | 640                         | 37.4                       | -                                    | **1.92**                                  | **7.2**                  | **16.5**                |
| YOLOv5m                                                                | 640                         | 45.4                       | -                                    | **4.03**                                  | 21.2                     | **49.0**                |
| YOLOv5l                                                                | 640                         | 49.0                       | -                                    | 6.61                                      | 46.5                     | 109.1                   |
| YOLOv5x                                                                | 640                         | 50.7                       | -                                    | 11.89                                     | 86.7                     | 205.7                   |
|                                                                        |                             |                            |                                      |                                           |                          |                         |
| [YOLO26n](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n) | 640                         | **40.9**                   | 38.9                                 | 1.7                                       | 2.4                      | 5.5                     |
| [YOLO26s](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s) | 640                         | **48.6**                   | 87.2                                 | 2.5                                       | 9.5                      | 20.9                    |
| [YOLO26m](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m) | 640                         | **53.1**                   | 220.0                                | 4.7                                       | **20.4**                 | 68.4                    |
| [YOLO26l](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l) | 640                         | **55.0**                   | 286.2                                | **6.2**                                   | **24.8**                 | **86.8**                |
| [YOLO26x](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x) | 640                         | **57.5**                   | 525.8                                | **11.8**                                  | **55.7**                 | **194.4**               |

_Note: The YOLO26 Nano (YOLO26n) achieves 40.9 mAP compared to YOLOv5n's 28.0 mAP, a 12.9-point gain for only 0.5M additional parameters, while DFL removal and the NMS-free head simplify CPU and edge deployment._

## Versatility and Task Support

YOLOv5 is primarily known for [object detection](https://docs.ultralytics.com/tasks/detect). While later updates introduced basic segmentation, YOLO26 was built from the ground up to be a unified multi-task engine.

YOLO26 inherently supports:

- **Instance Segmentation:** Featuring task-specific multi-scale protos and semantic segmentation loss.
- **[Semantic Segmentation](https://docs.ultralytics.com/tasks/semantic):** Dense per-pixel class labeling.
- **[Depth Estimation](https://docs.ultralytics.com/tasks/depth):** Per-pixel depth prediction from a single image.
- **Image Classification:** Standard full-image categorization.
- **Pose Estimation:** Utilizing Residual Log-Likelihood Estimation (RLE) for more accurate keypoint localization.
- **Oriented Bounding Boxes (OBB):** Including specialized angle loss to resolve boundary discontinuity issues, critical for [satellite image analysis](https://www.ultralytics.com/glossary/satellite-image-analysis).

!!! info "Ecosystem Integration"

    Both models benefit from the [Ultralytics Platform](https://platform.ultralytics.com), which provides data annotation, cloud training, and one-click deployment.

## Usage and Code Examples

The Ultralytics Python API makes switching between models simple. Because both models share the same well-maintained ecosystem, updating a YOLOv5u pipeline in the `ultralytics` package to YOLO26 only requires changing the weights file.

### Python Example

```python
from ultralytics import YOLO

# To use YOLOv5, load a v5 weights file
# model = YOLO("yolov5su.pt")

# Migrate to the recommended YOLO26 model
model = YOLO("yolo26n.pt")

# Train on the COCO8 dataset; optimizer="auto" selects MuSGD for longer runs (>10k iterations)
results = model.train(
    data="coco8.yaml",
    epochs=100,
    imgsz=640,
    batch=32,  # YOLO26's low memory footprint allows larger batch sizes
)

# Run an NMS-free inference
predictions = model("https://ultralytics.com/images/bus.jpg", nms=False)
predictions[0].show()
```

### CLI Example

You can deploy YOLO26 directly via the command line using the [TensorRT](https://developer.nvidia.com/tensorrt) integration for maximum GPU throughput:

```bash
# Export the model to TensorRT format
yolo export model=yolo26n.pt format=engine

# Run inference with the compiled engine
yolo predict model=yolo26n.engine source=path/to/video.mp4
```

## Ideal Use Cases

### When to choose YOLO26

For new computer vision projects, **YOLO26 is the recommended choice**.

- **Edge AI and IoT:** Its up to 43% faster CPU inference than YOLO11n and removal of DFL make it well suited to deployment on a [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi) or mobile devices.
- **High-Speed Pipelines:** The optional NMS-free head provides more predictable latency, which matters for autonomous robotics and real-time [security alarm systems](https://docs.ultralytics.com/guides/security-alarm-system).
- **Complex Scenarios:** If your application requires detecting small objects (e.g., [drone monitoring](https://docs.ultralytics.com/datasets/detect/visdrone)) or rotated objects (OBB), YOLO26's loss improvements (ProgLoss + STAL, plus a dedicated angle loss for OBB) improve accuracy.

### When to choose YOLOv5

- **Legacy Systems:** If your production environment has hardcoded dependencies on YOLOv5's specific anchor generation or NMS parsing logic, migrating might require a brief refactoring period.
- **Specific Academic Baselines:** Researchers often use YOLOv5 as a classic baseline to demonstrate the historical progression of [object detection architectures](https://www.ultralytics.com/glossary/object-detection-architectures).

## Summary

The transition from YOLOv5 to YOLO26 is more than an iterative update: it changes how object detection models are trained and deployed. With the MuSGD optimizer, an optional NMS-free head that drops post-processing, and faster CPU inference, **Ultralytics YOLO26** delivers a strong balance of speed and accuracy.

While YOLOv5 remains the model that made vision AI accessible to a broad developer community, developers building new production applications should start with YOLO26.
