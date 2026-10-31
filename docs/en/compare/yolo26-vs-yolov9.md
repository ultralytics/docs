---
title: YOLO26 vs YOLOv9 Comparison
comments: true
description: Compare Ultralytics YOLO26 and YOLOv9 — NMS-free inference vs PGI and GELAN, COCO mAP, T4 TensorRT speed, model size, and edge deployment tips.
keywords: YOLO26, YOLOv9, Ultralytics, NMS-free, MuSGD, ProgLoss, STAL, edge inference, CPU acceleration, real-time object detection, ONNX, TensorRT, benchmarks, deployment, Raspberry Pi, model comparison
---

# YOLO26 vs YOLOv9

This technical comparison examines the differences between **YOLO26** and **YOLOv9**, two influential models for real-time object detection. Both offer distinct architectural innovations, and understanding their performance trade-offs, deployment capabilities, and hardware requirements is crucial for selecting the right tool for your next vision project.

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO26", "YOLOv9"]'></canvas>

## YOLO26: The Edge-Optimized Powerhouse

Released in early 2026, [Ultralytics YOLO26](https://platform.ultralytics.com/ultralytics/yolo26) focuses on deployment efficiency and training stability. Its optional end-to-end inference addresses deployment bottlenecks that have long affected edge AI applications.

**Model Details:**

- **Authors:** Glenn Jocher and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com)
- **Date:** 2026-01-14
- **GitHub:** [Ultralytics Repository](https://github.com/ultralytics/ultralytics)
- **Docs:** [YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26)

### Architecture and Innovations

YOLO26 redesigns the post-processing pipeline by introducing an **End-to-End NMS-Free Design**. When run with its optional one-to-one head (`nms=False`), the model skips [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) and achieves lower latency variability. This makes deploying to mobile and edge platforms easier, especially when exporting to frameworks like [ONNX](https://onnx.ai/) and [Apple CoreML](https://developer.apple.com/documentation/coreml/).

Additionally, the removal of Distribution Focal Loss (DFL) simplifies the export process and improves compatibility with low-power edge devices. To improve training stability, YOLO26 integrates the **MuSGD Optimizer**, a hybrid of Stochastic Gradient Descent (SGD) and Muon inspired by innovations in large language model (LLM) training. This results in faster convergence.

!!! tip "Edge Device Inference"

    Thanks to architectural simplifications and the removal of DFL, YOLO26n achieves up to **43% faster CPU inference** than YOLO11n, making it a strong choice for resource-constrained edge devices like the [Raspberry Pi](https://www.raspberrypi.org/) or [NVIDIA Jetson Nano](https://docs.ultralytics.com/guides/nvidia-jetson).

For detecting small objects in scenes like [drone aerial imagery](https://www.ultralytics.com/blog/12-aerial-imagery-use-cases-powered-by-computer-vision), YOLO26 uses **ProgLoss + STAL** (Progressive Loss and Small-Target-Aware Label Assignment), which improve small-object recall. It also includes task-specific enhancements, including multi-scale proto for [instance segmentation](https://docs.ultralytics.com/tasks/segment), Residual Log-Likelihood Estimation (RLE) for [pose estimation](https://docs.ultralytics.com/tasks/pose), and specialized angle loss for detecting [Oriented Bounding Boxes (OBB)](https://docs.ultralytics.com/tasks/obb).

[Learn more about YOLO26](https://platform.ultralytics.com/ultralytics/yolo26){ .md-button }

## YOLOv9: Programmable Gradient Information

Introduced in early 2024, YOLOv9 brought theoretical advancements to the way neural networks handle gradient flow during the training phase, focusing on parameter efficiency and deep feature retention.

**Model Details:**

- **Authors:** Chien-Yao Wang and Hong-Yuan Mark Liao
- **Organization:** Institute of Information Science, Academia Sinica, Taiwan
- **Date:** 2024-02-21
- **Arxiv:** [YOLOv9 Paper](https://arxiv.org/abs/2402.13616)
- **GitHub:** [YOLOv9 Repository](https://github.com/WongKinYiu/yolov9)
- **Docs:** [YOLOv9 Documentation](https://docs.ultralytics.com/models/yolov9)

### Architecture and Strengths

YOLOv9 is built around the concept of Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN). These concepts address the information bottleneck problem often observed in deep neural networks. By preserving essential information through the feed-forward process, GELAN ensures that gradients used for weight updates remain reliable. This architecture delivers high accuracy and makes YOLOv9 a strong candidate for academic research into neural network theory and gradient path optimization using the [PyTorch](https://pytorch.org/) framework.

### Limitations

Despite its strong parameter efficiency, YOLOv9 relies on traditional NMS for bounding box post-processing, which can create computational bottlenecks during inference on edge devices. Furthermore, the official repository is largely focused on object detection, requiring custom engineering to adapt it for specialized tasks like [tracking](https://docs.ultralytics.com/modes/track) or pose estimation.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9){ .md-button }

## Performance Comparison

When evaluating these models for real-world deployment, balancing accuracy (mAP), inference speed, and memory usage is critical. Ultralytics models are known for low memory requirements during both training and inference, typically requiring less [CUDA memory](https://developer.nvidia.com/cuda) than transformer-based alternatives like [RT-DETR](https://docs.ultralytics.com/models/rtdetr).

Below is a direct comparison of YOLO26 and YOLOv9 performance on the [COCO dataset](https://cocodataset.org/). Best values in each column are highlighted in **bold**.

| Model                                                                  | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ---------------------------------------------------------------------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| [YOLO26n](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n) | 640                         | 40.9                       | **38.9**                             | **1.7**                                   | 2.4                      | **5.5**                 |
| [YOLO26s](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s) | 640                         | 48.6                       | 87.2                                 | 2.5                                       | 9.5                      | 20.9                    |
| [YOLO26m](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m) | 640                         | 53.1                       | 220.0                                | 4.7                                       | 20.4                     | 68.4                    |
| [YOLO26l](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l) | 640                         | 55.0                       | 286.2                                | 6.2                                       | 24.8                     | 86.8                    |
| [YOLO26x](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x) | 640                         | **57.5**                   | 525.8                                | 11.8                                      | 55.7                     | 194.4                   |
|                                                                        |                             |                            |                                      |                                           |                          |                         |
| YOLOv9t                                                                | 640                         | 38.3                       | -                                    | 2.3                                       | **2.0**                  | 7.7                     |
| YOLOv9s                                                                | 640                         | 46.8                       | -                                    | 3.54                                      | 7.2                      | 26.7                    |
| YOLOv9m                                                                | 640                         | 51.4                       | -                                    | 6.43                                      | 20.1                     | 76.8                    |
| YOLOv9c                                                                | 640                         | 53.0                       | -                                    | 7.16                                      | 25.5                     | 102.8                   |
| YOLOv9e                                                                | 640                         | 55.6                       | -                                    | 16.77                                     | 58.1                     | 192.5                   |

_Note: CPU ONNX speeds for YOLOv9 are not reported in this table. On T4 TensorRT, YOLO26 is faster than YOLOv9 at every size, and YOLO26 is more accurate at each comparable size (for example, 57.5 vs 55.6 mAP for YOLO26x vs YOLOv9e, with fewer parameters)._

## Use Cases and Recommendations

Choosing between YOLO26 and YOLOv9 depends on your specific project requirements, deployment constraints, and ecosystem preferences.

### When to Choose YOLO26

YOLO26 is a strong choice for:

- **NMS-Free Edge Deployment:** Applications requiring consistent, low-latency inference without the complexity of Non-Maximum Suppression post-processing.
- **CPU-Only Environments:** Devices without dedicated GPU acceleration, where YOLO26n's up to 43% faster CPU inference than YOLO11n provides a decisive advantage.
- **Small Object Detection:** Challenging scenarios like [aerial drone imagery](https://docs.ultralytics.com/datasets/detect/visdrone) or IoT sensor analysis where ProgLoss and STAL significantly boost accuracy on tiny objects.

### When to Choose YOLOv9

YOLOv9 is recommended for:

- **Information Bottleneck Research:** Academic projects studying Programmable Gradient Information (PGI) and Generalized Efficient Layer Aggregation Network (GELAN) architectures.
- **Gradient Flow Optimization Studies:** Research focused on understanding and mitigating information loss in deep network layers during training.
- **High-Accuracy Detection Benchmarking:** Scenarios where YOLOv9's strong COCO benchmark performance is needed as a reference point for architectural comparisons.

## The Ultralytics Advantage

Choosing a model involves more than reading an accuracy benchmark; the surrounding software ecosystem determines how fast you can go from data collection to production.

### Ease of Use and Ecosystem

The [Ultralytics Python API](https://docs.ultralytics.com/usage/python) keeps the workflow simple. Instead of cloning complex repositories or manually configuring distributed training scripts, developers can install the package via `pip` and start training immediately. The actively maintained [Ultralytics ecosystem](https://github.com/ultralytics/ultralytics) provides frequent updates, integrations with ML platforms like [Weights & Biases](https://wandb.ai/site), and extensive documentation.

!!! note "Other Ultralytics Models"

    If you are interested in exploring other models within the Ultralytics ecosystem, you might also consider comparing [YOLO11](https://platform.ultralytics.com/ultralytics/yolo11) or the classic [YOLOv8](https://platform.ultralytics.com/ultralytics/yolov8), both of which offer flexibility for custom applications.

### Versatility Across Vision Tasks

While YOLOv9 is primarily a detection model, YOLO26 is a general-purpose vision model. Using a single unified syntax, you can move from object detection to image segmentation, semantic segmentation, depth estimation, whole-image [classification](https://docs.ultralytics.com/tasks/classify), pose estimation, or OBB. This versatility reduces the technical debt of maintaining multiple disjointed codebases for different computer vision features.

### Efficient Training and Deployment

YOLO26 uses readily available pretrained weights and typically needs less memory than large vision transformers. Once trained, built-in export pipelines allow one-line conversions to optimized formats like [TensorRT](https://developer.nvidia.com/tensorrt) or [LiteRT](https://developers.google.com/edge/litert), smoothing the path to production.

## Code Example: Getting Started with YOLO26

The following Python snippet demonstrates how to load a pre-trained model, train it on custom data, and run inference using the Ultralytics API.

```python
from ultralytics import YOLO

# Load the latest state-of-the-art YOLO26 nano model
model = YOLO("yolo26n.pt")

# Train the model on the COCO8 dataset (optimizer='auto' selects MuSGD for longer training runs)
results = model.train(
    data="coco8.yaml",
    epochs=100,
    imgsz=640,
    device=0,  # Uses GPU 0, or use 'cpu' for CPU training
)

# Run NMS-free inference on a sample image
predictions = model("https://ultralytics.com/images/bus.jpg", nms=False)

# Display the bounding boxes and confidences
predictions[0].show()
```

With the speed, simplified architecture, and ecosystem of YOLO26, teams can bring vision AI applications to market faster and with fewer technical hurdles.
