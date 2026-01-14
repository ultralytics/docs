---
comments: true
description: Compare YOLOv10 and YOLOv5 models for object detection. Explore key features, performance metrics, strengths, and use cases to choose the right model.
keywords: YOLOv10, YOLOv5, object detection, real-time models, computer vision, NMS-free, model comparison, YOLO, Ultralytics, machine learning
---

# YOLOv10 vs YOLOv5: Architecture and Performance Comparison

In the rapidly evolving landscape of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv), choosing the right object detection model is critical for project success. This comparison explores two significant milestones in the YOLO (You Only Look Once) history: **YOLOv10**, a recent academic breakthrough introducing NMS-free detection, and **YOLOv5**, the legendary Ultralytics model that defined industry standards for ease of use and deployment.

While YOLOv5 remains a reliable workhorse for countless deployed applications, YOLOv10 pushes the boundaries of latency and accuracy. For developers seeking the absolute latest in [end-to-end efficiency](https://www.ultralytics.com/glossary/machine-learning-operations-mlops), the newly released **YOLO26** builds upon these innovations, offering native NMS-free design and enhanced edge optimization.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv10", "YOLOv5"]'></canvas>

## Detailed Performance Metrics

The following table provides a side-by-side technical comparison of model variants, highlighting improvements in [mean Average Precision (mAP)](https://www.ultralytics.com/glossary/mean-average-precision-map) and inference speed.

| Model    | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| -------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv10n | 640                   | **39.5**             | -                              | 1.56                                | **2.3**            | **6.7**           |
| YOLOv10s | 640                   | **46.7**             | -                              | 2.66                                | **7.2**            | **21.6**          |
| YOLOv10m | 640                   | **51.3**             | -                              | 5.48                                | **15.4**           | **59.1**          |
| YOLOv10b | 640                   | 52.7                 | -                              | 6.54                                | 24.4               | 92.0              |
| YOLOv10l | 640                   | 53.3                 | -                              | 8.33                                | **29.5**           | **120.3**         |
| YOLOv10x | 640                   | **54.4**             | -                              | 12.2                                | **56.9**           | **160.4**         |
|          |                       |                      |                                |                                     |                    |                   |
| YOLOv5n  | 640                   | 28.0                 | **73.6**                       | **1.12**                            | 2.6                | 7.7               |
| YOLOv5s  | 640                   | 37.4                 | 120.7                          | **1.92**                            | 9.1                | 24.0              |
| YOLOv5m  | 640                   | 45.4                 | 233.9                          | **4.03**                            | 25.1               | 64.2              |
| YOLOv5l  | 640                   | 49.0                 | 408.4                          | **6.61**                            | 53.2               | 135.0             |
| YOLOv5x  | 640                   | 50.7                 | 763.2                          | **11.89**                           | 97.2               | 246.4             |

## YOLOv10: The End-to-End Innovator

Released in May 2024 by researchers at [Tsinghua University](https://www.tsinghua.edu.cn/en/), YOLOv10 represents a significant architectural shift. Its primary innovation is the elimination of [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms), a post-processing step traditionally required to remove duplicate bounding boxes.

### Architecture and Innovation

YOLOv10 introduces **Consistent Dual Assignments**, a training strategy that combines one-to-many supervision (for rich signal) with one-to-one supervision (for efficient inference). This allows the model to predict a single best bounding box per object directly, rendering NMS obsolete.

Additional architectural features include:

- **Holisitic Efficiency-Accuracy Design:** Optimized decoupling of spatial and channel downsampling.
- **Large-Kernel Convolutions:** Enhanced [receptive fields](https://www.ultralytics.com/glossary/receptive-field) for better context understanding in complex scenes.
- **Partial Self-Attention (PSA):** Efficient global representation learning with minimal computational overhead.

**Key Details:**

- **Authors:** Ao Wang, Hui Chen, et al.
- **Organization:** Tsinghua University
- **Date:** 2024-05-23
- **Research Paper:** [arXiv:2405.14458](https://arxiv.org/abs/2405.14458)
- **Code:** [GitHub Repository](https://github.com/THU-MIG/yolov10)

[Learn more about YOLOv10](https://docs.ultralytics.com/models/yolov10/){ .md-button }

## YOLOv5: The Industry Standard for Reliability

Since its release in June 2020 by [Ultralytics](https://www.ultralytics.com/), **YOLOv5** has become one of the most deployed object detection models in history. While newer models may surpass it in raw metrics, YOLOv5's strength lies in its unmatched stability, extensive platform support, and "it just works" philosophy.

### Why It Remains Relevant

YOLOv5 introduced a user-friendly [PyTorch](https://pytorch.org/) workflow that democratized object detection. It features a CSPNet backbone and a PANet neck, optimizing gradient flow and feature fusion.

- **Exportability:** Known for seamless export to [ONNX](https://onnx.ai/), TensorRT, CoreML, and TFLite.
- **Versatility:** Supports detection, [instance segmentation](https://docs.ultralytics.com/tasks/segment/), and classification.
- **Low Memory Footprint:** Efficient training on consumer-grade hardware compared to transformer-heavy architectures.

**Key Details:**

- **Author:** Glenn Jocher
- **Organization:** Ultralytics
- **Date:** 2020-06-26
- **Repository:** [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)

[Learn more about YOLOv5](https://docs.ultralytics.com/models/yolov5/){ .md-button }

## Critical Comparison: Architecture and Use Cases

### 1. NMS vs. End-to-End

The most defining difference is post-processing. YOLOv5 relies on NMS, which requires sorting and filtering thousands of candidate boxes. This can introduce latency variability, especially in scenes with many objects (e.g., [crowd counting](https://docs.ultralytics.com/guides/queue-management/)). YOLOv10 removes this step, offering more deterministic inference times.

!!! note "The Future is NMS-Free"

    The industry is moving toward end-to-end architectures. The new **YOLO26** model adopts this NMS-free design natively, combining the speed of YOLOv10 with the robust ecosystem of Ultralytics.

### 2. Training Efficiency and Compute

Ultralytics models like YOLOv5 are famous for their training efficiency. They require significantly less [CUDA memory](https://developer.nvidia.com/cuda) than transformer-based detectors like RT-DETR. YOLOv10 maintains this efficiency while improving parameter utilization—YOLOv10s achieves significantly higher AP than YOLOv5s with fewer parameters (7.2M vs 9.1M).

### 3. Ease of Use and Ecosystem

While YOLOv10 provides excellent weights and architecture, YOLOv5 benefits from years of ecosystem development. The [Ultralytics Platform](https://www.ultralytics.com/) (formerly HUB) allows for one-click training and deployment. Users who value a well-maintained ecosystem with frequent updates often prefer the stability of Ultralytics-native models.

## Real-World Applications

- **YOLOv10 Ideal Use Cases:**
    - **High-Speed Robotics:** Where latency jitter from NMS is unacceptable.
    - **Edge Devices:** The reduced parameter count (e.g., YOLOv10n at 2.3M params) makes it excellent for microcontrollers.
    - **Academic Research:** A strong baseline for studying [label assignment](https://www.ultralytics.com/glossary/data-labeling) strategies.

- **YOLOv5 Ideal Use Cases:**
    - **Industrial Inspection:** Validated reliability for manufacturing lines detecting defects.
    - **Mobile Apps:** Proven compatibility with iOS (CoreML) and Android (TFLite) pipelines.
    - **Legacy Systems:** Easy integration into existing PyTorch 1.x workflows.

## The Next Step: YOLO26

For developers starting new projects in 2026, **Ultralytics YOLO26** offers the best of both worlds. It integrates the NMS-free breakthrough pioneered by YOLOv10 but refines it for even greater stability and speed.

**YOLO26 Advantages:**

- **MuSGD Optimizer:** Inspired by LLM training (Moonshot AI's Kimi K2), offering faster convergence.
- **DFL Removal:** Simplified loss functions for easier export to restricted edge hardware.
- **Task Versatility:** State-of-the-art performance across [OBB](https://docs.ultralytics.com/tasks/obb/), Pose, and Segmentation, which are not fully covered by the original YOLOv10 release.

[Learn more about YOLO26](https://docs.ultralytics.com/models/yolo26/){ .md-button }

## Code Example: Running Both Models

The `ultralytics` Python package provides a unified API to run inference on both models seamlessly.

```python
from ultralytics import YOLO

# Load the legendary YOLOv5 model (updated for v8 compatibility)
model_v5 = YOLO("yolov5su.pt")

# Load the NMS-free YOLOv10 model
model_v10 = YOLO("yolov10n.pt")

# Run inference on an image
results_v5 = model_v5("https://ultralytics.com/images/bus.jpg")
results_v10 = model_v10("https://ultralytics.com/images/bus.jpg")

# Display results
results_v5[0].show()  # Traditional NMS-based detection
results_v10[0].show()  # End-to-end NMS-free detection
```

## Conclusion

Both models hold an important place in computer vision history. **YOLOv5** remains a trusted choice for its robust ecosystem and ease of deployment, while **YOLOv10** offers a glimpse into the efficiency of end-to-end detection. However, for those seeking maximum performance, efficiency, and future-proofing, **YOLO26** is the recommended successor, combining the NMS-free architecture of v10 with the unparalleled support of the Ultralytics ecosystem.

For other versatile models, explore [YOLO11](https://docs.ultralytics.com/models/yolo11/) or the transformer-based [RT-DETR](https://docs.ultralytics.com/models/rtdetr/).
