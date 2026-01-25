---
comments: true
description: Compare YOLO11 and YOLOv8 architectures, performance, use cases, and benchmarks. Discover which YOLO model fits your object detection needs.
keywords: YOLO11, YOLOv8, object detection, model comparison, performance benchmarks, YOLO series, computer vision, Ultralytics YOLO, YOLO architecture
---

# YOLO11 vs. YOLOv8: The Evolution of Real-Time Object Detection

The progression of the YOLO (You Only Look Once) architecture has consistently redefined the boundaries of computer vision. **YOLO11**, released in late 2024, builds upon the solid foundation laid by **YOLOv8** to deliver enhanced efficiency and accuracy. This analysis explores the architectural shifts, performance metrics, and practical deployment considerations for both models, guiding developers toward the optimal choice for their specific applications.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO11", "YOLOv8"]'></canvas>

## Performance Metrics at a Glance

The following table highlights the performance improvements of YOLO11 over YOLOv8 across various model sizes. YOLO11 generally offers higher Mean Average Precision (mAP) while maintaining competitive inference speeds, particularly when optimized for CPU execution.

| Model       | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ----------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| **YOLO11n** | 640                   | **39.5**             | **56.1**                       | 1.5                                 | **2.6**            | **6.5**           |
| **YOLO11s** | 640                   | **47.0**             | **90.0**                       | **2.5**                             | **9.4**            | **21.5**          |
| **YOLO11m** | 640                   | **51.5**             | **183.2**                      | **4.7**                             | **20.1**           | **68.0**          |
| **YOLO11l** | 640                   | **53.4**             | **238.6**                      | **6.2**                             | **25.3**           | **86.9**          |
| **YOLO11x** | 640                   | **54.7**             | **462.8**                      | **11.3**                            | **56.9**           | **194.9**         |
|             |                       |                      |                                |                                     |                    |                   |
| YOLOv8n     | 640                   | 37.3                 | 80.4                           | **1.47**                            | 3.2                | 8.7               |
| YOLOv8s     | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m     | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l     | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x     | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |

## Architectural Overview

Both models stem from the Ultralytics philosophy of user-centric design, prioritizing ease of use without sacrificing power. However, architectural refinements in YOLO11 allow for greater feature extraction capabilities with fewer parameters.

### YOLO11: Refined Efficiency

**Authors:** Glenn Jocher, Jing Qiu  
**Organization:** [Ultralytics](https://www.ultralytics.com/)  
**Date:** 2024-09-27  
**GitHub:** [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)  
**Docs:** [YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/)

YOLO11 introduces an updated backbone and neck architecture that enhances feature integration. By optimizing the [Cross Stage Partial (CSP)](https://www.ultralytics.com/glossary/cross-validation) blocks, YOLO11 achieves a significant reduction in FLOPs (Floating Point Operations per Second) while boosting mAP. This efficiency makes it particularly well-suited for constrained environments like [edge computing devices](https://www.ultralytics.com/glossary/edge-computing).

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

### YOLOv8: The Reliable Standard

**Authors:** Glenn Jocher, Ayush Chaurasia, Jing Qiu  
**Organization:** [Ultralytics](https://www.ultralytics.com/)  
**Date:** 2023-01-10  
**GitHub:** [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)  
**Docs:** [YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8/)

YOLOv8 remains a robust and highly versatile model. It pioneered the anchor-free detection head in the Ultralytics lineup, simplifying the [training process](https://docs.ultralytics.com/modes/train/) by eliminating the need for manual anchor box calculations. Its proven track record in diverse industries, from [agriculture](https://www.ultralytics.com/solutions/ai-in-agriculture) to manufacturing, makes it a safe and reliable choice for legacy systems.

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

!!! note "Architecture Compatibility"

    Both YOLO11 and YOLOv8 are natively supported by the `ultralytics` Python package. Switching between them is often as simple as changing the model name string (e.g., from `yolov8n.pt` to `yolo11n.pt`) in your code, preserving your existing [dataset configuration](https://docs.ultralytics.com/guides/model-yaml-config/) and training pipelines.

## Key Advantages of Ultralytics Models

Regardless of the specific version, choosing an Ultralytics model provides distinct benefits over other frameworks.

1.  **Well-Maintained Ecosystem:** Both models benefit from active development and community support. Regular updates ensure compatibility with the latest versions of [PyTorch](https://pytorch.org/) and CUDA, minimizing technical debt.
2.  **Memory Requirements:** Ultralytics engineers its models to be memory-efficient. Compared to massive [transformer-based detectors](https://docs.ultralytics.com/models/rtdetr/), YOLO models require significantly less GPU memory (VRAM) during training, making them accessible to developers using consumer-grade hardware.
3.  **Versatility:** Beyond simple bounding boxes, both architectures support [Instance Segmentation](https://docs.ultralytics.com/tasks/segment/), [Pose Estimation](https://docs.ultralytics.com/tasks/pose/), [OBB (Oriented Bounding Box)](https://docs.ultralytics.com/tasks/obb/), and Classification.
4.  **Training Efficiency:** Pre-trained weights are readily available, allowing for transfer learning that drastically reduces training time and energy consumption.

## Real-World Use Cases

The choice between YOLO11 and YOLOv8 often depends on the specific constraints of the deployment environment.

### Where YOLO11 Excels

YOLO11 is the superior choice for **latency-sensitive edge applications**. Its reduced parameter count and lower FLOPs translate to faster inference on CPUs and mobile processors.

- **Smart Retail:** For real-time [customer behavior analysis](https://www.ultralytics.com/solutions/ai-in-retail) on store servers without dedicated GPUs.
- **Drone Imagery:** Processing high-resolution aerial footage where every millisecond of battery life counts. The improved small-object detection is critical here.
- **Mobile Apps:** Deploying via [CoreML](https://docs.ultralytics.com/integrations/coreml/) or [TFLite](https://docs.ultralytics.com/integrations/tflite/) to iOS and Android devices benefits from the lighter architecture.

### Where YOLOv8 Remains Strong

YOLOv8 is ideal for **established workflows** where consistency is paramount.

- **Industrial Automation:** In factories already standardized on YOLOv8 for [quality control](https://www.ultralytics.com/solutions/ai-in-manufacturing), continuing with v8 avoids the need for re-validation of the entire pipeline.
- **Academic Research:** As a highly cited baseline, YOLOv8 serves as an excellent reference point for comparing new architectural novelties.

## Ease of Use and Implementation

One of the hallmarks of the Ultralytics ecosystem is the unified API. Developers can train, validate, and deploy either model using identical syntax.

```python
from ultralytics import YOLO

# Load a model (switch 'yolo11n.pt' to 'yolov8n.pt' to use v8)
model = YOLO("yolo11n.pt")

# Train the model on the COCO8 dataset
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference on a local image
results = model("path/to/image.jpg")

# Export to ONNX for deployment
path = model.export(format="onnx")
```

This simplicity extends to the [Command Line Interface (CLI)](https://docs.ultralytics.com/usage/cli/), allowing for rapid prototyping without writing a single line of Python.

```bash
# Train YOLO11n
yolo train model=yolo11n.pt data=coco8.yaml epochs=50 imgsz=640

# Train YOLOv8n
yolo train model=yolov8n.pt data=coco8.yaml epochs=50 imgsz=640
```

## Conclusion

Both **YOLO11** and **YOLOv8** represent the pinnacle of real-time object detection technology. **YOLOv8** remains a dependable and versatile workhorse, perfect for general-purpose applications. However, **YOLO11** pushes the envelope further with optimized efficiency, making it the recommended starting point for new projects—especially those targeting edge devices or requiring the highest possible accuracy-to-compute ratio.

For developers seeking the absolute cutting edge in performance and NMS-free architecture, we also recommend exploring the newly released **[YOLO26](https://docs.ultralytics.com/models/yolo26/)**. It combines the best traits of previous generations with an end-to-end design that simplifies deployment even further.

### Explore Other Models

- **[YOLO26](https://docs.ultralytics.com/models/yolo26/):** The latest state-of-the-art model featuring end-to-end NMS-free detection and 43% faster CPU inference.
- **[RT-DETR](https://docs.ultralytics.com/models/rtdetr/):** A transformer-based model offering high accuracy, ideal when inference speed is secondary to precision.
- **[SAM 2](https://docs.ultralytics.com/models/sam-2/):** Meta's Segment Anything Model, perfect for zero-shot segmentation tasks where training data is scarce.
