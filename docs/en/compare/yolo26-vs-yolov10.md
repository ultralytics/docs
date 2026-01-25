---
comments: true
description: Technical comparison of Ultralytics YOLO26 and YOLOv10 — NMS-free end-to-end detection, CPU/edge speed, accuracy, architecture, and deployment tips.
keywords: YOLO26, YOLOv10, Ultralytics, object detection, NMS-free, end-to-end detection, edge AI, CPU inference, model comparison, ONNX, TensorRT, deployment, instance segmentation, pose estimation, MuSGD, ProgLoss, DFL removal, benchmark, computer vision, real-time detection
---

# YOLO26 vs. YOLOv10: The Evolution of End-to-End Object Detection

The landscape of real-time [object detection](https://docs.ultralytics.com/tasks/detect/) changes rapidly. In 2024, **YOLOv10** made headlines by pioneering a non-maximum suppression (NMS) free training approach, effectively removing a significant bottleneck in inference pipelines. Fast forward to 2026, and **Ultralytics YOLO26** has refined and expanded upon these concepts, delivering a natively end-to-end architecture that is faster, more accurate, and deeply integrated into the Ultralytics ecosystem.

This guide provides a technical comparison between these two influential models, helping developers, researchers, and engineers choose the right tool for their [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) applications.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO26", "YOLOv10"]'></canvas>

## Performance Metrics Comparison

When evaluating modern detectors, the trade-off between speed and accuracy is paramount. **YOLO26** introduces significant optimizations specifically targeting edge devices and CPU inference, achieving up to a **43% speed increase on CPUs** compared to previous generations. While **YOLOv10** remains a highly efficient model, YOLO26 pushes the boundaries of what is possible with lighter computational resources.

| Model    | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| -------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLO26n  | 640                   | **40.9**             | **38.9**                       | 1.7                                 | 2.4                | **5.4**           |
| YOLO26s  | 640                   | **48.6**             | 87.2                           | **2.5**                             | 9.5                | **20.7**          |
| YOLO26m  | 640                   | **53.1**             | 220.0                          | **4.7**                             | 20.4               | 68.2              |
| YOLO26l  | 640                   | **55.0**             | 286.2                          | **6.2**                             | **24.8**           | **86.4**          |
| YOLO26x  | 640                   | **57.5**             | 525.8                          | **11.8**                            | **55.7**           | 193.9             |
|          |                       |                      |                                |                                     |                    |                   |
| YOLOv10n | 640                   | 39.5                 | -                              | **1.56**                            | **2.3**            | 6.7               |
| YOLOv10s | 640                   | 46.7                 | -                              | 2.66                                | **7.2**            | 21.6              |
| YOLOv10m | 640                   | 51.3                 | -                              | 5.48                                | **15.4**           | **59.1**          |
| YOLOv10b | 640                   | 52.7                 | -                              | 6.54                                | 24.4               | 92.0              |
| YOLOv10l | 640                   | 53.3                 | -                              | 8.33                                | 29.5               | 120.3             |
| YOLOv10x | 640                   | 54.4                 | -                              | 12.2                                | 56.9               | **160.4**         |

## Architectural Innovations

### Ultralytics YOLO26: The New Standard

**Authors:** Glenn Jocher, Jing Qiu  
**Organization:** [Ultralytics](https://www.ultralytics.com)  
**Date:** January 14, 2026

YOLO26 represents the culmination of research into efficiency and ease of use. It adopts an **End-to-End NMS-Free Design**, similar to YOLOv10, but enhances it with several key architectural changes designed for robustness and deployment flexibility.

1.  **DFL Removal:** By removing Distribution Focal Loss (DFL), the model architecture is simplified. This change is crucial for export compatibility, making the model easier to deploy on restricted edge hardware like [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/) or mobile devices where complex output layers can cause latency.
2.  **MuSGD Optimizer:** Inspired by the training stability of Large Language Models (LLMs), YOLO26 utilizes a hybrid optimizer combining SGD and Muon. This innovation, adapted from Moonshot AI's Kimi K2, ensures faster convergence and stable training runs, reducing the cost of compute.
3.  **ProgLoss + STAL:** The introduction of Progressive Loss (ProgLoss) and Soft-Target Anchor Loss (STAL) significantly boosts performance on small objects. This makes YOLO26 particularly adept at tasks like aerial imagery analysis or [defect detection](https://www.ultralytics.com/blog/using-ai-for-crack-detection-and-segmentation) in manufacturing.

[Learn more about YOLO26](https://docs.ultralytics.com/models/yolo26/){ .md-button }

### YOLOv10: The NMS-Free Pioneer

**Authors:** Ao Wang et al.  
**Organization:** Tsinghua University  
**Date:** May 23, 2024

YOLOv10 was a landmark release that addressed the redundancy of NMS post-processing. Its primary innovation was the use of **Consistent Dual Assignments** for NMS-free training.

- **Dual Assignments:** During training, the model uses both one-to-many and one-to-one label assignments. This allows the model to learn rich representations while ensuring that during inference, only one prediction is made per object, eliminating the need for NMS.
- **Holistic Efficiency Design:** The authors introduced lightweight classification heads and spatial-channel decoupled downsampling to reduce computational overhead, which is reflected in its low [FLOPs](https://www.ultralytics.com/glossary/flops) count.

[Learn more about YOLOv10](https://docs.ultralytics.com/models/yolov10/){ .md-button }

!!! note "The NMS Bottleneck"

    Non-Maximum Suppression (NMS) is a post-processing step used to filter overlapping bounding boxes. While effective, it introduces latency variance and complicates deployment. Both YOLO26 and YOLOv10 remove this step, making inference times deterministic and faster.

## Integration and Ecosystem

One of the most significant differences lies in the surrounding ecosystem. **Ultralytics YOLO26** is the flagship model of the Ultralytics library, ensuring immediate support for all tasks and modes.

### The Ultralytics Advantage

- **Versatility:** While YOLOv10 focuses primarily on detection, YOLO26 offers native support for [Instance Segmentation](https://docs.ultralytics.com/tasks/segment/), [Pose Estimation](https://docs.ultralytics.com/tasks/pose/), [OBB](https://docs.ultralytics.com/tasks/obb/), and Classification.
- **Ultralytics Platform:** YOLO26 is fully integrated with the [Ultralytics Platform](https://platform.ultralytics.com) (formerly HUB), allowing for seamless dataset management, one-click cloud training, and deployment to formats like [TFLite](https://docs.ultralytics.com/integrations/tflite/) and [OpenVINO](https://docs.ultralytics.com/integrations/openvino/).
- **Maintenance:** As a core product, YOLO26 receives frequent updates, bug fixes, and community support via [GitHub](https://github.com/ultralytics/ultralytics) and [Discord](https://discord.com/invite/ultralytics).

### Code Comparison

Both models can be run using the `ultralytics` Python package, highlighting the library's flexibility. However, YOLO26 benefits from the latest utility functions and optimizations.

```python
from ultralytics import YOLO

# ----------------- YOLO26 -----------------
# Load the latest YOLO26 model (NMS-free, optimized for CPU)
model_26 = YOLO("yolo26n.pt")

# Train on a custom dataset with MuSGD optimizer enabled automatically
model_26.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference with simplified output (no NMS overhead)
results_26 = model_26("path/to/image.jpg")


# ----------------- YOLOv10 -----------------
# Load the YOLOv10 model (Historical academic checkpoint)
model_10 = YOLO("yolov10n.pt")

# Train using standard settings
model_10.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference
results_10 = model_10("path/to/image.jpg")
```

## Use Cases and Recommendations

Choosing between these models depends on your specific deployment constraints and project goals.

### Ideal Scenarios for YOLO26

- **Edge AI on CPU:** If your application runs on hardware without a dedicated GPU (e.g., standard laptops, low-power IoT gateways), YOLO26's **43% faster CPU inference** makes it the undisputed choice.
- **Commercial Solutions:** For enterprise applications requiring long-term maintainability, strict licensing clarity ([Enterprise License](https://www.ultralytics.com/license)), and reliable support, YOLO26 is designed for production.
- **Complex Tasks:** Projects requiring [oriented bounding boxes](https://docs.ultralytics.com/tasks/obb/) for aerial survey or [pose estimation](https://docs.ultralytics.com/tasks/pose/) for sports analytics will benefit from YOLO26's multi-task capabilities.

### Ideal Scenarios for YOLOv10

- **Academic Research:** Researchers studying the theoretical underpinnings of NMS-free training or label assignment strategies will find YOLOv10's [arXiv paper](https://arxiv.org/abs/2405.14458) and architecture a valuable reference.
- **Legacy Benchmarking:** For comparing against 2024-era baselines, YOLOv10 serves as an excellent standard for efficiency-focused architectures.

!!! tip "Deployment Flexibility"

    Ultralytics models excel at exportability. You can easily export a trained YOLO26 model to [ONNX](https://docs.ultralytics.com/integrations/onnx/), TensorRT, or CoreML with a single command: `yolo export model=yolo26n.pt format=onnx`.

## Conclusion

Both architectures have played pivotal roles in advancing computer vision. **YOLOv10** successfully challenged the necessity of NMS, proving that end-to-end detection was viable for real-time applications.

**Ultralytics YOLO26** takes that breakthrough and perfects it. By combining the NMS-free design with the stability of the MuSGD optimizer, the edge-friendly removal of DFL, and the versatile support of the Ultralytics ecosystem, YOLO26 offers the most balanced, high-performance solution for developers today. Whether you are building a [smart city traffic system](https://www.ultralytics.com/blog/computer-vision-ai-in-smart-cities) or a mobile document scanner, YOLO26 provides the speed and accuracy required for success.

## Further Reading

- [YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26/)
- [Object Detection Task Guide](https://docs.ultralytics.com/tasks/detect/)
- [YOLO Performance Metrics Explained](https://docs.ultralytics.com/guides/yolo-performance-metrics/)
- [Ultralytics Platform for Model Training](https://platform.ultralytics.com)
- [Guide to Model Export Modes](https://docs.ultralytics.com/modes/export/)
