---
comments: true
description: Compare YOLOv8 and YOLO11 for object detection. Explore their performance, architecture, and best-use cases to find the right model for your needs.
keywords: YOLOv8, YOLO11, object detection, Ultralytics, YOLO comparison, machine learning, computer vision, inference speed, model accuracy
---

# YOLOv8 vs YOLO11: Evolution of Real-Time Object Detection

Choosing the right computer vision architecture is a critical decision that impacts the speed, accuracy, and scalability of your AI projects. This guide provides an in-depth technical comparison between **Ultralytics YOLOv8**, a widely adopted industry standard released in 2023, and **Ultralytics YOLO11**, the latest evolution in the YOLO series designed for superior efficiency and performance. We will analyze their architectural differences, benchmark metrics, and ideal use cases to help you select the best model for your needs.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv8", "YOLO11"]'></canvas>

## Ultralytics YOLOv8

**Authors:** Glenn Jocher, Ayush Chaurasia, and Jing Qiu  
**Organization:** [Ultralytics](https://www.ultralytics.com/)  
**Date:** 2023-01-10  
**GitHub:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)  
**Docs:** [https://docs.ultralytics.com/models/yolov8/](https://docs.ultralytics.com/models/yolov8/)

Released in early 2023, YOLOv8 marked a significant milestone in the history of [object detection](https://docs.ultralytics.com/tasks/detect/). It introduced a unified framework that supports multiple computer vision tasks—including detection, [instance segmentation](https://docs.ultralytics.com/tasks/segment/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), and [image classification](https://docs.ultralytics.com/tasks/classify/)—within a single repository. YOLOv8 moved away from anchor-based detection to an **anchor-free** approach, which simplifies the design and improves generalization across different object shapes.

### Architecture and Key Features

YOLOv8 replaced the C3 modules found in [YOLOv5](https://docs.ultralytics.com/models/yolov5/) with the **C2f module** (Cross-Stage Partial bottleneck with two convolutions). This change improved gradient flow and feature integration while maintaining a lightweight footprint. The architecture also features a decoupled head, separating objectness, classification, and regression tasks to increase accuracy.

!!! tip "Legacy of Reliability"

    YOLOv8 has been tested in thousands of commercial applications, from [manufacturing automation](https://www.ultralytics.com/solutions/ai-in-manufacturing) to [autonomous vehicles](https://www.ultralytics.com/solutions/ai-in-automotive), establishing a reputation for stability and ease of deployment.

### Strengths and Weaknesses

- **Strengths:**
  - **Mature Ecosystem:** Supported by a vast array of community tutorials, [integrations](https://docs.ultralytics.com/integrations/), and deployment guides.
  - **Versatility:** Natively supports OBB (Oriented Bounding Box) and classification alongside standard detection.
  - **Proven Stability:** A safe choice for production environments requiring a model with a long track record.
- **Weaknesses:**
  - **Speed Efficiency:** While fast, it is outperformed by YOLO11 in CPU inference speeds and parameter efficiency.
  - **Compute Requirements:** Larger variants (L, X) demand more VRAM and FLOPs compared to the optimized YOLO11 equivalents.

```python
from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Train the model on a custom dataset
model.train(data="coco8.yaml", epochs=50, imgsz=640)
```

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## Ultralytics YOLO11

**Authors:** Glenn Jocher and Jing Qiu  
**Organization:** [Ultralytics](https://www.ultralytics.com/)  
**Date:** 2024-09-27  
**GitHub:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)  
**Docs:** [https://docs.ultralytics.com/models/yolo11/](https://docs.ultralytics.com/models/yolo11/)

**YOLO11** represents the cutting edge of the Ultralytics model family. Engineered to redefine [real-time inference](https://www.ultralytics.com/glossary/real-time-inference), it builds upon the successes of YOLOv8 but introduces substantial architectural refinements. YOLO11 focuses on maximizing accuracy while minimizing computational cost, making it the premier choice for modern AI applications ranging from edge devices to cloud servers.

### Architecture and Key Features

YOLO11 introduces the **C3k2 block** and **C2PSA** (Cross-Stage Partial with Spatial Attention) module. These components enhance the model's ability to extract intricate features and handle occlusion more effectively than previous iterations. The architecture is optimized for speed, delivering significantly faster processing times on CPUs—a critical factor for [edge AI](https://www.ultralytics.com/glossary/edge-ai) deployments where GPU resources may be unavailable.

The model maintains the unified interface characteristic of Ultralytics, ensuring that developers can switch between tasks like [OBB](https://docs.ultralytics.com/tasks/obb/) or segmentation without changing their workflow.

### Strengths and Weaknesses

- **Strengths:**
  - **Superior Efficiency:** Achieves higher mAP with up to **22% fewer parameters** than YOLOv8, reducing model size and storage needs.
  - **Faster Inference:** Optimized specifically for modern hardware, offering faster speeds on both CPU and GPU backends.
  - **Enhanced Feature Extraction:** The new backbone improves detection of small objects and performance in cluttered scenes.
  - **Lower Memory Usage:** Requires less CUDA memory during training compared to transformer-based models like [RT-DETR](https://docs.ultralytics.com/models/rtdetr/), enabling training on more accessible hardware.
- **Weaknesses:**
  - **Newer Release:** As a recent model, specific niche third-party tools may take time to fully update support, though the core Ultralytics ecosystem supports it day-one.

```python
from ultralytics import YOLO

# Load the latest YOLO11 model
model = YOLO("yolo11n.pt")

# Run inference on an image
results = model("https://ultralytics.com/images/bus.jpg")
results[0].show()
```

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

## Performance Head-to-Head

The comparison below highlights the efficiency gains of YOLO11. While YOLOv8 remains a powerful contender, YOLO11 consistently delivers higher accuracy (mAP) with reduced computational complexity (FLOPs) and faster inference speeds. This is particularly noticeable in the "Nano" and "Small" models, where YOLO11n achieves a **39.5 mAP** compared to YOLOv8n's 37.3, all while running significantly faster on CPU.

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv8n | 640                   | 37.3                 | 80.4                           | **1.47**                            | 3.2                | 8.7               |
| YOLOv8s | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |
|         |                       |                      |                                |                                     |                    |                   |
| YOLO11n | 640                   | **39.5**             | **56.1**                       | 1.5                                 | **2.6**            | **6.5**           |
| YOLO11s | 640                   | **47.0**             | **90.0**                       | **2.5**                             | **9.4**            | **21.5**          |
| YOLO11m | 640                   | **51.5**             | **183.2**                      | **4.7**                             | **20.1**           | **68.0**          |
| YOLO11l | 640                   | **53.4**             | **238.6**                      | **6.2**                             | **25.3**           | **86.9**          |
| YOLO11x | 640                   | **54.7**             | **462.8**                      | **11.3**                            | **56.9**           | **194.9**         |

!!! note "Metric Analysis"

    YOLO11 demonstrates a clear advantage in the **speed-accuracy trade-off**. For example, the YOLO11l model surpasses the YOLOv8l in accuracy (+0.5 mAP) while using roughly **42% fewer parameters** and running **36% faster on CPU**.

## Ecosystem and Ease of Use

Both models benefit from the robust [Ultralytics ecosystem](https://github.com/ultralytics/ultralytics), which is designed to democratize AI by making state-of-the-art technology accessible to everyone.

- **Unified API:** Switching between YOLOv8 and YOLO11 is as simple as changing the model string from `yolov8n.pt` to `yolo11n.pt`. No code refactoring is required.
- **Training Efficiency:** Ultralytics provides [auto-downloading datasets](https://docs.ultralytics.com/datasets/) and pre-trained weights, streamlining the pipeline from data collection to model training.
- **Deployment Versatility:** Both models support one-click [export](https://docs.ultralytics.com/modes/export/) to formats like ONNX, TensorRT, CoreML, and TFLite, facilitating deployment on diverse hardware including Raspberry Pis, mobile phones, and cloud instances.
- **Well-Maintained:** Frequent updates ensure compatibility with the latest versions of PyTorch and CUDA, backed by an active community on [Discord](https://discord.com/invite/ultralytics) and GitHub.

## Conclusion and Recommendations

While **YOLOv8** remains a dependable and highly capable model suitable for maintaining legacy systems, **YOLO11** is the clear recommendation for all new development.

- **Choose YOLO11 if:** You need the highest possible accuracy, faster inference speeds (especially on CPU), or are deploying to resource-constrained edge devices where memory and storage are premium. Its architectural improvements provide a future-proof foundation for commercial applications.
- **Choose YOLOv8 if:** You have an existing pipeline heavily tuned for v8 specific behaviors or are constrained by strict project requirements that prevent updating to the latest architecture.

For those interested in exploring other architectures, the Ultralytics docs also cover models like [YOLOv9](https://docs.ultralytics.com/models/yolov9/), [YOLOv10](https://docs.ultralytics.com/models/yolov10/), and [RT-DETR](https://docs.ultralytics.com/models/rtdetr/). You can view broader comparisons on our [model comparison page](https://docs.ultralytics.com/compare/).
