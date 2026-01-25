---
comments: true
description: Compare YOLOv5 and YOLOv8 for speed, accuracy, and versatility. Learn which Ultralytics model is best for your object detection and vision tasks.
keywords: YOLOv5, YOLOv8, Ultralytics, object detection, computer vision, YOLO models, model comparison, AI, machine learning, deep learning
---

# YOLOv5 vs. YOLOv8: The Evolution of Ultralytics Real-Time Detection

The field of computer vision has advanced rapidly, driven significantly by the continuous innovation within the YOLO (You Only Look Once) family of object detectors. Two of the most impactful versions in this lineage are **YOLOv5** and **YOLOv8**, both developed by Ultralytics. While YOLOv5 set the industry standard for ease of use and flexibility upon its release in 2020, YOLOv8 (released in 2023) introduced architectural breakthroughs that redefined state-of-the-art performance.

This guide provides an in-depth technical comparison to help developers, researchers, and engineers choose the right model for their specific application needs, while also highlighting the newest advancements in the field, such as **YOLO26**.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv5", "YOLOv8"]'></canvas>

## Ultralytics YOLOv5: The Industry Standard

Released in June 2020, [YOLOv5](https://docs.ultralytics.com/models/yolov5/) marked a pivotal moment in the democratization of AI. Unlike its predecessors, which were primarily written in C (Darknet), YOLOv5 was the first native [PyTorch](https://pytorch.org/) implementation, making it exceptionally accessible to the Python developer community.

### Key Features and Architecture

YOLOv5 is celebrated for its balance of speed, accuracy, and user-friendly design. Its architecture introduced several key improvements over YOLOv4:

- **CSPDarknet Backbone:** Utilizes Cross-Stage Partial connections to improve gradient flow and reduce parameters.
- **Auto-Learning Anchor Boxes:** Automatically learns optimal anchor box dimensions for the custom dataset before training begins.
- **Mosaic Data Augmentation:** A training technique that combines four images into one, enhancing the model's ability to detect smaller objects and improving context generalization.

**Technical Specifications:**

- **Authors:** Glenn Jocher
- **Organization:** [Ultralytics](https://www.ultralytics.com/)
- **Date:** 2020-06-26
- **GitHub:** [ultralytics/yolov5](https://github.com/ultralytics/yolov5)

[Learn more about YOLOv5](https://docs.ultralytics.com/models/yolov5/){ .md-button }

## Ultralytics YOLOv8: Defining State-of-the-Art

Launched in January 2023, [YOLOv8](https://docs.ultralytics.com/models/yolov8/) represented a significant leap forward in computer vision technology. It moved away from the anchor-based detection used in YOLOv5 to an anchor-free design, simplifying the learning process and improving generalization across different object shapes.

### Architectural Innovations

YOLOv8 introduced a host of modern techniques that boosted both speed and accuracy:

- **Anchor-Free Detection:** Eliminates the need for manual anchor box configuration, predicting object centers directly. This reduces the number of box predictions and speeds up Non-Maximum Suppression (NMS).
- **C2f Module:** Replaces the C3 module from YOLOv5, offering richer gradient flow and adjusting the channel numbers for better feature extraction.
- **Decoupled Head:** Separates the objectness, classification, and regression tasks into different branches, allowing each to converge more effectively.
- **Task Versatility:** Designed from the ground up to support not just detection, but also [instance segmentation](https://docs.ultralytics.com/tasks/segment/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), [classification](https://docs.ultralytics.com/tasks/classify/), and [OBB](https://docs.ultralytics.com/tasks/obb/) (Oriented Bounding Box).

**Technical Specifications:**

- **Authors:** Glenn Jocher, Ayush Chaurasia, and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com/)
- **Date:** 2023-01-10
- **GitHub:** [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## Performance Comparison

When comparing these two powerhouses, it is clear that YOLOv8 generally outperforms YOLOv5 in both accuracy (mAP) and latency on comparable hardware. However, YOLOv5 remains a highly capable model that is extremely efficient for legacy systems.

The table below highlights performance on the [COCO dataset](https://docs.ultralytics.com/datasets/detect/coco/). **Bold** values indicate the best performance in each category.

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv5n | 640                   | 28.0                 | **73.6**                       | **1.12**                            | **2.6**            | **7.7**           |
| YOLOv5s | 640                   | 37.4                 | **120.7**                      | **1.92**                            | **9.1**            | **24.0**          |
| YOLOv5m | 640                   | 45.4                 | **233.9**                      | **4.03**                            | **25.1**           | **64.2**          |
| YOLOv5l | 640                   | 49.0                 | 408.4                          | **6.61**                            | 53.2               | **135.0**         |
| YOLOv5x | 640                   | 50.7                 | 763.2                          | **11.89**                           | 97.2               | **246.4**         |
|         |                       |                      |                                |                                     |                    |                   |
| YOLOv8n | 640                   | **37.3**             | 80.4                           | 1.47                                | 3.2                | 8.7               |
| YOLOv8s | 640                   | **44.9**             | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m | 640                   | **50.2**             | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l | 640                   | **52.9**             | **375.2**                      | 9.06                                | **43.7**           | 165.2             |
| YOLOv8x | 640                   | **53.9**             | **479.1**                      | 14.37                               | **68.2**           | 257.8             |

!!! info "Analysis"

    YOLOv8n (Nano) achieves a significantly higher mAP (37.3) compared to YOLOv5n (28.0) with only a marginal increase in parameter count. This efficiency gain makes YOLOv8 the superior choice for modern edge applications where every percentage of accuracy counts.

## Training and Ecosystem

Both models benefit immensely from the [Ultralytics ecosystem](https://docs.ultralytics.com/), which prioritizes **Ease of Use**.

### Simplified Training Workflow

The transition from YOLOv5 to YOLOv8 also introduced a unified CLI and Python API that supports all tasks. While YOLOv5 relied on specific scripts (e.g., `train.py`, `detect.py`), YOLOv8 and subsequent models like **YOLO26** use a modular package structure.

**YOLOv5 Training:**

```bash
python train.py --img 640 --batch 16 --epochs 50 --data coco128.yaml --weights yolov5s.pt
```

**YOLOv8 Training:**

```bash
yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640
```

### The Ultralytics Platform Advantage

Both models integrate seamlessly with the [Ultralytics Platform](https://platform.ultralytics.com/). This allows users to visualize training runs, manage datasets, and perform one-click [model export](https://docs.ultralytics.com/modes/export/) to formats like [ONNX](https://docs.ultralytics.com/integrations/onnx/), [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/), and CoreML without writing complex conversion scripts.

## Use Cases and Recommendations

Choosing between these two models depends on your specific constraints, although newer models are generally recommended for new projects.

### Ideal Scenarios for YOLOv5

- **Legacy Systems:** Projects already deeply integrated with the YOLOv5 codebase where migration costs are high.
- **Specific Hardware Support:** Certain older edge AI accelerators may have highly optimized kernels specifically tuned for YOLOv5's specific layer structures.
- **Simplicity:** For purely educational purposes, the explicit script-based structure of the YOLOv5 repository can be easier for beginners to dissect line-by-line.

### Ideal Scenarios for YOLOv8

- **High Accuracy Requirements:** Applications like [medical imaging](https://www.ultralytics.com/solutions/ai-in-healthcare) or [quality inspection](https://www.ultralytics.com/blog/quality-inspection-in-manufacturing-traditional-vs-deep-learning-methods) where detecting subtle features is critical.
- **Multi-Task Learning:** Projects requiring [segmentation](https://docs.ultralytics.com/tasks/segment/) or [pose estimation](https://docs.ultralytics.com/tasks/pose/) alongside detection.
- **Future-Proofing:** Developers starting new projects should opt for YOLOv8 (or newer) to ensure long-term support and compatibility with the latest deployment tools.

## The Future: Ultralytics YOLO26

While YOLOv5 and YOLOv8 are excellent, the field has continued to evolve. For developers seeking the absolute peak of performance in 2026, we strongly recommend **Ultralytics YOLO26**.

**Why Choose YOLO26?**
YOLO26 builds upon the legacy of v5 and v8 but introduces revolutionary changes for speed and efficiency:

- **End-to-End NMS-Free:** By removing the need for Non-Maximum Suppression (NMS), YOLO26 simplifies deployment logic and reduces inference latency, a concept pioneered in [YOLOv10](https://docs.ultralytics.com/models/yolov10/).
- **MuSGD Optimizer:** A hybrid optimizer bringing LLM training stability to vision models, ensuring faster convergence.
- **Enhanced for Edge:** With **DFL removal** and specific CPU optimizations, YOLO26 runs up to **43% faster** on CPUs compared to previous generations.
- **Superior Small Object Detection:** The new **ProgLoss** and **STAL** functions significantly improve performance on small targets, vital for drone imagery and [IoT applications](https://www.ultralytics.com/solutions/ai-in-agriculture).

[Learn more about YOLO26](https://docs.ultralytics.com/models/yolo26/){ .md-button }

## Conclusion

Both YOLOv5 and YOLOv8 represent monumental achievements in the history of computer vision. **YOLOv5** remains a reliable, low-memory workhorse for many existing applications, celebrated for its stability and lower resource footprint in training. **YOLOv8**, however, offers superior versatility, higher accuracy, and a more modern architectural design that aligns with current research trends.

For those demanding the cutting edge, looking toward **[YOLO26](https://docs.ultralytics.com/models/yolo26/)** or **[YOLO11](https://docs.ultralytics.com/models/yolo11/)** will provide even greater benefits in speed and precision. Ultimately, the robust **[Ultralytics ecosystem](https://docs.ultralytics.com/)** ensures that whichever model you choose, you have the tools, documentation, and community support to succeed.

## Code Example: Running Inference

Experience the simplicity of the Ultralytics API. This code works for YOLOv8, YOLO11, and YOLO26 models interchangeably.

```python
from ultralytics import YOLO

# Load a pretrained model (choose yolov8n.pt or yolo26n.pt)
model = YOLO("yolov8n.pt")

# Run inference on an image from the web
results = model("https://ultralytics.com/images/bus.jpg")

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
```

For more details on integrating these models into your workflow, visit our [Quickstart Guide](https://docs.ultralytics.com/quickstart/).
