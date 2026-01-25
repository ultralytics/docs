---
comments: true
description: Compare YOLOv8 and YOLO11 for object detection. Explore their performance, architecture, and best-use cases to find the right model for your needs.
keywords: YOLOv8, YOLO11, object detection, Ultralytics, YOLO comparison, machine learning, computer vision, inference speed, model accuracy
---

# Ultralytics YOLOv8 vs. YOLO11: Architectural Evolution and Performance Analysis

The evolution of object detection architectures has been rapid, with each iteration bringing significant improvements in accuracy, speed, and usability. **Ultralytics YOLOv8**, released in early 2023, set a new standard for versatility and ease of use in computer vision. In late 2024, **Ultralytics YOLO11** arrived, refining the architecture for even greater efficiency and performance across a wider range of tasks.

This comprehensive guide compares these two powerful models, analyzing their architectural differences, performance metrics, and ideal use cases to help you choose the right tool for your next [computer vision project](https://www.ultralytics.com/blog/all-you-need-to-know-about-computer-vision-tasks).

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv8", "YOLO11"]'></canvas>

## Model Overview

Before diving into the technical specifications, it is essential to understand the context and goals behind each model's development. Both are products of Ultralytics' commitment to creating accessible, state-of-the-art vision AI.

### Ultralytics YOLOv8

Released in January 2023, YOLOv8 marked a major milestone by unifying multiple tasks—detection, segmentation, classification, pose estimation, and OBB—under a single, user-friendly API. It introduced a new backbone and anchor-free detection head, making it highly versatile for diverse applications.

**Key Details:**

- **Authors:** Glenn Jocher, Ayush Chaurasia, and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com)
- **Date:** January 10, 2023
- **Docs:** [YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8/)
- **GitHub:** [Ultralytics Repository](https://github.com/ultralytics/ultralytics)

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

### Ultralytics YOLO11

Launched in September 2024, YOLO11 builds upon the solid foundation of YOLOv8. It focuses on architectural refinements to boost feature extraction efficiency and processing speed. YOLO11 is designed to offer higher accuracy with fewer parameters, making it particularly effective for real-time edge applications.

**Key Details:**

- **Authors:** Glenn Jocher and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com)
- **Date:** September 27, 2024
- **Docs:** [YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/)
- **GitHub:** [Ultralytics Repository](https://github.com/ultralytics/ultralytics)

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

!!! tip "Latest Innovation: YOLO26"

    While YOLO11 represents a significant leap over YOLOv8, developers seeking the absolute cutting edge should explore **[YOLO26](https://docs.ultralytics.com/models/yolo26/)**. Released in 2026, it introduces an NMS-free end-to-end design, MuSGD optimizer, and up to 43% faster CPU inference, setting a new benchmark for production-grade AI.

## Architectural Differences

The transition from YOLOv8 to YOLO11 involves several key architectural shifts aimed at optimizing the trade-off between computational cost and accuracy.

### Backbone and Feature Extraction

YOLOv8 utilizes a modified CSPDarknet53 backbone with C2f modules, which replaced the C3 modules from previous generations. This design improved gradient flow and feature richness.

YOLO11 enhances this further by refining the bottleneck structures and attention mechanisms within the backbone. These changes allow the model to capture more complex patterns and spatial hierarchies with reduced computational overhead. This is particularly beneficial for difficult tasks like [small object detection](https://www.ultralytics.com/blog/exploring-small-object-detection-with-ultralytics-yolo11) in aerial imagery or manufacturing quality control.

### Head Architecture

Both models employ anchor-free heads, which simplifies the training process and improves generalization across different object shapes. However, YOLO11 integrates more advanced feature fusion techniques in the neck and head, resulting in better localization precision and class separation compared to YOLOv8.

## Performance Analysis

When selecting a model for production, metrics such as Mean Average Precision (mAP), inference speed, and model size are critical. The table below provides a detailed comparison of pre-trained weights on the COCO dataset.

| Model       | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ----------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv8n     | 640                   | 37.3                 | 80.4                           | 1.47                                | 3.2                | 8.7               |
| YOLOv8s     | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m     | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l     | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x     | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |
|             |                       |                      |                                |                                     |                    |                   |
| **YOLO11n** | 640                   | **39.5**             | **56.1**                       | **1.5**                             | **2.6**            | **6.5**           |
| **YOLO11s** | 640                   | **47.0**             | **90.0**                       | **2.5**                             | **9.4**            | **21.5**          |
| **YOLO11m** | 640                   | **51.5**             | **183.2**                      | **4.7**                             | **20.1**           | **68.0**          |
| **YOLO11l** | 640                   | **53.4**             | **238.6**                      | **6.2**                             | **25.3**           | **86.9**          |
| **YOLO11x** | 640                   | **54.7**             | **462.8**                      | **11.3**                            | **56.9**           | **194.9**         |

### Key Performance Takeaways

- **Efficiency:** YOLO11 models are consistently lighter (fewer parameters) and faster (lower latency) than their YOLOv8 counterparts while achieving higher accuracy. For example, **YOLO11n** is approximately **22% faster** on CPU ONNX inference than YOLOv8n while boasting a higher mAP.
- **Compute:** The reduced [FLOPs](https://www.ultralytics.com/glossary/flops) in YOLO11 make it an excellent choice for battery-powered or resource-constrained devices, such as mobile phones or embedded IoT sensors.
- **Accuracy:** The mAP improvements in YOLO11, particularly in the smaller model variants (Nano and Small), are significant for applications requiring high reliability without heavy hardware.

## Training and Ease of Use

One of the defining strengths of the Ultralytics ecosystem is the unified and simplified user experience. Both YOLOv8 and YOLO11 share the same intuitive API, allowing developers to switch between architectures with a single line of code change.

### The Ultralytics Advantage

Unlike complex transformer models that often require massive amounts of GPU memory and intricate configuration, Ultralytics models are optimized for [training efficiency](https://docs.ultralytics.com/modes/train/). They can be trained effectively on consumer-grade GPUs, democratizing access to high-performance AI.

Features common to both models include:

- **Simple Python API:** Load, train, and deploy models in minutes.
- **Robust Documentation:** Comprehensive guides on [hyperparameter tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning/), data augmentation, and deployment.
- **Ecosystem Integration:** Seamless compatibility with the [Ultralytics Platform](https://platform.ultralytics.com) for dataset management, remote training, and one-click model export.

**Training Example:**

The following code demonstrates how easily you can switch between training YOLOv8 and YOLO11.

```python
from ultralytics import YOLO

# Load a YOLOv8 model
model_v8 = YOLO("yolov8n.pt")
# Train YOLOv8
model_v8.train(data="coco8.yaml", epochs=100, imgsz=640)

# Load a YOLO11 model - Same API!
model_11 = YOLO("yolo11n.pt")
# Train YOLO11
model_11.train(data="coco8.yaml", epochs=100, imgsz=640)
```

## Ideal Use Cases

While both models are highly capable, their specific strengths make them suitable for different scenarios.

### When to Choose YOLOv8

YOLOv8 remains a robust and reliable choice, particularly for:

- **Legacy Projects:** Existing pipelines already optimized for YOLOv8 that require stability without immediate need for architectural upgrades.
- **Broad Community Resources:** Due to its longer time in the market, YOLOv8 has an extensive library of third-party tutorials, [videos](https://www.youtube.com/ultralytics), and community implementations.
- **General Purpose Vision:** Excellent for standard object detection tasks where extreme edge optimization is not the primary constraint.

### When to Choose YOLO11

YOLO11 is the recommended choice for most new deployments, especially for:

- **Edge Computing:** Its lower parameter count and faster inference speed make it ideal for [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/), Jetson Nano, and mobile deployments.
- **Real-Time Applications:** Critical for tasks like [autonomous driving](https://www.ultralytics.com/blog/ai-in-self-driving-cars) or high-speed manufacturing lines where every millisecond of latency counts.
- **Complex Tasks:** The architectural refinements improve performance in challenging scenarios, such as [pose estimation](https://docs.ultralytics.com/tasks/pose/) for sports analytics or instance segmentation for medical imaging.

!!! note "Versatility Across Tasks"

    Both YOLOv8 and YOLO11 support a wide range of tasks beyond simple bounding box detection, including **Instance Segmentation**, **Pose Estimation**, **Oriented Bounding Box (OBB)**, and **Classification**. This versatility allows developers to solve multi-faceted problems using a single framework.

## Conclusion

Both YOLOv8 and YOLO11 represent the pinnacle of efficient computer vision. **YOLOv8** established a versatile, user-friendly standard that powered countless AI applications globally. **YOLO11** refines this legacy, offering a streamlined, faster, and more accurate architecture that pushes the boundaries of what is possible on edge devices.

For developers starting new projects today, **YOLO11** offers a superior balance of speed and accuracy. However, for those demanding the absolute latest innovations, such as end-to-end NMS-free detection and optimized loss functions, we strongly recommend exploring the newly released **[YOLO26](https://docs.ultralytics.com/models/yolo26/)**, which represents the future of real-time vision AI.

### Further Reading

- [YOLO Performance Metrics Explained](https://docs.ultralytics.com/guides/yolo-performance-metrics/)
- [Guide to Model Export (ONNX, TensorRT, CoreML)](https://docs.ultralytics.com/modes/export/)
- [Ultralytics Platform: Train and Deploy Effortlessly](https://platform.ultralytics.com)
- [Real-World Vision AI Applications](https://www.ultralytics.com/blog/60-impactful-computer-vision-applications)

### Other Models to Explore

- [YOLO26](https://docs.ultralytics.com/models/yolo26/): The latest state-of-the-art model from Ultralytics (Jan 2026) featuring NMS-free design.
- [RT-DETR](https://docs.ultralytics.com/models/rtdetr/): A transformer-based detector offering high accuracy for scenarios where speed is less critical.
- [SAM 2](https://docs.ultralytics.com/models/sam-2/): Meta's Segment Anything Model, ideal for zero-shot segmentation tasks.
