---
comments: true
description: Compare YOLOX and YOLOv9 for object detection. Explore performance, architecture, and use cases to choose the best model for your vision tasks.
keywords: YOLOX, YOLOv9, object detection, model comparison, computer vision, AI models, deep learning, performance benchmarks, architecture, real-time detection
---

# YOLOX vs. YOLOv9: Evolution of High-Performance Object Detection

In the rapidly advancing field of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv), selecting the right object detection model is critical for balancing accuracy, speed, and deployment complexity. This comparison explores two significant milestones in the YOLO family: **YOLOX**, a robust anchor-free detector released in 2021, and **YOLOv9**, a 2024 architecture introducing Programmable Gradient Information (PGI) for superior feature retention.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOX", "YOLOv9"]'></canvas>

## YOLOX: The Anchor-Free Pioneer

**YOLOX** represented a major shift in the YOLO series by moving away from anchor-based mechanisms to an **anchor-free** design. This simplification eliminated the need for manual anchor box tuning, making the model more adaptable to diverse datasets and aspect ratios. By incorporating a decoupled head and the advanced [SimOTA](https://arxiv.org/abs/2107.08430) label assignment strategy, YOLOX achieved state-of-the-art results upon its release, bridging the gap between academic research and industrial application.

- **Authors:** Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun
- **Organization:** [Megvii](https://www.megvii.com/)
- **Date:** 2021-07-18
- **Arxiv:** [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430)
- **GitHub:** [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

[Learn more about YOLOX](https://docs.ultralytics.com/compare/yolox-vs-damo-yolo/){ .md-button }

### Key Architectural Features

- **Anchor-Free Mechanism:** Removes the complexity of anchor box clustering, reducing the number of design parameters and improving generalization.
- **Decoupled Head:** Separates the classification and regression tasks into different branches, resolving the conflict between these two objectives and improving convergence speed.
- **SimOTA Label Assignment:** A dynamic label assignment strategy that views the training process as an optimal transport problem, assigning ground truths to predictions more effectively than static IoU thresholds.

## YOLOv9: Programmable Gradients for Deep Learning

**YOLOv9** tackles the fundamental issue of information loss in deep neural networks. As networks become deeper, essential feature information can vanish during forward propagation. YOLOv9 introduces **Programmable Gradient Information (PGI)** and the **Generalized Efficient Layer Aggregation Network (GELAN)** to preserve critical data throughout the network layers. This results in significant improvements in detection accuracy, particularly for lightweight models, while maintaining high efficiency.

- **Authors:** Chien-Yao Wang and Hong-Yuan Mark Liao
- **Organization:** Institute of Information Science, Academia Sinica
- **Date:** 2024-02-21
- **Arxiv:** [YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)
- **GitHub:** [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)
- **Docs:** [Ultralytics YOLOv9 Documentation](https://docs.ultralytics.com/models/yolov9/)

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

### Key Architectural Features

- **GELAN Architecture:** Combines CSPNet and ELAN design principles to maximize parameter efficiency and computational speed, allowing the model to run effectively on various hardware.
- **Programmable Gradient Information (PGI):** An auxiliary supervision framework that generates reliable gradients for updating network weights, ensuring the main branch learns complete features even in very deep architectures.
- **Reversible Functions:** Mitigates the information bottleneck problem by ensuring data can be reconstructed effectively, preserving semantic information across layers.

## Performance Comparison

When evaluating these models, **YOLOv9** generally outperforms YOLOX in terms of accuracy-to-parameter ratio. While YOLOX-x achieves a respectable **51.1% mAP**, the newer YOLOv9c surpasses it with **53.0% mAP** while using significantly fewer parameters (25.3M vs 99.1M) and less computational power. This efficiency makes YOLOv9 a stronger candidate for real-time applications where hardware resources are constrained but high accuracy is required.

However, **YOLOX** remains highly relevant for legacy edge devices. Its simpler anchor-free design can sometimes be easier to optimize for specific mobile chipsets or NPU architectures that may not fully support the complex layer aggregations found in newer models like GELAN.

### Detailed Metrics

| Model     | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| --------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOXnano | 416                   | 25.8                 | -                              | -                                   | 0.91               | 1.08              |
| YOLOXtiny | 416                   | 32.8                 | -                              | -                                   | 5.06               | 6.45              |
| YOLOXs    | 640                   | 40.5                 | -                              | 2.56                                | 9.0                | 26.8              |
| YOLOXm    | 640                   | 46.9                 | -                              | 5.43                                | 25.3               | 73.8              |
| YOLOXl    | 640                   | 49.7                 | -                              | 9.04                                | 54.2               | 155.6             |
| YOLOXx    | 640                   | 51.1                 | -                              | 16.1                                | 99.1               | 281.9             |
|           |                       |                      |                                |                                     |                    |                   |
| YOLOv9t   | 640                   | 38.3                 | -                              | **2.3**                             | 2.0                | 7.7               |
| YOLOv9s   | 640                   | 46.8                 | -                              | 3.54                                | **7.1**            | **26.4**          |
| YOLOv9m   | 640                   | 51.4                 | -                              | 6.43                                | **20.0**           | 76.3              |
| YOLOv9c   | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e   | 640                   | **55.6**             | -                              | 16.77                               | 57.3               | 189.0             |

!!! info "Efficiency Highlight"

    Notice that **YOLOv9c** achieves higher accuracy (53.0% mAP) than the largest **YOLOX-x** (51.1% mAP) while using roughly **75% fewer parameters**. This demonstrates the rapid advancement in architectural efficiency over the three years between these releases.

## Training and Ease of Use with Ultralytics

A critical differentiator for developers is the ecosystem surrounding the model. **YOLOv9** is fully integrated into the Ultralytics ecosystem, providing a significant advantage in usability.

### The Ultralytics Advantage

Using the Ultralytics Python API allows you to access state-of-the-art models with unified syntax. You do not need to clone complex repositories or manually compile C++ operators, which is often a hurdle with original research implementations like YOLOX.

```python
from ultralytics import YOLO

# Load a model (YOLOv9c or the new YOLO26s)
model = YOLO("yolov9c.pt")

# Train on custom data in one line
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Validate performance
metrics = model.val()
```

This integration provides:

1.  **Streamlined Workflow:** Seamlessly switch between [detection](https://docs.ultralytics.com/tasks/detect/), [segmentation](https://docs.ultralytics.com/tasks/segment/), and [pose estimation](https://docs.ultralytics.com/tasks/pose/) tasks.
2.  **Memory Efficiency:** Ultralytics training pipelines are optimized for consumer hardware, often requiring less [GPU memory](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit) than transformer-based alternatives or unoptimized research codebases.
3.  **Deployment Readiness:** Built-in export functions allow you to convert trained models to [ONNX](https://docs.ultralytics.com/integrations/onnx/), [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/), CoreML, and TFLite with a single command.

## Real-World Applications

Choosing between these models depends on your specific deployment constraints.

### High-Speed Retail Analytics

For retail environments requiring real-time product recognition on edge devices, **YOLOv9** is often the superior choice. Its [GELAN architecture](https://docs.ultralytics.com/models/yolov9/) allows for high throughput on devices like the NVIDIA Jetson Orin Nano, enabling features like automated checkout or shelf stocking analysis without significant lag.

### Legacy Mobile Deployment

In scenarios involving older mobile hardware or specific NPU architectures that favor simple convolution patterns, **YOLOX-Nano** or **YOLOX-Tiny** might still be preferred. Their pure anchor-free design without complex aggregation blocks can sometimes be easier to quantize and deploy on very restricted [microcontrollers](https://www.ultralytics.com/glossary/edge-computing) or legacy Android devices.

### Autonomous Robotics

For robotics applications where maximizing [accuracy](https://www.ultralytics.com/glossary/accuracy) is paramount to avoid collisions, the superior feature retention of **YOLOv9e** provides a safety margin that older models cannot match. The PGI framework ensures that small obstacles are not lost in the feature extraction process, which is critical for navigation in cluttered environments.

## The Future: Enter YOLO26

While YOLOv9 offers exceptional performance, the field of AI never stands still. The newly released **YOLO26** builds upon these foundations to offer the ultimate balance of speed and precision.

**YOLO26** introduces a native **end-to-end NMS-free design**, completely eliminating the need for Non-Maximum Suppression during inference. This results in significantly simpler deployment pipelines and faster execution speeds. Furthermore, by removing Distribution Focal Loss (DFL) and utilizing the novel **MuSGD optimizer** (a hybrid of SGD and Muon), YOLO26 achieves up to **43% faster CPU inference** compared to previous generations, making it the ideal choice for modern edge computing.

For developers looking for the absolute best in class, we recommend evaluating [YOLO26](https://docs.ultralytics.com/models/yolo26/) for your next project to leverage these cutting-edge advancements in [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv).

### Similar Models to Explore

- [YOLO11](https://docs.ultralytics.com/models/yolo11/): A powerful predecessor to YOLO26, offering excellent versatility across various vision tasks.
- [RT-DETR](https://docs.ultralytics.com/models/rtdetr/): A transformer-based detector that also eliminates NMS, ideal for scenarios where high accuracy is prioritized over pure inference speed.
- [YOLOv10](https://docs.ultralytics.com/models/yolov10/): The first YOLO model to introduce the NMS-free training paradigm, serving as a bridge to the modern YOLO26 architecture.
