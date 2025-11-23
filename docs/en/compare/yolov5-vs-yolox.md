---
comments: true
description: Compare YOLOv5 and YOLOX object detection models. Explore performance metrics, strengths, weaknesses, and use cases to choose the best fit for your needs.
keywords: YOLOv5, YOLOX, object detection, model comparison, computer vision, Ultralytics, anchor-based, anchor-free, real-time detection, AI models
---

# YOLOv5 vs YOLOX: Architectural Shifts and Performance Metrics

The landscape of [object detection](https://docs.ultralytics.com/tasks/detect/) has evolved rapidly, with various architectures vying for the optimal balance between inference speed and detection accuracy. Two significant milestones in this journey are **YOLOv5**, developed by Ultralytics, and **YOLOX**, a research-focused model from Megvii. While both models stem from the "You Only Look Once" lineage, they diverge significantly in their architectural philosophies—specifically regarding anchor-based versus anchor-free detection mechanisms.

This comparison explores the technical specifications, architectural differences, and performance metrics of both models to help developers and researchers choose the right tool for their [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) projects.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv5", "YOLOX"]'></canvas>

## Ultralytics YOLOv5: The Engineering Standard

Released in 2020, YOLOv5 quickly became the industry standard for practical object detection. Unlike its predecessors, which were primarily academic research projects, YOLOv5 was engineered with a focus on usability, deployment ease, and real-world performance. It introduced a streamlined PyTorch-based workflow that made training and deploying custom models accessible to a broader audience.

- **Authors:** Glenn Jocher
- **Organization:** [Ultralytics](https://www.ultralytics.com)
- **Date:** 2020-06-26
- **GitHub:** [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- **Docs:** [https://docs.ultralytics.com/models/yolov5/](https://docs.ultralytics.com/models/yolov5/)

YOLOv5 employs an **anchor-based architecture**, utilizing pre-defined [anchor boxes](https://www.ultralytics.com/glossary/anchor-boxes) to predict object locations. It integrates an "AutoAnchor" feature that evolves anchor shapes to fit custom datasets before training, ensuring optimal convergence. The model features a [CSPNet](https://www.ultralytics.com/glossary/backbone) backbone and a PANet neck, optimized for rapid feature extraction and aggregation. Its primary strength lies in its **exceptional inference speed** and low memory footprint, making it ideal for edge computing and mobile applications.

[Learn more about YOLOv5](https://docs.ultralytics.com/models/yolov5/){ .md-button }

## YOLOX: The Anchor-Free Contender

YOLOX, released in 2021 by Megvii, sought to push the boundaries of the YOLO family by adopting an **anchor-free** design. This approach eliminates the need for pre-defined anchor boxes, instead predicting object centers and sizes directly. This shift was aimed at simplifying the design process and improving generalization across diverse object shapes.

- **Authors:** Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun
- **Organization:** [Megvii](https://www.megvii.com/)
- **Date:** 2021-07-18
- **Arxiv:** [https://arxiv.org/abs/2107.08430](https://arxiv.org/abs/2107.08430)
- **GitHub:** [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- **Docs:** [https://yolox.readthedocs.io/en/latest/](https://yolox.readthedocs.io/en/latest/)

YOLOX introduces a **decoupled head** architecture, separating the classification and regression tasks into different branches. This theoretically allows the model to learn distinct feature representations for identifying _what_ an object is versus _where_ it is. Additionally, it employs an advanced label assignment strategy known as **SimOTA** (Simplified Optimal Transport Assignment) to dynamically assign positive samples during training. While these innovations contribute to high accuracy, they often come with increased computational complexity.

[Learn more about YOLOX](https://yolox.readthedocs.io/en/latest/){ .md-button }

!!! tip "Looking for the Latest Technology?"

    While YOLOv5 and YOLOX represent significant steps in computer vision history, the field moves fast. **[YOLO11](https://docs.ultralytics.com/models/yolo11/)**, the latest model from Ultralytics, offers superior accuracy and speed compared to both, featuring a refined architecture that supports detection, [segmentation](https://docs.ultralytics.com/tasks/segment/), pose estimation, and more.

## Performance Analysis: Speed vs. Accuracy

When comparing YOLOv5 and YOLOX, the trade-off usually centers on inference latency versus absolute [precision](https://www.ultralytics.com/glossary/precision). YOLOv5 is meticulously optimized for speed, particularly on hardware accelerators using [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/) and ONNX Runtime. As shown in the data below, YOLOv5 models demonstrate significantly lower latency (higher speed) across equivalent model sizes.

| Model     | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| --------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv5n   | 640                   | 28.0                 | **73.6**                       | **1.12**                            | 2.6                | 7.7               |
| YOLOv5s   | 640                   | 37.4                 | 120.7                          | 1.92                                | 9.1                | 24.0              |
| YOLOv5m   | 640                   | 45.4                 | 233.9                          | 4.03                                | 25.1               | 64.2              |
| YOLOv5l   | 640                   | 49.0                 | 408.4                          | 6.61                                | 53.2               | 135.0             |
| YOLOv5x   | 640                   | 50.7                 | 763.2                          | 11.89                               | **97.2**           | **246.4**         |
|           |                       |                      |                                |                                     |                    |                   |
| YOLOXnano | 416                   | 25.8                 | -                              | -                                   | **0.91**           | **1.08**          |
| YOLOXtiny | 416                   | 32.8                 | -                              | -                                   | 5.06               | 6.45              |
| YOLOXs    | 640                   | 40.5                 | -                              | 2.56                                | 9.0                | 26.8              |
| YOLOXm    | 640                   | 46.9                 | -                              | 5.43                                | 25.3               | 73.8              |
| YOLOXl    | 640                   | 49.7                 | -                              | 9.04                                | 54.2               | 155.6             |
| YOLOXx    | 640                   | **51.1**             | -                              | 16.1                                | 99.1               | 281.9             |

### Key Takeaways

- **Inference Speed:** YOLOv5 holds a decisive advantage in speed. For instance, **YOLOv5n** achieves a TensorRT latency of just **1.12 ms**, making it exceptionally suitable for high-FPS video processing on edge devices like the [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/). In contrast, the smallest YOLOX models lack comparable benchmark data for CPU, and their GPU latency is generally higher for similar accuracy tiers.
- **Accuracy (mAP):** YOLOX tends to achieve slightly higher [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) scores on the COCO dataset, particularly with its larger variants (YOLOX-x at 51.1 vs YOLOv5x at 50.7). This is attributed to its anchor-free design and decoupled head, which can better handle object variations. However, this marginal gain often comes at the cost of significantly higher computational overhead (FLOPs).
- **Efficiency:** YOLOv5 models generally require fewer [FLOPs](https://www.ultralytics.com/glossary/flops) for a given inference speed. The coupled head design of YOLOv5 is more hardware-friendly, allowing for faster execution on both CPUs and GPUs.

## Architectural Deep Dive

The fundamental difference lies in how each model approaches the detection problem.

**YOLOv5 (Anchor-Based):**
YOLOv5 utilizes a predefined set of anchor boxes. During training, the model learns to adjust these boxes to fit the objects. This method relies on the correlation between the object's size and the grid cell size.

- **Pros:** Stable training, established methodology, excellent performance on standard datasets.
- **Cons:** Requires [hyperparameter tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning/) for anchors on exotic datasets (though YOLOv5's AutoAnchor mitigates this).

**YOLOX (Anchor-Free):**
YOLOX treats object detection as a point regression problem. It predicts the distance from the center of the grid cell to the object's boundaries.

- **Pros:** Reduces the number of design parameters (no anchors to tune), potential for better generalization on irregular aspect ratios.
- **Cons:** Can be slower to converge during training, and the decoupled head adds layers that increase [inference latency](https://www.ultralytics.com/glossary/inference-latency).

## User Experience and Ecosystem

One of the most defining characteristics of **Ultralytics YOLOv5** is its robust ecosystem. While YOLOX provides a strong academic baseline, YOLOv5 offers a product-ready framework designed for developers.

### Ease of Use

YOLOv5 is renowned for its "start-to-finish" simplicity. From [data annotation](https://docs.ultralytics.com/guides/data-collection-and-annotation/) to model training and deployment, the Ultralytics ecosystem streamlines every step. The model can be loaded with a few lines of code, and it supports automatic export to formats like [TFLite](https://docs.ultralytics.com/integrations/tflite/), [CoreML](https://docs.ultralytics.com/integrations/coreml/), and [ONNX](https://docs.ultralytics.com/integrations/onnx/).

```python
import torch

# Load a pretrained YOLOv5s model from PyTorch Hub
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Run inference on an image
results = model("https://ultralytics.com/images/zidane.jpg")

# Print results
results.print()
```

### Versatility and Maintenance

Ultralytics models are not just about detection. The framework supports [image classification](https://docs.ultralytics.com/tasks/classify/) and [instance segmentation](https://docs.ultralytics.com/tasks/segment/), offering a unified API for multiple tasks. This versatility is often lacking in research-specific repositories like YOLOX, which primarily focus on detection. Furthermore, the active maintenance by Ultralytics ensures compatibility with the latest versions of PyTorch and CUDA, reducing "code rot" over time.

## Ideal Use Cases

- **Choose Ultralytics YOLOv5 if:**
  - You need **real-time performance** on edge devices (Raspberry Pi, mobile phones).
  - You prioritize **ease of deployment** and need built-in support for exporting to TensorRT, CoreML, or TFLite.
  - You prefer a **stable, well-documented** framework with active community support.
  - Your application involves [security surveillance](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8) or autonomous navigation where low latency is critical.

- **Choose YOLOX if:**
  - You are conducting academic research specifically on **anchor-free** architectures.
  - You require the absolute maximum mAP for a competition or benchmark, regardless of inference speed.
  - You have a specialized dataset where anchor-based methods have demonstrably failed (e.g., extreme aspect ratios), and AutoAnchor did not resolve the issue.

## Conclusion

Both YOLOv5 and YOLOX have earned their places in the history of computer vision. **YOLOX** demonstrated the viability of anchor-free detectors in the YOLO family, offering a strong baseline for academic research. However, for the vast majority of practical applications, **Ultralytics YOLOv5** remains the superior choice due to its unmatched speed, efficiency, and developer-friendly ecosystem.

For those starting new projects today, we highly recommend exploring **[YOLO11](https://docs.ultralytics.com/models/yolo11/)**. It builds upon the strengths of YOLOv5—ease of use and speed—while integrating modern architectural advancements that surpass both YOLOv5 and YOLOX in accuracy and versatility.

## Other Model Comparisons

Explore how Ultralytics models compare against other architectures in the field:

- [YOLOv5 vs YOLOv8](https://docs.ultralytics.com/compare/yolov8-vs-yolov5/)
- [YOLO11 vs YOLOX](https://docs.ultralytics.com/compare/yolo11-vs-yolox/)
- [RT-DETR vs YOLOv5](https://docs.ultralytics.com/compare/rtdetr-vs-yolov5/)
- [YOLOv8 vs YOLOX](https://docs.ultralytics.com/compare/yolov8-vs-yolox/)
- [YOLOv10 vs YOLOX](https://docs.ultralytics.com/compare/yolov10-vs-yolox/)
