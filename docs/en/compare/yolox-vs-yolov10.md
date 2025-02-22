---
description: Compare YOLOv10 and YOLOX for object detection. Explore architecture, benchmarks, and use cases to choose the best real-time detection model for your needs.
keywords: YOLOv10, YOLOX, object detection, Ultralytics, real-time, model comparison, benchmark, computer vision, deep learning, AI
---

# Technical Comparison: YOLOv10 vs YOLOX for Object Detection

Ultralytics YOLO models are at the forefront of real-time object detection, known for their speed and accuracy. This page provides a technical comparison between two prominent models: **YOLOv10** and **YOLOX**. We will explore their architectural designs, performance benchmarks, training approaches, and ideal applications to guide you in selecting the best model for your computer vision needs.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOX", "YOLOv10"]'></canvas>

## YOLOv10: The Cutting-Edge Real-Time Detector

[Ultralytics YOLOv10](https://docs.ultralytics.com/models/yolov10/) is the latest iteration in real-time object detection, prioritizing exceptional speed and efficiency. It is particularly well-suited for applications where minimal latency is crucial.

### Architecture and Key Features

YOLOv10 refines the anchor-free detection approach, leading to a more streamlined architecture and reduced computational demands. Key architectural features include:

- **Efficient Backbone and Neck:** Optimized for rapid feature extraction using fewer parameters and FLOPs, ensuring quick processing.
- **NMS-Free Approach:** Bypasses the Non-Maximum Suppression (NMS) post-processing step, further accelerating inference.
- **Scalable Model Variants:** Offers a range of model sizes (n, s, m, b, l, x) to accommodate varying computational resources and accuracy needs.

### Performance Metrics

YOLOv10 excels in speed while maintaining a strong balance with accuracy. Refer to the comparison table for detailed performance metrics.

### Use Cases

- **Edge Devices:** Optimized for deployment on devices with limited resources like smartphones, embedded systems, and IoT devices due to its compact size and fast inference.
- **Real-time Applications:** Ideal for scenarios requiring immediate object detection, such as autonomous driving, robotics, and real-time video analytics.
- **High-Speed Processing:** Excels in applications where rapid processing is essential, including high-throughput industrial inspection and fast-paced surveillance systems.

### Strengths and Weaknesses

**Strengths:**

- **Inference Speed:** Engineered for extremely fast inference, vital for real-time applications.
- **Model Size:** Small model sizes, especially the YOLOv10n and YOLOv10s variants, are perfect for edge deployment with limited resources.
- **Efficiency:** Delivers high performance relative to computational cost, resulting in energy efficiency.

**Weaknesses:**

- **mAP:** While efficient, larger models like YOLOX-x might achieve slightly higher mAP in certain scenarios where accuracy is prioritized over speed.

[Learn more about YOLOv10](https://docs.ultralytics.com/models/yolov10/){ .md-button }

## YOLOX: High-Performance Anchor-Free Detector

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), introduced by Megvii in July 2021 ([YOLOX Arxiv](https://arxiv.org/abs/2107.08430)), is a high-performance anchor-free object detector focused on simplicity and efficacy. It has become a widely adopted model in the computer vision community.

### Architecture and Key Features

YOLOX adopts an anchor-free approach, simplifying the detection process and improving overall performance. Key architectural features include:

- **Anchor-Free Detection:** Removes the necessity for predefined anchors, reducing complexity and enhancing generalization.
- **Decoupled Head:** Separates classification and localization heads to improve learning and detection accuracy.
- **Advanced Training Techniques:** Incorporates techniques such as SimOTA label assignment and robust data augmentation for more effective training.

### Performance Metrics

YOLOX models provide a robust balance of accuracy and speed. As shown in the table, YOLOX models achieve competitive mAP scores while maintaining reasonable inference speeds.

### Use Cases

- **General Object Detection:** Well-suited for a broad spectrum of object detection tasks where a balance between accuracy and speed is needed.
- **Research and Development:** Favored in research settings due to its strong performance and well-documented implementation ([YOLOX GitHub](https://github.com/Megvii-BaseDetection/YOLOX)).
- **Industrial Applications:** Applicable in diverse industrial contexts requiring reliable and accurate object detection.

### Strengths and Weaknesses

**Strengths:**

- **Accuracy:** Achieves high mAP scores, particularly with larger models like YOLOX-x, demonstrating strong detection capabilities.
- **Established Model:** A recognized and validated model with extensive community support and resources ([YOLOX Docs](https://yolox.readthedocs.io/en/latest/)).
- **Versatility:** Performs effectively across various object detection tasks and datasets.

**Weaknesses:**

- **Inference Speed (vs. YOLOv10):** While fast, YOLOX might not reach the extreme inference speeds of the most optimized YOLOv10 variants, especially the 'n' and 's' models.
- **Model Size (vs. YOLOv10n):** Larger YOLOX models (x, l) have a significantly larger parameter count and FLOPs compared to the smallest YOLOv10 models.

[Learn more about YOLOX](https://github.com/Megvii-BaseDetection/YOLOX){ .md-button }

## Model Comparison Table

| Model     | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| --------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOXnano | 416                   | 25.8                 | -                              | -                                   | 0.91               | 1.08              |
| YOLOXtiny | 416                   | 32.8                 | -                              | -                                   | 5.06               | 6.45              |
| YOLOXs    | 640                   | 40.5                 | -                              | 2.56                                | 9.0                | 26.8              |
| YOLOXm    | 640                   | 46.9                 | -                              | 5.43                                | 25.3               | 73.8              |
| YOLOXl    | 640                   | 49.7                 | -                              | 9.04                                | 54.2               | 155.6             |
| YOLOXx    | 640                   | 51.1                 | -                              | 16.1                                | 99.1               | 281.9             |
|           |                       |                      |                                |                                     |                    |                   |
| YOLOv10n  | 640                   | 39.5                 | -                              | 1.56                                | 2.3                | 6.7               |
| YOLOv10s  | 640                   | 46.7                 | -                              | 2.66                                | 7.2                | 21.6              |
| YOLOv10m  | 640                   | 51.3                 | -                              | 5.48                                | 15.4               | 59.1              |
| YOLOv10b  | 640                   | 52.7                 | -                              | 6.54                                | 24.4               | 92.0              |
| YOLOv10l  | 640                   | 53.3                 | -                              | 8.33                                | 29.5               | 120.3             |
| YOLOv10x  | 640                   | 54.4                 | -                              | 12.2                                | 56.9               | 160.4             |

For users interested in exploring other models, Ultralytics also offers a range of YOLO models including [YOLOv8](https://docs.ultralytics.com/models/yolov8/), [YOLOv9](https://docs.ultralytics.com/models/yolov9/), and [YOLO11](https://docs.ultralytics.com/models/yolo11/), as well as comparisons against other architectures like [PP-YOLOE](https://docs.ultralytics.com/compare/pp-yoloe-vs-yolov10/).
