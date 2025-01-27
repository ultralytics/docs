---
comments: true
description: Discover a detailed comparison of YOLOv9 and YOLO11 object detection models. Explore architecture, benchmarks, and use cases to choose the right model.
keywords: YOLOv9, YOLO11, object detection, model comparison, Ultralytics, real-time AI, machine learning, computer vision, edge AI, benchmarks, performance metrics
---

# YOLOv9 vs YOLO11: A Detailed Comparison for Object Detection

Ultralytics YOLO models are renowned for their real-time object detection capabilities, continually evolving to meet diverse application needs. This page provides a technical comparison between two cutting-edge models: YOLOv9 and YOLO11, highlighting their architectural nuances, performance benchmarks, and suitability for different use cases.

<script async src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv9", "YOLO11"]'></canvas>

## Ultralytics YOLO11: Refined Efficiency and Accuracy

[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) represents the latest iteration in the YOLO series, building upon the strengths of its predecessors like [YOLOv8](https://docs.ultralytics.com/models/yolov8/). YOLO11 focuses on enhancing both efficiency and accuracy, making it an excellent choice for real-time applications across various platforms, from edge devices to cloud systems.

**Architecture and Key Features:**
YOLO11 maintains an anchor-free detection approach, simplifying the model architecture and improving generalization. Key improvements include refined network layers for more efficient feature extraction and a streamlined design that reduces computational overhead. This allows YOLO11 to achieve comparable or better accuracy than previous models with fewer parameters.

**Performance Metrics:**
YOLO11 demonstrates a strong balance between speed and accuracy. As shown in the comparison table, YOLO11 models achieve competitive mAP scores with impressive inference speeds, especially when utilizing hardware acceleration like TensorRT. The model sizes remain relatively compact, facilitating deployment on resource-constrained devices. Refer to [YOLO Performance Metrics guide](https://docs.ultralytics.com/guides/yolo-performance-metrics/) for a deeper understanding of these metrics.

**Use Cases:**
Ideal use cases for YOLO11 include applications requiring high accuracy and real-time performance, such as:

- **Security and Surveillance:** [Computer vision for theft prevention](https://www.ultralytics.com/blog/computer-vision-for-theft-prevention-enhancing-security), [security alarm systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8).
- **Robotics:** [Integrating YOLO with ROS](https://docs.ultralytics.com/guides/ros-quickstart/) for real-time object perception in robotic systems.
- **Industrial Automation:** [Quality inspection in manufacturing](https://www.ultralytics.com/blog/quality-inspection-in-manufacturing-traditional-vs-deep-learning-methods), [conveyor automation](https://www.ultralytics.com/blog/yolo11-enhancing-efficiency-conveyor-automation).
- **Edge AI Deployment:** [Running YOLO models on Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/), [NVIDIA Jetson devices](https://docs.ultralytics.com/guides/nvidia-jetson/).

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

## Ultralytics YOLOv9: High Accuracy with Advanced Techniques

[Ultralytics YOLOv9](https://docs.ultralytics.com/models/yolov9/), while not as recent as YOLO11, introduces innovative techniques to achieve state-of-the-art accuracy in object detection. It focuses on enhancing information preservation and leveraging novel architectural components for improved performance.

**Architecture and Key Features:**
YOLOv9 incorporates techniques like Programmable Gradient Information (PGI) and Generalized Efficient Layer Aggregation Network (GELAN) to address information loss during network propagation. PGI helps the model learn more reliable feature representations, while GELAN optimizes network efficiency without compromising accuracy. These advancements make YOLOv9 particularly effective in scenarios demanding the highest possible precision.

**Performance Metrics:**
YOLOv9 models are designed to push the boundaries of accuracy. As reflected in the comparison table, YOLOv9 achieves impressive mAP scores, positioning it among the top-performing object detection models. However, this emphasis on accuracy may come with a trade-off in inference speed and model size compared to more efficiency-focused models like YOLO11.

**Use Cases:**
YOLOv9 is best suited for applications where accuracy is paramount, even if it requires more computational resources. These use cases include:

- **High-Precision Object Detection:** Scenarios requiring meticulous object recognition, such as [medical image analysis](https://www.ultralytics.com/glossary/medical-image-analysis) like [tumor detection](https://www.ultralytics.com/blog/using-yolo11-for-tumor-detection-in-medical-imaging).
- **Detailed Scene Understanding:** Applications needing fine-grained object detection in complex environments.
- **Research and Development:** Exploring the limits of object detection accuracy and pushing model performance.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

## Model Comparison Table

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv9t | 640                   | 38.3                 | -                              | 2.3                                 | 2.0                | 7.7               |
| YOLOv9s | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e | 640                   | 55.6                 | -                              | 16.77                               | 57.3               | 189.0             |
|         |                       |                      |                                |                                     |                    |                   |
| YOLO11n | 640                   | 39.5                 | 56.1                           | 1.5                                 | 2.6                | 6.5               |
| YOLO11s | 640                   | 47.0                 | 90.0                           | 2.5                                 | 9.4                | 21.5              |
| YOLO11m | 640                   | 51.5                 | 183.2                          | 4.7                                 | 20.1               | 68.0              |
| YOLO11l | 640                   | 53.4                 | 238.6                          | 6.2                                 | 25.3               | 86.9              |
| YOLO11x | 640                   | 54.7                 | 462.8                          | 11.3                                | 56.9               | 194.9             |

## Choosing the Right Model

The choice between YOLOv9 and YOLO11 depends on the specific application requirements. If real-time performance and efficiency are critical, particularly on edge devices, YOLO11 is often the preferred choice. For scenarios demanding the highest possible object detection accuracy, and where computational resources are less constrained, YOLOv9 offers superior performance.

Users may also be interested in exploring other models in the YOLO family, such as the widely adopted [YOLOv8](https://docs.ultralytics.com/models/yolov8/) for a balance of performance and versatility, or [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/) for a Neural Architecture Search optimized model.

Refer to the [Ultralytics Docs](https://docs.ultralytics.com/guides/) and [GitHub repository](https://github.com/ultralytics/ultralytics) for more detailed information, tutorials, and guides on all YOLO models.
