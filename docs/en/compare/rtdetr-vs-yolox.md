---
comments: true
description: Compare RTDETRv2 and YOLOX object detection models. Explore architectures, performance metrics, and use cases to choose the best for your project.
keywords: RTDETRv2, YOLOX, object detection, model comparison, performance metrics, real-time detection, Ultralytics, machine learning, computer vision
---

# RTDETRv2 vs YOLOX: A Detailed Model Comparison for Object Detection

Choosing the right object detection model is crucial for the success of computer vision projects. This page provides a technical comparison between two popular models: RTDETRv2 and YOLOX, both available through Ultralytics. We will delve into their architectures, performance metrics, and ideal applications to help you make an informed decision.

<script async src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["RTDETRv2", "YOLOX"]'></canvas>

## Model Architectures

### RTDETRv2

RT-DETR (Real-Time Detection Transformer) v2 is a cutting-edge, anchor-free object detection model that leverages a Vision Transformer (ViT) backbone. It's designed for real-time performance, making it suitable for applications where low latency is critical. RTDETRv2 employs a hybrid efficient encoder and IoU-aware query selection to balance accuracy and speed. This architecture allows for efficient feature extraction and precise object localization, even in complex scenes.

[Learn more about RT-DETR](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

### YOLOX

YOLOX, standing for "You Only Look Once (X)," is an evolutionary improvement in the YOLO series, focusing on high performance with a simplified design. As an anchor-free detector, YOLOX eliminates the complexities associated with anchor boxes, leading to easier implementation and training. It incorporates a decoupled head for classification and localization tasks and the SimOTA (Simplified Optimal Transport Assignment) label assignment strategy to enhance training efficiency and accuracy.

[Learn more about YOLOX](https://github.com/Megvii-BaseDetection/YOLOX){ .md-button }

## Performance Metrics

The table below summarizes the performance metrics for various sizes of RTDETRv2 and YOLOX models. These metrics are crucial for understanding the trade-offs between model size, speed, and accuracy.

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | 54.3                 | -                              | 15.03                               | 76                 | 259               |
|            |                       |                      |                                |                                     |                    |                   |
| YOLOXnano  | 416                   | 25.8                 | -                              | -                                   | 0.91               | 1.08              |
| YOLOXtiny  | 416                   | 32.8                 | -                              | -                                   | 5.06               | 6.45              |
| YOLOXs     | 640                   | 40.5                 | -                              | 2.56                                | 9.0                | 26.8              |
| YOLOXm     | 640                   | 46.9                 | -                              | 5.43                                | 25.3               | 73.8              |
| YOLOXl     | 640                   | 49.7                 | -                              | 9.04                                | 54.2               | 155.6             |
| YOLOXx     | 640                   | 51.1                 | -                              | 16.1                                | 99.1               | 281.9             |

- **mAP<sup>val 50-95**: Mean Average Precision across IoU thresholds from 0.50 to 0.95 on the validation dataset, a key metric for object detection accuracy. [Learn more about mAP](https://www.ultralytics.com/glossary/mean-average-precision-map)
- **Speed**: Inference speed measured in milliseconds (ms) on CPU (ONNX) and NVIDIA T4 GPU (TensorRT10), indicating real-time capability. [Explore OpenVINO Latency vs Throughput Modes](https://docs.ultralytics.com/guides/optimizing-openvino-latency-vs-throughput-modes/) for optimizing inference performance.
- **Params (M)**: Number of parameters in millions, reflecting model size and complexity. [Understand Model Pruning](https://www.ultralytics.com/glossary/pruning) to optimize model size.
- **FLOPs (B)**: Floating Point Operations in billions, representing computational cost.

## Use Cases

**RTDETRv2:** Excels in scenarios demanding ultra-fast inference. Ideal for applications such as:

- **Robotics**: Real-time object detection for robot navigation and interaction. [Explore Robotics applications](https://www.ultralytics.com/glossary/robotics).
- **Autonomous Vehicles**: Perception in self-driving cars requiring minimal latency. [Learn about AI in self-driving cars](https://www.ultralytics.com/solutions/ai-in-self-driving).
- **Industrial Automation**: High-speed quality control and monitoring in manufacturing. [Discover AI in Manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing).

**YOLOX:** A versatile choice for a broad range of object detection tasks, balancing accuracy and speed effectively. Suitable for:

- **Security Systems**: Robust object detection for surveillance and security applications. [Build Security Alarm Systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8).
- **Retail Analytics**: Inventory management and customer behavior analysis in retail environments. [Explore AI for Retail Inventory Management](https://www.ultralytics.com/blog/ai-for-smarter-retail-inventory-management).
- **Wildlife Monitoring**: Detection and tracking of animals in conservation efforts. [Read about Wildlife Detection](https://www.ultralytics.com/blog/yolovme-colony-counting-smear-evaluation-and-wildlife-detection).

## Strengths and Weaknesses

**RTDETRv2 Strengths:**

- **High Inference Speed**: Optimized for real-time applications with minimal latency.
- **Anchor-Free**: Simplifies model design and training pipeline.
- **Transformer-Based**: Benefits from the global context understanding of transformer architectures. [Discover Vision Transformer (ViT)](https://www.ultralytics.com/glossary/vision-transformer-vit).

**RTDETRv2 Weaknesses:**

- **Model Size**: Larger variants may have a bigger model size compared to some YOLO models, potentially requiring more resources.

**YOLOX Strengths:**

- **Balanced Performance**: Offers a strong balance between accuracy and inference speed.
- **Simple and Efficient**: Anchor-free design and decoupled head contribute to simplicity and efficiency.
- **Adaptability**: Well-suited for various hardware platforms and application needs.

**YOLOX Weaknesses:**

- **Speed**: May be slightly slower than RTDETRv2 in extremely latency-sensitive scenarios.

## Conclusion

Both RTDETRv2 and YOLOX are powerful object detection models, each with its own strengths. RTDETRv2 is ideal when real-time performance is paramount, leveraging transformer architecture for speed. YOLOX provides a robust and versatile solution with a good balance of accuracy and speed, suitable for a wider range of applications.

For users seeking other high-performance object detectors, consider exploring [YOLOv8](https://docs.ultralytics.com/models/yolov8/) and [YOLO11](https://docs.ultralytics.com/models/yolo11/) within the Ultralytics YOLO family, as well as models like [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/). Your choice should be guided by the specific requirements of your project, balancing accuracy, speed, and resource constraints.
