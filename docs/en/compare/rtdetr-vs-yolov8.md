---
comments: true
description: Technical comparison of RTDETRv2 and YOLOv8 object detection models by Ultralytics. Explore architecture, performance, use cases, and metrics.
keywords: RTDETRv2, YOLOv8, object detection, model comparison, Ultralytics, computer vision, AI, performance metrics, architecture
---

# RTDETRv2 vs YOLOv8: A Detailed Model Comparison

Choosing the right object detection model is crucial for computer vision projects. Ultralytics offers a range of models, and two popular choices are RTDETRv2 and YOLOv8. This page provides a detailed technical comparison to help you decide which model best fits your needs.

<script async src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["RTDETRv2", "YOLOv8"]'></canvas>

## RTDETRv2: Real-Time Detection Transformer v2

RTDETRv2 is a cutting-edge, real-time object detection model that leverages the efficiency of Vision Transformers (ViTs). It stands out due to its architecture optimized for speed without sacrificing accuracy. Built upon a hybrid efficient encoder and transformer decoder, RTDETRv2 is designed for applications requiring high-performance object detection in real-time scenarios.

**Architecture and Key Features:**

- **Real-time DETR:** Inherits the Deformable DETR (DEtection TRansformer) architecture, known for its strong performance.
- **Hybrid Efficient Encoder:** Combines convolutional layers and transformer layers to efficiently extract features.
- **Optimized for Speed:** Designed for low latency and high throughput inference.

**Performance Metrics:**
RTDETRv2 achieves impressive mAP while maintaining fast inference speeds. For detailed performance metrics, refer to the comparison table below.

**Strengths:**

- **High Accuracy:** Excels in scenarios demanding precise object detection.
- **Real-time Capability:** Offers a good balance between speed and accuracy, suitable for real-time applications.
- **Scalability:** Available in various sizes (s, m, l, x) to suit different computational resources.

**Weaknesses:**

- **Computational Cost:** Transformer-based models can be more computationally intensive compared to some other architectures, especially for the larger variants.
- **Inference Speed:** While optimized for real-time, its inference speed might be slower than the fastest YOLO models, particularly on CPU.

**Ideal Use Cases:**

- **Autonomous Driving:** Critical for accurate and fast detection of vehicles, pedestrians, and other road users.
- **Robotics:** Enables robots to perceive and interact with their environment in real-time.
- **Advanced Security Systems:** For applications requiring high precision in object detection for surveillance and monitoring.

[Learn more about RTDETR](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

## YOLOv8: State-of-the-Art Object Detector

Ultralytics YOLOv8 is the latest iteration in the YOLO (You Only Look Once) series, renowned for its exceptional speed and efficiency in object detection. YOLOv8 is a versatile and user-friendly model, designed for a wide range of object detection, image segmentation and image classification tasks. It boasts state-of-the-art performance across various metrics, making it a go-to choice for both research and industry applications.

**Architecture and Key Features:**

- **One-Stage Detector:** Maintains the hallmark YOLO architecture for fast, single-pass detection.
- **Anchor-Free Approach:** Simplifies design and improves generalization.
- **Versatile Model Family:** Offers a range of model sizes (n, s, m, l, x) to optimize for different speed and accuracy requirements.

**Performance Metrics:**
YOLOv8 is known for its rapid inference speed and strong mAP across different model sizes. Check the comparison table for specific metrics.

**Strengths:**

- **Exceptional Speed:** One of the fastest object detection models available, ideal for real-time processing.
- **Efficiency:** Operates with high efficiency, making it suitable for deployment on various hardware, including edge devices.
- **Ease of Use:** Simple to train, deploy, and integrate into existing systems with Ultralytics HUB and Python package.
- **Flexibility:** Supports object detection, segmentation, classification, and pose estimation tasks.

**Weaknesses:**

- **Accuracy Trade-off:** In some complex scenarios, particularly with small object detection, YOLOv8 might have slightly lower accuracy compared to more computationally intensive models like RTDETRv2, depending on the specific model size chosen.

**Ideal Use Cases:**

- **Real-time Surveillance:** Efficiently processes video feeds for object detection in security and monitoring applications.
- **Industrial Automation:** Fast and reliable detection for quality control and process automation in manufacturing.
- **Mobile and Edge Deployments:** Model variants like YOLOv8n and YOLOv8s are optimized for resource-constrained environments such as mobile devices and edge computing platforms like NVIDIA Jetson and Raspberry Pi.
- **Webcam based applications:** For real-time object detection with webcams, achieving high frames per second.

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## Model Comparison Table

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | 54.3                 | -                              | 15.03                               | 76                 | 259               |
|            |                       |                      |                                |                                     |                    |                   |
| YOLOv8n    | 640                   | 37.3                 | 80.4                           | 1.47                                | 3.2                | 8.7               |
| YOLOv8s    | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m    | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l    | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x    | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |

## Conclusion

Both RTDETRv2 and YOLOv8 are powerful object detection models, each with unique strengths. RTDETRv2 excels in accuracy for real-time applications, making it suitable for scenarios where precision is paramount. YOLOv8, on the other hand, prioritizes speed and efficiency, making it ideal for applications where rapid inference and resource optimization are key.

Your choice between RTDETRv2 and YOLOv8 should depend on the specific requirements of your project, balancing accuracy, speed, and computational resources. For further exploration, consider also reviewing other Ultralytics models like [YOLOv11](https://docs.ultralytics.com/models/yolo11/), [YOLOv9](https://docs.ultralytics.com/models/yolov9/), [YOLOv7](https://docs.ultralytics.com/models/yolov7/), [YOLOv6](https://docs.ultralytics.com/models/yolov6/), [YOLOv5](https://docs.ultralytics.com/models/yolov5/), [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/), and [YOLO-World](https://docs.ultralytics.com/models/yolo-world/) to find the perfect fit for your computer vision tasks.

For practical guidance and troubleshooting tips, refer to our [YOLO guides](https://docs.ultralytics.com/guides/) and explore solutions for [common YOLO issues](https://docs.ultralytics.com/guides/yolo-common-issues/). You can also learn more about [YOLO performance metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/) to understand how to evaluate your models effectively.
