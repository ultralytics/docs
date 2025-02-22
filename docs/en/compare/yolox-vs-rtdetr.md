---
comments: true
description: Explore a detailed comparison of YOLOX and RTDETRv2 object detection models, covering architecture, performance, and best use cases for computer vision tasks.
keywords: YOLOX, RTDETRv2, object detection, computer vision, anchor-free, transformer, real-time detection, YOLO models, Ultralytics comparison
---

# YOLOX vs RTDETRv2: A Technical Comparison for Object Detection

Choosing the right object detection model is crucial for computer vision tasks. This page provides a detailed technical comparison between two popular models: [YOLOX](https://arxiv.org/abs/2107.08430) and [RTDETRv2](https://arxiv.org/abs/2304.08069), both available in Ultralytics. We will analyze their architectures, performance, and ideal use cases to help you make an informed decision.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOX", "RTDETRv2"]'></canvas>

## YOLOX: High-Performance Anchor-Free Object Detection

YOLOX, standing for **You Only Look Once (X)**, is an anchor-free object detection model known for its simplicity and high performance. It builds upon the YOLO series, streamlining the architecture and training process while achieving state-of-the-art results.

**Architecture and Key Features:**

- **Anchor-Free Approach**: YOLOX eliminates the need for predefined anchor boxes, simplifying the design and reducing hyperparameters. This leads to easier training and improved generalization, especially for datasets with varying object sizes. Anchor-free detectors are beneficial in scenarios where object scales differ significantly, making YOLOX adaptable to diverse applications.
- **Decoupled Head**: YOLOX employs a decoupled head for classification and localization, which enhances training efficiency and improves accuracy. This separation allows for specialized optimization of each task, leading to better overall detection performance.
- **Advanced Augmentation**: Utilizes MixUp and Mosaic data augmentation techniques to improve robustness and generalization ability. These augmentations create richer training samples, forcing the model to learn more invariant features.
- **Multiple Model Sizes**: YOLOX offers a range of model sizes (Nano, Tiny, Small, Medium, Large, XLarge) to cater to different computational budgets and accuracy requirements. This scalability makes YOLOX suitable for deployment on various hardware, from edge devices to cloud servers.

**Performance and Use Cases:**

YOLOX excels in scenarios demanding a balance of speed and accuracy. Its anchor-free nature and efficient design make it a strong candidate for real-time object detection tasks. Applications include:

- **Robotics**: Real-time perception for robot navigation and interaction.
- **Surveillance**: Efficient object detection in video streams for security applications.
- **Industrial Inspection**: Automated visual inspection on production lines for defect detection.

[Learn more about YOLOX](https://github.com/Megvii-BaseDetection/YOLOX){ .md-button }

## RTDETRv2: Real-Time Detection Transformer v2

RTDETRv2 is a real-time object detection model based on the DETR (DEtection TRansformer) architecture. It leverages the power of transformers for object detection, offering a different approach compared to traditional CNN-based YOLO models.

**Architecture and Key Features:**

- **Transformer-Based Backbone**: RTDETRv2 utilizes a Vision Transformer (ViT) backbone, which is effective at capturing global context in images. This global context awareness can lead to improved detection accuracy, especially in complex scenes. [Vision Transformers (ViTs)](https://www.ultralytics.com/glossary/vision-transformer-vit) have shown remarkable capabilities in various computer vision tasks.
- **Hybrid Encoder**: Employs a hybrid encoder combining CNNs and transformers to efficiently extract features and capture both local and global information. This hybrid approach aims to leverage the strengths of both CNNs and transformers for optimal performance.
- **Deformable Attention**: Uses deformable attention mechanisms to focus on relevant image regions, improving efficiency and accuracy. Deformable attention is particularly useful for handling objects of varying shapes and sizes.
- **Optimized for Real-Time**: RTDETRv2 is engineered for real-time inference, making it suitable for applications with latency constraints. Despite being transformer-based, it achieves competitive speed compared to other real-time detectors.

**Performance and Use Cases:**

RTDETRv2 is particularly well-suited for applications where high accuracy is paramount, while still maintaining reasonable real-time performance. Its transformer architecture and global context understanding make it effective in complex detection scenarios. Ideal use cases include:

- **Autonomous Driving**: Robust object detection in diverse and challenging driving environments. [AI in self-driving cars](https://www.ultralytics.com/solutions/ai-in-self-driving) relies heavily on accurate and fast object detection.
- **Advanced Driver-Assistance Systems (ADAS)**: Reliable perception for safety-critical automotive applications.
- **High-Accuracy Vision AI**: Applications requiring precise object detection, such as medical image analysis or satellite image analysis.

[Learn more about RTDETRv2](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

## Model Comparison Table

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOXnano  | 416                   | 25.8                 | -                              | -                                   | 0.91               | 1.08              |
| YOLOXtiny  | 416                   | 32.8                 | -                              | -                                   | 5.06               | 6.45              |
| YOLOXs     | 640                   | 40.5                 | -                              | 2.56                                | 9.0                | 26.8              |
| YOLOXm     | 640                   | 46.9                 | -                              | 5.43                                | 25.3               | 73.8              |
| YOLOXl     | 640                   | 49.7                 | -                              | 9.04                                | 54.2               | 155.6             |
| YOLOXx     | 640                   | 51.1                 | -                              | 16.1                                | 99.1               | 281.9             |
|            |                       |                      |                                |                                     |                    |                   |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | 54.3                 | -                              | 15.03                               | 76                 | 259               |

## Conclusion

Both YOLOX and RTDETRv2 are powerful object detection models, each with unique strengths. YOLOX offers a streamlined, anchor-free architecture with excellent speed and a good balance of accuracy, making it suitable for real-time applications across various hardware platforms. RTDETRv2, leveraging transformers, provides higher accuracy, especially in complex scenarios, while maintaining competitive real-time performance.

For users seeking the fastest possible inference speed with good accuracy, especially on resource-constrained devices, YOLOX is an excellent choice. For applications prioritizing maximum accuracy and robustness, particularly in complex and safety-critical systems, RTDETRv2 is highly recommended.

Consider exploring other models in the Ultralytics ecosystem, such as [YOLOv8](https://docs.ultralytics.com/models/yolov8/), [YOLOv9](https://docs.ultralytics.com/models/yolov9/), [YOLOv10](https://docs.ultralytics.com/models/yolov10/), [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/), and [YOLO-World](https://docs.ultralytics.com/models/yolo-world/) to find the best fit for your specific computer vision needs. You can also find more information about model performance metrics and evaluation in our [YOLO Performance Metrics guide](https://docs.ultralytics.com/guides/yolo-performance-metrics/).
