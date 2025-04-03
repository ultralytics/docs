---
comments: true
description: Compare YOLOv10 and YOLOv9 object detection models. Explore architectures, metrics, and use cases to choose the best model for your application.
keywords: YOLOv10,YOLOv9,Ultralytics,object detection,real-time AI,computer vision,model comparison,AI deployment,deep learning
---

# YOLOv10 vs YOLOv9: A Technical Comparison for Object Detection

Choosing the right object detection model is crucial for computer vision projects. Ultralytics offers state-of-the-art YOLO models, and this page provides a detailed technical comparison between two cutting-edge options: [YOLOv10](https://docs.ultralytics.com/models/yolov10/) and [YOLOv9](https://docs.ultralytics.com/models/yolov9/). We will analyze their architectures, performance metrics, and ideal applications to help you make an informed decision based on factors like [accuracy](https://www.ultralytics.com/glossary/accuracy), speed, and resource requirements.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv10", "YOLOv9"]'></canvas>

## YOLOv10: Real-Time End-to-End Efficiency

[YOLOv10](https://docs.ultralytics.com/models/yolov10/) is the latest evolution in the YOLO series, developed by researchers at Tsinghua University. Released in May 2024, YOLOv10 focuses on maximizing efficiency and speed for real-time object detection without sacrificing accuracy. This model is particularly designed for end-to-end deployment, minimizing [latency](https://www.ultralytics.com/glossary/inference-latency) by employing NMS-free training techniques.

**Technical Details:**

- **Authors:** Ao Wang, Hui Chen, Lihao Liu, et al.
- **Organization:** Tsinghua University
- **Date:** 2024-05-23
- **Arxiv Link:** <https://arxiv.org/abs/2405.14458>
- **GitHub Link:** <https://github.com/THU-MIG/yolov10>
- **Docs Link:** <https://docs.ultralytics.com/models/yolov10/>

### Architecture and Key Features

YOLOv10 introduces several architectural innovations to enhance both efficiency and accuracy:

- **Consistent Dual Assignments**: This method combines one-to-many and one-to-one label assignment strategies during training, enabling rich supervision and [NMS](https://www.ultralytics.com/glossary/non-maximum-suppression-nms)-free inference, reducing post-processing overhead.
- **Holistic Efficiency-Accuracy Driven Design**: This strategy involves optimizing components like a lightweight classification head, spatial-channel decoupled downsampling, and rank-guided block design to reduce computational cost and improve parameter utilization. Accuracy is boosted via large-kernel convolutions and partial self-attention (PSA) for better global representation learning.

### Performance Metrics

YOLOv10 demonstrates superior performance and efficiency compared to previous YOLO versions and other state-of-the-art models. As detailed in the table below, YOLOv10-B achieves 46% less latency and 25% fewer parameters than YOLOv9-C for comparable performance. Its integration within the Ultralytics ecosystem allows users to leverage these advancements easily through the familiar [Python package](https://docs.ultralytics.com/usage/python/).

### Strengths

- **High Efficiency and Speed**: Optimized for real-time performance and low latency.
- **NMS-Free Training**: Enables end-to-end deployment and reduces inference time.
- **State-of-the-art Performance**: Outperforms previous YOLO versions in speed and parameter efficiency while maintaining competitive accuracy.
- **Ultralytics Integration**: Benefits from the well-maintained Ultralytics ecosystem, offering a streamlined user experience, simple API, and extensive [documentation](https://docs.ultralytics.com/).

### Weaknesses

- **Relatively New Model**: Being a recent model, it may have a smaller community and fewer deployment examples compared to more mature models like [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/).

### Ideal Use Cases

YOLOv10 is ideally suited for applications that demand high speed and efficiency, especially on resource-constrained devices:

- **Edge Computing**: Real-time object detection on edge devices like [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/) and mobile platforms.
- **High-Speed Applications**: Scenarios requiring minimal latency, such as [autonomous driving](https://www.ultralytics.com/solutions/ai-in-automotive) and [robotics](https://www.ultralytics.com/blog/from-algorithms-to-automation-ais-role-in-robotics).
- **Efficient Deployment**: Applications where model size and computational cost are critical, like mobile and embedded systems.

[Learn more about YOLOv10](https://docs.ultralytics.com/models/yolov10/){ .md-button }

## YOLOv9: Programmable Gradient Information

[YOLOv9](https://docs.ultralytics.com/models/yolov9/), introduced in February 2024 by Chien-Yao Wang and Hong-Yuan Mark Liao, represents a significant advancement in efficient object detection. The core innovation of YOLOv9 lies in its **Programmable Gradient Information (PGI)**, designed to address information loss during the deep learning process using techniques like **Generalized Efficient Layer Aggregation Networks (GELAN)**.

**Technical Details:**

- **Authors:** Chien-Yao Wang, Hong-Yuan Mark Liao
- **Organization:** Institute of Information Science, Academia Sinica, Taiwan
- **Date:** 2024-02-21
- **Arxiv Link:** <https://arxiv.org/abs/2402.13616>
- **GitHub Link:** <https://github.com/WongKinYiu/yolov9>
- **Docs Link:** <https://docs.ultralytics.com/models/yolov9/>

### Architecture and Key Features

YOLOv9 leverages GELAN to enhance feature extraction and maintain information integrity. PGI helps manage gradient flow, ensuring the model learns effectively and mitigating information bottlenecks common in deep networks. This approach leads to a model that is both accurate and parameter-efficient.

### Performance Metrics

YOLOv9 demonstrates impressive performance on the [COCO dataset](https://docs.ultralytics.com/datasets/detect/coco/). For example, YOLOv9c achieves a mAP<sup>val</sup> 50-95 of 53.0% with 25.3M parameters. Its architecture allows high accuracy with relatively fewer parameters and computations compared to some prior models.

### Strengths

- **High Accuracy**: Achieves state-of-the-art accuracy in object detection, particularly the larger YOLOv9e variant.
- **Parameter Efficiency**: Utilizes parameters and computations effectively due to GELAN and PGI.
- **Novel Approach**: Introduces Programmable Gradient Information for better learning and information retention.

### Weaknesses

- **Relatively New:** Might have a smaller community and fewer deployment examples compared to more established models integrated longer within the Ultralytics ecosystem.
- **Complexity:** The PGI concept might add complexity compared to simpler architectures.

### Ideal Use Cases

YOLOv9 is well-suited for applications requiring high accuracy and efficiency:

- **Advanced Robotics:** Object detection in complex robotic systems.
- **High-Resolution Image Analysis:** Scenarios demanding detailed analysis of large images.
- **Resource-Constrained Environments:** Edge devices and mobile applications where computational power is limited but high accuracy is needed.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

## Performance Comparison: YOLOv10 vs. YOLOv9

The table below provides a quantitative comparison of different YOLOv10 and YOLOv9 model variants based on performance metrics evaluated on the COCO dataset.

| Model    | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| -------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv10n | 640                   | 39.5                 | -                              | **1.56**                            | 2.3                | **6.7**           |
| YOLOv10s | 640                   | 46.7                 | -                              | 2.66                                | 7.2                | 21.6              |
| YOLOv10m | 640                   | 51.3                 | -                              | 5.48                                | 15.4               | 59.1              |
| YOLOv10b | 640                   | 52.7                 | -                              | 6.54                                | 24.4               | 92.0              |
| YOLOv10l | 640                   | 53.3                 | -                              | 8.33                                | 29.5               | 120.3             |
| YOLOv10x | 640                   | 54.4                 | -                              | 12.2                                | 56.9               | 160.4             |
|          |                       |                      |                                |                                     |                    |                   |
| YOLOv9t  | 640                   | 38.3                 | -                              | 2.3                                 | **2.0**            | 7.7               |
| YOLOv9s  | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m  | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c  | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e  | 640                   | **55.6**             | -                              | 16.77                               | 57.3               | 189.0             |

Analysis of the table reveals:

- **Speed:** YOLOv10 models generally exhibit lower latency (faster inference) on T4 TensorRT, especially the smaller variants like YOLOv10n, benefiting from the NMS-free design.
- **Accuracy:** YOLOv9e achieves the highest mAP<sup>val</sup> 50-95 among the listed models, showcasing the effectiveness of PGI and GELAN for accuracy. However, YOLOv10x is highly competitive. Across comparable sizes (e.g., YOLOv10s vs YOLOv9s, YOLOv10m vs YOLOv9m), performance is very close, with YOLOv10 often having a slight edge in speed and YOLOv9 sometimes slightly higher mAP.
- **Efficiency:** YOLOv9t has the fewest parameters, while YOLOv10n has the lowest FLOPs and fastest TensorRT speed. YOLOv10 models generally offer a strong balance between parameters, FLOPs, speed, and accuracy.

Both YOLOv10 and YOLOv9 offer compelling advantages. YOLOv10 excels in real-time, low-latency scenarios due to its NMS-free architecture and efficiency optimizations. YOLOv9 pushes accuracy boundaries with its novel PGI and GELAN architecture, offering high precision, especially in larger model variants. The integration of these models into the Ultralytics framework provides users with a streamlined experience, leveraging efficient training processes, readily available pre-trained weights, and the benefits of a well-maintained ecosystem with extensive documentation and community support.

## Explore Other Models

Users interested in YOLOv10 and YOLOv9 might also want to explore other state-of-the-art models available through Ultralytics:

- **[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/)**: A highly successful and versatile model known for its excellent balance of speed and accuracy across various tasks like detection, segmentation, pose estimation, and classification.
- **[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)**: The latest model from Ultralytics, focusing on speed, efficiency, and enhanced task versatility, built upon an anchor-free architecture.
- **[YOLOv5](https://docs.ultralytics.com/models/yolov5/)**: A widely adopted and foundational model known for its reliability, speed, and ease of use.
- **[RT-DETR](https://docs.ultralytics.com/models/rtdetr/)**: A real-time transformer-based detection model offering competitive performance.
