---
comments: true
description: Compare YOLOv6 and RTDETR for object detection. Explore their architectures, performances, and use cases to choose your optimal computer vision model.
keywords: YOLOv6, RTDETR, object detection, model comparison, YOLO, Vision Transformers, CNN, real-time detection, Ultralytics, computer vision
---

# YOLOv6-3.0 vs RTDETRv2: Detailed Model Comparison

Choosing the optimal object detection model is vital for successful computer vision applications. This page offers a technical comparison between YOLOv6-3.0 and RTDETRv2, two leading models in the field, to assist you in making an informed choice. We analyze their architectural designs, performance benchmarks, and suitability for different applications.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv6-3.0", "RTDETRv2"]'></canvas>

## YOLOv6-3.0: Streamlined Efficiency

YOLOv6-3.0, developed by Meituan and detailed in their [arXiv paper](https://arxiv.org/abs/2301.05586) released on 2023-01-13, is designed for **high efficiency and speed** in object detection. As a member of the Ultralytics YOLO family, YOLOv6-3.0 prioritizes rapid inference, making it excellent for real-time applications and environments with limited resources.

### Architecture and Key Features

YOLOv6-3.0 utilizes a **Convolutional Neural Network (CNN)** architecture, focusing on computational efficiency. Key aspects include:

- **Efficient Backbone:** Employs a streamlined backbone for feature extraction, minimizing computational overhead.
- **Streamlined Detection Head:** Features a lightweight detection head to ensure rapid processing.
- **One-Stage Detector:** As a one-stage detector, it offers a balance of speed and accuracy, suitable for various object detection needs.

These architectural choices enable YOLOv6-3.0 to achieve fast inference times without significantly sacrificing accuracy.

### Performance Metrics

YOLOv6-3.0 excels in speed and efficiency, making it a strong contender for real-time tasks. Key performance indicators include:

- **mAPval50-95**: Up to 52.8% for YOLOv6-3.0l
- **Inference Speed (T4 TensorRT10)**: As low as 1.17 ms for YOLOv6-3.0n
- **Model Size (parameters)**: Starting from 4.7M for YOLOv6-3.0n

### Use Cases and Strengths

YOLOv6-3.0 is particularly well-suited for applications requiring **real-time object detection** and deployment in **resource-constrained environments**. Ideal use cases include:

- **Edge Deployment:** Efficient performance on edge devices like [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/) and [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/).
- **Real-time Systems:** Applications such as [security alarm systems](https://docs.ultralytics.com/guides/security-alarm-system/) and robotics where low latency is critical.
- **Mobile Applications:** Lightweight design is suitable for mobile platforms.

Its primary strength lies in its speed and efficiency, making it highly deployable and practical for real-world applications where computational resources are limited.

[Learn more about YOLOv6](https://docs.ultralytics.com/models/yolov6/){ .md-button }

## RTDETRv2: Accuracy with Transformers

RTDETRv2, authored by Wenyu Lv et al. from Baidu and introduced in their [arXiv paper](https://arxiv.org/abs/2304.08069) on 2023-04-17, takes a different approach by leveraging **Vision Transformers (ViT)**. This model prioritizes accuracy and robust feature extraction, utilizing transformers to capture global context within images.

### Architecture and Key Features

RTDETRv2's architecture is characterized by:

- **Transformer Encoder:** Employs a transformer encoder to process the entire image, capturing long-range dependencies for enhanced context understanding.
- **Hybrid CNN Feature Extraction:** Combines CNNs for initial feature extraction with transformer layers to incorporate global context effectively.
- **Anchor-Free Detection:** Simplifies the detection process by eliminating the need for predefined anchor boxes.

This transformer-based design allows RTDETRv2 to potentially achieve higher accuracy, especially in complex and detailed scenes, by better understanding the global context of the image.

### Performance Metrics

RTDETRv2 prioritizes accuracy and delivers competitive performance, especially in mean Average Precision. Key performance indicators include:

- **mAPval50-95**: Up to 54.3% for RTDETRv2-x
- **Inference Speed (T4 TensorRT10)**: Starting from 5.03 ms for RTDETRv2-s
- **Model Size (parameters)**: Starting from 20M for RTDETRv2-s

### Use Cases and Strengths

RTDETRv2 is ideally suited for applications where **high accuracy is paramount** and sufficient computational resources are available. These include:

- **Autonomous Vehicles:** For precise and reliable environmental perception, crucial for safety. [AI in self-driving cars](https://www.ultralytics.com/solutions/ai-in-automotive)
- **Medical Imaging:** For detailed and accurate detection of anomalies in medical images, aiding in diagnostics. [AI in Healthcare](https://www.ultralytics.com/solutions/ai-in-healthcare)
- **Robotics:** Enabling robots to interact with objects in complex environments with high precision. [From Algorithms to Automation: AI's Role in Robotics](https://www.ultralytics.com/blog/from-algorithms-to-automation-ais-role-in-robotics)

RTDETRv2's strength lies in its transformer-based architecture, enabling it to achieve superior accuracy and robust feature extraction for complex object detection tasks.

[Learn more about RTDETRv2](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

| Model       | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ----------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv6-3.0n | 640                   | 37.5                 | -                              | 1.17                                | 4.7                | 11.4              |
| YOLOv6-3.0s | 640                   | 45.0                 | -                              | 2.66                                | 18.5               | 45.3              |
| YOLOv6-3.0m | 640                   | 50.0                 | -                              | 5.28                                | 34.9               | 85.8              |
| YOLOv6-3.0l | 640                   | 52.8                 | -                              | 8.95                                | 59.6               | 150.7             |
|             |                       |                      |                                |                                     |                    |                   |
| RTDETRv2-s  | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m  | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l  | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x  | 640                   | 54.3                 | -                              | 15.03                               | 76                 | 259               |

## Conclusion

Both YOLOv6-3.0 and RTDETRv2 are powerful object detection models, each with unique strengths. YOLOv6-3.0 excels in speed and efficiency, making it ideal for real-time applications on resource-limited devices. RTDETRv2, with its transformer-based architecture, prioritizes accuracy and is better suited for applications demanding high precision and having access to more computational resources.

Depending on your project requirements, you might also consider other models in the Ultralytics YOLO family, such as [YOLOv5](https://docs.ultralytics.com/models/yolov5/) for its versatility and ease of use, [YOLOv7](https://docs.ultralytics.com/models/yolov7/) for a balance of speed and accuracy, or the cutting-edge [YOLOv8](https://docs.ultralytics.com/models/yolov8/) and [YOLO11](https://docs.ultralytics.com/models/yolo11/) for state-of-the-art performance. For scenarios where transformer architectures are preferred, exploring [DAMO-YOLO](https://docs.ultralytics.com/compare/damo-yolo-vs-rtdetr/) could also be beneficial.
