---
description: Explore YOLO11 and YOLOX, two leading object detection models. Compare architecture, performance, and use cases to select the best model for your needs.
keywords: YOLO11, YOLOX, object detection, machine learning, computer vision, model comparison, deep learning, Ultralytics, real-time detection, anchor-free models
---

# Model Comparison: YOLO11 vs YOLOX for Object Detection

Choosing the right object detection model is crucial for achieving optimal performance in computer vision applications. This page offers a detailed technical comparison between Ultralytics YOLO11 and YOLOX, two advanced models designed for object detection tasks. We will explore their architectural nuances, performance benchmarks, and suitability for different use cases to guide you in making an informed decision.

<script async src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO11", "YOLOX"]'></canvas>

## YOLO11: The Latest Real-Time Object Detector from Ultralytics

[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/), introduced in 2024 by Glenn Jocher and Jing Qiu from Ultralytics, is the newest iteration in the YOLO series. It's engineered for state-of-the-art performance in real-time object detection and various other vision tasks. YOLO11 builds upon previous YOLO versions, incorporating architectural improvements for enhanced speed and accuracy.

**Key Features and Architecture:**

YOLO11 maintains the real-time detection focus of its predecessors while introducing refinements to boost performance. It is designed to be versatile and user-friendly, suitable for a broad spectrum of applications. The architecture emphasizes efficiency, aiming for a balance between high accuracy and rapid inference.

**Performance Metrics:**

YOLO11 models are available in various sizes, from YOLO11n to YOLO11x, catering to different computational needs. Below are the performance metrics for object detection on the COCO dataset:

| Model   | size<sup>(pixels) | mAP<sup>val</sup><sub>50-95</sub> | Speed<sup>CPU ONNX</sup><sub>(ms)</sub> | Speed<sup>T4 TensorRT10</sup><sub>(ms)</sub> | params<sup>(M) | FLOPs<sup>(B) |
| ------- | ----------------- | --------------------------------- | --------------------------------------- | -------------------------------------------- | -------------- | ------------- |
| YOLO11n | 640               | 39.5                              | 56.1                                    | 1.5                                          | 2.6            | 6.5           |
| YOLO11s | 640               | 47.0                              | 90.0                                    | 2.5                                          | 9.4            | 21.5          |
| YOLO11m | 640               | 51.5                              | 183.2                                   | 4.7                                          | 20.1           | 68.0          |
| YOLO11l | 640               | 53.4                              | 238.6                                   | 6.2                                          | 25.3           | 86.9          |
| YOLO11x | 640               | 54.7                              | 462.8                                   | 11.3                                         | 56.9           | 194.9         |

**Strengths:**

- **State-of-the-art Performance:** Achieves high mAP and fast inference speeds, making it suitable for demanding applications.
- **Versatility:** Supports object detection, segmentation, pose estimation, and image classification tasks, providing a unified solution for computer vision needs.
- **User-Friendly:** Part of the Ultralytics ecosystem, known for ease of use and excellent documentation.
- **Scalability:** Offers multiple model sizes to suit different hardware and performance requirements.

**Weaknesses:**

- Larger models (YOLO11x) can have slower inference speeds compared to smaller variants, requiring more computational resources.

**Ideal Use Cases:**

YOLO11 is well-suited for applications that require a balance of high accuracy and real-time processing, such as:

- **Autonomous systems:** For use in [robotics](https://www.ultralytics.com/glossary/robotics) and [autonomous vehicles](https://www.ultralytics.com/solutions/ai-in-self-driving) requiring robust and fast object detection.
- **Security and surveillance:** In [security alarm systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8) for real-time monitoring and threat detection.
- **Industrial automation:** For quality control and process monitoring in [manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing).

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

## YOLOX: High-Performance Anchor-Free Detector

[YOLOX](https://yolox.readthedocs.io/en/latest/), introduced by Megvii in 2021 and detailed in their [Arxiv report](https://arxiv.org/abs/2107.08430), is an anchor-free version of YOLO known for its simplicity and high performance. Developed by Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun, YOLOX aims to simplify the YOLO architecture while improving accuracy and speed.

**Key Features and Architecture:**

YOLOX distinguishes itself with an anchor-free detection mechanism, which simplifies the model design and enhances generalization. Key architectural elements include:

- **Anchor-Free Approach:** Removes the complexity of anchor boxes, leading to a more streamlined and efficient detection process.
- **Decoupled Head:** Separates the classification and localization heads, improving training efficiency and overall performance.
- **Advanced Training Techniques:** Utilizes techniques like SimOTA label assignment and strong data augmentation to enhance robustness and accuracy.

**Performance Metrics:**

YOLOX offers a range of models, including Nano, Tiny, S, M, L, and X, each optimized for different trade-offs between speed and accuracy. Here's a comparison of performance metrics:

| Model     | size<sup>(pixels) | mAP<sup>val</sup><sub>50-95</sub> | Speed<sup>T4 TensorRT10</sup><sub>(ms)</sub> | params<sup>(M) | FLOPs<sup>(B) |
| --------- | ----------------- | --------------------------------- | -------------------------------------------- | -------------- | ------------- |
| YOLOXnano | 416               | 25.8                              | -                                            | 0.91           | 1.08          |
| YOLOXtiny | 416               | 32.8                              | -                                            | 5.06           | 6.45          |
| YOLOXs    | 640               | 40.5                              | 2.56                                         | 9.0            | 26.8          |
| YOLOXm    | 640               | 46.9                              | 5.43                                         | 25.3           | 73.8          |
| YOLOXl    | 640               | 49.7                              | 9.04                                         | 54.2           | 155.6         |
| YOLOXx    | 640               | 51.1                              | 16.1                                         | 99.1           | 281.9         |

**Strengths:**

- **High Accuracy:** Achieves competitive mAP scores, particularly in larger models like YOLOX-x.
- **Anchor-Free Design:** Simplifies the architecture and reduces the number of design parameters, potentially improving generalization.
- **Efficient Training:** Benefits from advanced training techniques for faster convergence and better performance.
- **Variety of Models:** Offers a range of model sizes to cater to different computational constraints and application needs.

**Weaknesses:**

- Inference speed may be slower for larger models compared to some highly optimized models like YOLO11n, especially on CPU.
- Deployment and integration may require familiarity with the Megvii ecosystem and codebase.

**Ideal Use Cases:**

YOLOX is suitable for applications that prioritize a balance of high accuracy and reasonable speed, including:

- **Research and development:** As a strong baseline model for object detection research and further development.
- **Industrial inspection:** Where high accuracy is needed to detect defects or anomalies.
- **Advanced driver-assistance systems (ADAS):** For applications requiring reliable object detection with moderate real-time requirements.

## Comparison Table

| Model     | size<sup>(pixels) | mAP<sup>val</sup><sub>50-95</sub> | Speed<sup>CPU ONNX</sup><sub>(ms)</sub> | Speed<sup>T4 TensorRT10</sup><sub>(ms)</sub> | params<sup>(M) | FLOPs<sup>(B) |
| --------- | ----------------- | --------------------------------- | --------------------------------------- | -------------------------------------------- | -------------- | ------------- |
| YOLO11n   | 640               | 39.5                              | 56.1                                    | 1.5                                          | 2.6            | 6.5           |
| YOLO11s   | 640               | 47.0                              | 90.0                                    | 2.5                                          | 9.4            | 21.5          |
| YOLO11m   | 640               | 51.5                              | 183.2                                   | 4.7                                          | 20.1           | 68.0          |
| YOLO11l   | 640               | 53.4                              | 238.6                                   | 6.2                                          | 25.3           | 86.9          |
| YOLO11x   | 640               | 54.7                              | 462.8                                   | 11.3                                         | 56.9           | 194.9         |
|           |                   |                                   |                                         |                                              |                |               |
| YOLOXnano | 416               | 25.8                              | -                                       | -                                            | 0.91           | 1.08          |
| YOLOXtiny | 416               | 32.8                              | -                                       | -                                            | 5.06           | 6.45          |
| YOLOXs    | 640               | 40.5                              | -                                       | 2.56                                         | 9.0            | 26.8          |
| YOLOXm    | 640               | 46.9                              | -                                       | 5.43                                         | 25.3           | 73.8          |
| YOLOXl    | 640               | 49.7                              | -                                       | 9.04                                         | 54.2           | 155.6         |
| YOLOXx    | 640               | 51.1                              | -                                       | 16.1                                         | 99.1           | 281.9         |

## Conclusion

Both YOLO11 and YOLOX are powerful object detection models, each with unique strengths. YOLO11 excels in versatility and real-time performance within the Ultralytics ecosystem, making it ideal for users seeking ease of use and broad task support. YOLOX, with its anchor-free design and high accuracy, is a robust choice for research and applications requiring precise detection.

For users interested in exploring other models, Ultralytics also offers a range of cutting-edge models such as [YOLOv8](https://docs.ultralytics.com/models/yolov8/), [YOLOv9](https://docs.ultralytics.com/models/yolov9/), and [YOLOv10](https://docs.ultralytics.com/models/yolov10/), each with its own set of features and performance characteristics. Consider exploring [Model Deployment Options](https://docs.ultralytics.com/guides/model-deployment-options/) to understand how to best deploy these models for your specific needs.

[Learn more about YOLOX](https://github.com/Megvii-BaseDetection/YOLOX){ .md-button }