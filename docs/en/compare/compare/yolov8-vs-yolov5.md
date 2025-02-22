---
description: Discover key differences between YOLOv8 and YOLOv5. Compare speed, accuracy, use cases, and more to choose the ideal model for your computer vision needs.
keywords: YOLOv8, YOLOv5, object detection, YOLO comparison, computer vision, model comparison, speed, accuracy, Ultralytics, deep learning
---

# YOLOv8 vs YOLOv5: A Detailed Comparison

<script async src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv8", "YOLOv5"]'></canvas>

Comparing Ultralytics YOLOv8 and YOLOv5 for object detection reveals significant advancements and distinct strengths in each model. Both models, developed by Ultralytics, are renowned for their speed and accuracy, but cater to different user needs and priorities in the field of computer vision. This page provides a technical comparison to help users make informed decisions based on their project requirements.

## YOLOv8: The Cutting-Edge Solution

[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/), introduced on January 10, 2023, by Glenn Jocher, Ayush Chaurasia, and Jing Qiu at Ultralytics, represents the latest iteration in the YOLO series. It is designed as a versatile framework supporting a full range of vision tasks, including detection, segmentation, classification, pose estimation, and oriented bounding boxes. YOLOv8 is not merely an incremental update but a redesigned model that incorporates new architectural features and improvements for enhanced performance and flexibility.

**Architecture and Key Features:**
YOLOv8 adopts a flexible, modular architecture, allowing for easier adaptation and customization. Key architectural changes include a new backbone network, a refined anchor-free detection head, and a novel loss function. These enhancements contribute to improved accuracy and efficiency across various tasks. The model is designed to be user-friendly and is well-documented, making it accessible for both research and practical applications.

**Performance Metrics:**
YOLOv8 demonstrates state-of-the-art performance, achieving a higher mAP with comparable or improved inference speeds compared to its predecessors. The performance table below illustrates the metrics for different YOLOv8 models:

| Model   | size(pixels) | mAP<sup>val</sup><sub>50-95</sub> | Speed<sup>CPU ONNX</sup>(ms) | Speed<sup>T4 TensorRT10</sup>(ms) | params(M) | FLOPs(B) |
| ------- | ------------ | --------------------------------- | ---------------------------- | --------------------------------- | --------- | -------- |
| YOLOv8n | 640          | 37.3                              | 80.4                         | 1.47                              | 3.2       | 8.7      |
| YOLOv8s | 640          | 44.9                              | 128.4                        | 2.66                              | 11.2      | 28.6     |
| YOLOv8m | 640          | 50.2                              | 234.7                        | 5.86                              | 25.9      | 78.9     |
| YOLOv8l | 640          | 52.9                              | 375.2                        | 9.06                              | 43.7      | 165.2    |
| YOLOv8x | 640          | 53.9                              | 479.1                        | 14.37                             | 68.2      | 257.8    |

**Strengths:**

- **Improved Accuracy and Speed:** YOLOv8 generally offers better accuracy and competitive speed compared to YOLOv5, especially in the larger model variants.
- **Versatility:** Supports a wide array of vision tasks beyond object detection, such as instance segmentation and pose estimation, making it a comprehensive tool for various applications.
- **User-Friendly Design:** Ultralytics emphasizes ease of use with YOLOv8, providing clear documentation and tools for training, validation, and deployment, simplifying the workflow for users.
- **Active Development:** As the latest model, YOLOv8 benefits from ongoing updates, community support, and integration with Ultralytics HUB for streamlined model management.

**Weaknesses:**

- **Computational Demand:** Larger YOLOv8 models require more computational resources, which might be a limitation for resource-constrained environments.
- **Newer Architecture:** Being newer, the community and available third-party resources might be less extensive compared to the more established YOLOv5.

**Ideal Use Cases:**
YOLOv8 is well-suited for applications requiring high accuracy and versatility, such as:

- **Advanced Robotics:** For complex object recognition and scene understanding in robotic systems.
- **High-Resolution Image Analysis:** Excels in detailed analysis of high-resolution images where fine-grained object detection is crucial.
- **Multi-Task Vision Systems:** Ideal for systems needing to perform multiple vision tasks simultaneously, like detection and segmentation.

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## YOLOv5: The Established Industry Standard

[Ultralytics YOLOv5](https://docs.ultralytics.com/models/yolov5/), released on June 26, 2020, by Glenn Jocher at Ultralytics, quickly became an industry favorite due to its exceptional balance of speed and accuracy. It is known for its ease of use and rapid deployment capabilities, making it a go-to choice for many developers and researchers.

**Architecture and Key Features:**
YOLOv5 is built on PyTorch and is lauded for its efficient architecture that allows for fast training and inference. It features a CSPDarknet53 backbone, PANet path aggregation network, and a YOLOv5 detection head. YOLOv5 is available in various sizes (n, s, m, l, x), offering scalability and adaptability to different hardware and performance requirements.

**Performance Metrics:**
YOLOv5 is celebrated for its speed and efficiency, providing a range of models that cater to different speed-accuracy trade-offs. The performance metrics for YOLOv5 models are detailed below:

| Model   | size(pixels) | mAP<sup>val</sup><sub>50-95</sub> | Speed<sup>CPU ONNX</sup>(ms) | Speed<sup>T4 TensorRT10</sup>(ms) | params(M) | FLOPs(B) |
| ------- | ------------ | --------------------------------- | ---------------------------- | --------------------------------- | --------- | -------- |
| YOLOv5n | 640          | 28.0                              | 73.6                         | 1.12                              | 2.6       | 7.7      |
| YOLOv5s | 640          | 37.4                              | 120.7                        | 1.92                              | 9.1       | 24.0     |
| YOLOv5m | 640          | 45.4                              | 233.9                        | 4.03                              | 25.1      | 64.2     |
| YOLOv5l | 640          | 49.0                              | 408.4                        | 6.61                              | 53.2      | 135.0    |
| YOLOv5x | 640          | 50.7                              | 763.2                        | 11.89                             | 97.2      | 246.4    |

**Strengths:**

- **Exceptional Speed:** YOLOv5 is optimized for speed, making it highly suitable for real-time object detection applications.
- **Ease of Use and Deployment:** Known for its simplicity, YOLOv5 is easy to train, deploy, and integrate into existing systems. Extensive tutorials and documentation are available.
- **Mature Ecosystem:** YOLOv5 has a large and active community, offering abundant resources, pre-trained models, and support, which is beneficial for troubleshooting and development.
- **Versatile Model Sizes:** The availability of different model sizes allows users to choose the best fit for their specific performance and resource constraints.

**Weaknesses:**

- **Lower Accuracy Compared to YOLOv8:** Generally, YOLOv5 models tend to have slightly lower accuracy compared to the newer YOLOv8, especially in tasks requiring very high precision.
- **Limited Task Support:** Primarily focused on object detection, with segmentation and pose estimation capabilities introduced later but not as central as in YOLOv8.

**Ideal Use Cases:**
YOLOv5 is optimally used in scenarios where speed and efficiency are paramount:

- **Real-time Video Surveillance:** Ideal for applications needing rapid object detection in video streams.
- **Edge Computing Devices:** Smaller YOLOv5 models are excellent for deployment on edge devices with limited computational power due to their efficiency.
- **Mobile Applications:** Suitable for mobile applications where fast inference times and smaller model sizes are crucial.

[Learn more about YOLOv5](https://docs.ultralytics.com/models/yolov5/){ .md-button }

## Model Comparison Table

| Model   | size<br><sup>(pixels) | mAP<sup>val</sup><sub>50-95</sub> | Speed<sup>CPU ONNX</sup>(ms) | Speed<sup>T4 TensorRT10</sup>(ms) | params(M) | FLOPs(B) |
| ------- | --------------------- | --------------------------------- | ---------------------------- | --------------------------------- | --------- | -------- |
| YOLOv8n | 640                   | 37.3                              | 80.4                         | 1.47                              | 3.2       | 8.7      |
| YOLOv8s | 640                   | 44.9                              | 128.4                        | 2.66                              | 11.2      | 28.6     |
| YOLOv8m | 640                   | 50.2                              | 234.7                        | 5.86                              | 25.9      | 78.9     |
| YOLOv8l | 640                   | 52.9                              | 375.2                        | 9.06                              | 43.7      | 165.2    |
| YOLOv8x | 640                   | 53.9                              | 479.1                        | 14.37                             | 68.2      | 257.8    |
|         |                       |                                   |                              |                                   |           |          |
| YOLOv5n | 640                   | 28.0                              | 73.6                         | 1.12                              | 2.6       | 7.7      |
| YOLOv5s | 640                   | 37.4                              | 120.7                        | 1.92                              | 9.1       | 24.0     |
| YOLOv5m | 640                   | 45.4                              | 233.9                        | 4.03                              | 25.1      | 64.2     |
| YOLOv5l | 640                   | 49.0                              | 408.4                        | 6.61                              | 53.2      | 135.0    |
| YOLOv5x | 640                   | 50.7                              | 763.2                        | 11.89                             | 97.2      | 246.4    |

## Conclusion

Choosing between YOLOv8 and YOLOv5 depends on the specific needs of your project. If accuracy and versatility across different vision tasks are paramount, and computational resources are less of a constraint, YOLOv8 is the superior choice. It excels in scenarios requiring state-of-the-art performance and multi-task capabilities. Conversely, if real-time performance, ease of deployment, and resource efficiency are critical, especially on edge devices or in mobile applications, YOLOv5 remains a highly effective and widely supported option. Users might also be interested in exploring other models in the YOLO family such as [YOLOv7](https://docs.ultralytics.com/models/yolov7/), [YOLOv9](https://docs.ultralytics.com/models/yolov9/), [YOLO10](https://docs.ultralytics.com/models/yolov10/), [YOLO11](https://docs.ultralytics.com/models/yolo11/), and [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/) to find the best model for their specific needs.