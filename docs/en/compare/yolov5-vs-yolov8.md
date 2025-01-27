---
comments: true
description: Discover key differences between YOLOv5 and YOLOv8. Compare speed, accuracy, and versatility to choose the right object detection model for your project.
keywords: YOLOv5, YOLOv8, object detection, model comparison, Ultralytics, AI models, computer vision, speed, accuracy, versatility
---

# YOLOv5 vs YOLOv8: A Detailed Comparison for Object Detection

<script async src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv5", "YOLOv8"]'></canvas>

Choosing the right object detection model is crucial for computer vision projects. Ultralytics YOLOv5 and YOLOv8 are both state-of-the-art models, but cater to different needs and priorities. This page provides a technical comparison to help you make an informed decision.

## YOLOv5: Efficiency and Speed

[Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) is renowned for its efficiency and speed, making it an excellent choice for real-time applications and resource-constrained environments. Its architecture is based on a single-stage detector, focusing on streamlining the detection process for rapid inference.

**Architecture and Key Features:**

- **Single-Stage Detector:** YOLOv5 employs a single-stage detection approach, directly predicting bounding boxes and class probabilities in one forward pass, optimizing for speed.
- **CSP Bottleneck:** Utilizes CSP (Cross Stage Partial) bottlenecks in its backbone to enhance feature extraction and reduce computational load.
- **Path Aggregation Network (PANet):** Employs PANet for efficient feature fusion, improving information flow across different network levels.

**Performance Metrics:**

YOLOv5 models are available in various sizes (n, s, m, l, x), offering a trade-off between speed and accuracy. The smaller models like YOLOv5n and YOLOv5s are particularly fast, making them suitable for edge devices and applications requiring high frame rates. While slightly less accurate than its successor, YOLOv5 still provides excellent performance for many object detection tasks, achieving a balance between speed and mAP.

**Use Cases:**

- **Edge Deployment:** Ideal for deployment on edge devices like Raspberry Pi or NVIDIA Jetson due to its lightweight nature and fast inference speeds.
- **Real-time Applications:** Suitable for applications requiring real-time object detection, such as robotics, drone vision, and real-time analytics.
- **Wildlife Conservation:** As demonstrated by the Kashmir World Foundation, YOLOv5 is effectively used in wildlife conservation efforts.

[Learn more about YOLOv5](https://docs.ultralytics.com/models/yolov5/){ .md-button }

## YOLOv8: Accuracy and Versatility

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) builds upon the strengths of previous YOLO versions, introducing architectural improvements and training methodology enhancements to achieve higher accuracy and greater versatility across various vision AI tasks including [object tracking](https://docs.ultralytics.com/modes/track/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), [segmentation](https://docs.ultralytics.com/tasks/segment/), and [classification](https://docs.ultralytics.com/tasks/classify/).

**Architecture and Key Features:**

- **Anchor-Free Detection:** YOLOv8 moves to an anchor-free detection paradigm, simplifying the model and potentially improving generalization.
- **C2f Module:** Introduces a new C2f module in the backbone, further enhancing feature extraction efficiency.
- **Decoupled Head:** Employs a decoupled detection head to separate classification and regression tasks, potentially leading to improved accuracy.

**Performance Metrics:**

YOLOv8 generally outperforms YOLOv5 in terms of accuracy (mAP), especially in the larger model sizes (l, x). While slightly slower than YOLOv5n and YOLOv5s, YOLOv8 still maintains impressive inference speeds, making it suitable for real-time applications where higher accuracy is prioritized. Its improved architecture leads to better performance across a range of metrics, as indicated in the comparison table.

**Use Cases:**

- **High-Accuracy Demands:** Applications where accuracy is paramount, such as medical imaging or complex scene analysis.
- **Versatile Tasks:** YOLOv8's architecture is designed to handle a broader range of tasks beyond object detection, including segmentation and pose estimation, making it a versatile choice for diverse projects.
- **Smart Retail and Queue Management:** YOLOv8 can be effectively applied in scenarios like [intelligent stores](https://www.ultralytics.com/event/build-intelligent-stores-with-ultralytics-yolov8-and-seeed-studio) and [queue management systems](https://www.ultralytics.com/blog/revolutionizing-queue-management-with-ultralytics-yolov8-and-openvino).

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## Model Comparison Table

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv5n | 640                   | 28.0                 | 73.6                           | 1.12                                | 2.6                | 7.7               |
| YOLOv5s | 640                   | 37.4                 | 120.7                          | 1.92                                | 9.1                | 24.0              |
| YOLOv5m | 640                   | 45.4                 | 233.9                          | 4.03                                | 25.1               | 64.2              |
| YOLOv5l | 640                   | 49.0                 | 408.4                          | 6.61                                | 53.2               | 135.0             |
| YOLOv5x | 640                   | 50.7                 | 763.2                          | 11.89                               | 97.2               | 246.4             |
|         |                       |                      |                                |                                     |                    |                   |
| YOLOv8n | 640                   | 37.3                 | 80.4                           | 1.47                                | 3.2                | 8.7               |
| YOLOv8s | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |

## Strengths and Weaknesses

- **YOLOv5 Strengths:**
    - **Speed:** Faster inference speeds, particularly in smaller models.
    - **Efficiency:** Lower computational requirements, suitable for edge devices.
    - **Mature and Stable:** Well-established and widely adopted with a strong community.
- **YOLOv5 Weaknesses:**

    - **Accuracy:** Generally lower mAP compared to YOLOv8, especially in larger models.
    - **Versatility:** Primarily focused on object detection, with less native support for other vision tasks compared to YOLOv8.

- **YOLOv8 Strengths:**
    - **Accuracy:** Higher mAP across model sizes due to architectural improvements.
    - **Versatility:** Supports a broader range of vision tasks, including detection, segmentation, pose estimation, and classification.
    - **State-of-the-art:** Represents the latest advancements in the YOLO series, incorporating modern detection techniques.
- **YOLOv8 Weaknesses:**
    - **Speed:** Slightly slower inference speeds than YOLOv5, especially in smaller models.
    - **Resource Intensity:** Higher computational requirements, potentially less suitable for extremely resource-constrained devices compared to YOLOv5n/s.

## Conclusion and Alternatives

Choosing between YOLOv5 and YOLOv8 depends on your project priorities. If speed and efficiency are paramount, especially for edge deployment, YOLOv5 remains an excellent choice. If higher accuracy and versatility across tasks are needed, YOLOv8 is the preferred option.

Users interested in exploring other models within the Ultralytics ecosystem may also consider:

- [YOLOv7](https://docs.ultralytics.com/models/yolov7/): A predecessor to YOLOv8, offering a balance of speed and accuracy.
- [YOLOv9](https://docs.ultralytics.com/models/yolov9/) and [YOLOv10](https://docs.ultralytics.com/models/yolov10/): The newest models in the YOLO family, pushing the boundaries of real-time object detection.
- [RT-DETR](https://docs.ultralytics.com/models/rtdetr/): A real-time detector based on DETR architecture, offering a different approach to object detection.

Ultimately, evaluating your specific use case requirements against the strengths and weaknesses of each model will guide you to the optimal choice for your computer vision project. For further exploration, refer to the [Ultralytics documentation](https://docs.ultralytics.com/guides/) for comprehensive tutorials and guides.