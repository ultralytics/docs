---
description: Compare YOLOv8 and YOLOv9 models for object detection. Explore their accuracy, speed, resources, and use cases to choose the best model for your needs.
keywords: YOLOv8, YOLOv9, object detection, model comparison, Ultralytics, performance metrics, real-time AI, computer vision, YOLO series
---

# Model Comparison: YOLOv8 vs YOLOv9 for Object Detection

Choosing the right object detection model is crucial for balancing accuracy, speed, and computational resources. This page offers a detailed technical comparison between Ultralytics YOLOv8 and YOLOv9, both cutting-edge models in the YOLO series. We will analyze their architectures, performance, and use cases to help you determine the best fit for your needs.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv8", "YOLOv9"]'></canvas>

## YOLOv8: Streamlined and Versatile

[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) is the latest iteration in the YOLO family, known for its balance of speed and accuracy. It's designed to be user-friendly and versatile, supporting a wide range of vision tasks beyond object detection, including [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) and [pose estimation](https://docs.ultralytics.com/tasks/pose/). YOLOv8 builds upon previous versions with architectural refinements and a focus on ease of use, making it suitable for rapid deployment across various applications.

**Strengths:**

- **Balanced Performance:** YOLOv8 achieves state-of-the-art accuracy while maintaining fast inference speeds, making it suitable for real-time applications. See [YOLOv8 performance metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/) for detailed benchmarks.
- **Versatility:** Supports multiple vision tasks including detection, segmentation, classification, and pose estimation, offering a comprehensive solution for diverse computer vision needs.
- **Ease of Use:** Ultralytics provides excellent [documentation](https://docs.ultralytics.com/) and tools, simplifying training, validation, and deployment workflows.
- **Strong Ecosystem:** Benefits from a large community, continuous updates, and integration with [Ultralytics HUB](https://www.ultralytics.com/hub) for streamlined model management.

**Weaknesses:**

- **Resource Intensive:** Larger YOLOv8 models require significant computational resources, especially for training and inference, compared to smaller, more lightweight models.
- **Optimization Needs:** For extremely resource-constrained devices, further optimization like [model pruning](https://www.ultralytics.com/glossary/pruning) may be necessary to achieve optimal speed.

**Use Cases:**

YOLOv8's versatility makes it ideal for a broad spectrum of applications, from real-time object detection in [security alarm systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8) and [smart cities](https://www.ultralytics.com/blog/computer-vision-ai-in-smart-cities) to complex tasks like [pose estimation](https://www.ultralytics.com/blog/pose-estimation-with-ultralytics-yolov8) in [healthcare](https://www.ultralytics.com/solutions/ai-in-healthcare) and [manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing). Its ease of use also makes it excellent for rapid prototyping and development.

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## YOLOv9: Efficiency and Accuracy Innovations

[YOLOv9](https://docs.ultralytics.com/models/yolov9/) represents a significant step forward in real-time object detection, focusing on improved efficiency and accuracy through architectural innovations. Developed by Wang et al. and introduced in 2024, YOLOv9 introduces Programmable Gradient Information (PGI) and Generalized Efficient Layer Aggregation Network (GELAN). These innovations are designed to address information loss, leading to enhanced performance, especially in lightweight models.

**Strengths:**

- **High Accuracy and Efficiency:** YOLOv9 achieves higher mAP with fewer parameters and FLOPs compared to previous models, demonstrating improved efficiency.
- **PGI and GELAN:** PGI prevents information loss during feature extraction, while GELAN optimizes network architecture for better parameter utilization and computational efficiency.
- **Lightweight Model Performance:** YOLOv9 particularly excels in lightweight models, offering superior accuracy compared to similar-sized models with reduced computational overhead.
- **State-of-the-Art Results:** Sets new benchmarks in real-time object detection, outperforming other models in terms of accuracy and efficiency on the COCO dataset.

**Weaknesses:**

- **Higher Training Resources:** Training YOLOv9 models may require more computational resources and time compared to YOLOv8, especially for larger models.
- **Newer Architecture:** As a newer model, the community and ecosystem around YOLOv9 are still developing compared to the more mature YOLOv8.

**Use Cases:**

YOLOv9 is particularly well-suited for applications where high accuracy and efficiency are paramount, especially in resource-constrained environments. This includes real-time object detection on edge devices, applications requiring fast inference with limited computational power, and scenarios benefiting from highly accurate lightweight models. Its enhanced efficiency makes it valuable in [robotics](https://www.ultralytics.com/glossary/robotics) and [autonomous vehicles](https://www.ultralytics.com/solutions/ai-in-self-driving) where computational resources are limited.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

## Performance Comparison Table

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv8n | 640                   | 37.3                 | 80.4                           | 1.47                                | 3.2                | 8.7               |
| YOLOv8s | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |
|         |                       |                      |                                |                                     |                    |                   |
| YOLOv9t | 640                   | 38.3                 | -                              | 2.3                                 | 2.0                | 7.7               |
| YOLOv9s | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e | 640                   | 55.6                 | -                              | 16.77                               | 57.3               | 189.0             |

## Conclusion

Both YOLOv8 and YOLOv9 are excellent choices for object detection, each with unique strengths. YOLOv8 excels in versatility and ease of use, making it a robust general-purpose model. YOLOv9, with its innovative architecture, offers enhanced efficiency and accuracy, particularly beneficial for lightweight models and resource-limited environments. The choice between them depends on the specific application requirements, with YOLOv9 being preferable when efficiency and top accuracy are critical, and YOLOv8 favored for its broader task support and established ecosystem.

For users interested in other models, Ultralytics also offers [YOLOv5](https://docs.ultralytics.com/models/yolov5/), [YOLOv7](https://docs.ultralytics.com/models/yolov7/) and [YOLO11](https://docs.ultralytics.com/models/yolo11/), each with its own performance characteristics.

**YOLOv8 Details:**

- Authors: Glenn Jocher, Ayush Chaurasia, and Jing Qiu
- Organization: Ultralytics
- Date: 2023-01-10
- Arxiv Link: None
- GitHub: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- Docs: [YOLOv8 Docs](https://docs.ultralytics.com/models/yolov8/)

**YOLOv9 Details:**

- Authors: Chien-Yao Wang and Hong-Yuan Mark Liao
- Organization: Institute of Information Science, Academia Sinica, Taiwan
- Date: 2024-02-21
- Arxiv Link: [arXiv:2402.13616](https://arxiv.org/abs/2402.13616)
- GitHub: [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)
- Docs: [YOLOv9 Docs](https://docs.ultralytics.com/models/yolov9/)
