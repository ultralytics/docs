---
description: Compare YOLOv5 and YOLOv9 - performance, architecture, and use cases. Find the best model for real-time object detection and computer vision tasks.
keywords: YOLOv5, YOLOv9, object detection, model comparison, performance metrics, real-time detection, computer vision, Ultralytics, machine learning
---

# YOLOv5 vs YOLOv9: A Detailed Comparison

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv5", "YOLOv9"]'></canvas>

This page provides a technical comparison between two state-of-the-art object detection models: Ultralytics YOLOv5 and YOLOv9. Both models are part of the YOLO (You Only Look Once) series, renowned for their speed and accuracy in real-time object detection. This comparison will delve into their architectural differences, performance metrics, and suitable use cases to help you choose the right model for your computer vision needs.

## YOLOv5

**Authors:** Glenn Jocher
**Organization:** Ultralytics
**Date:** 2020-06-26
**GitHub:** [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
**Documentation:** [https://docs.ultralytics.com/models/yolov5/](https://docs.ultralytics.com/models/yolov5/)

Ultralytics YOLOv5 is a highly popular one-stage object detection model known for its ease of use and excellent balance of speed and accuracy. It utilizes an architecture that builds upon previous YOLO versions, incorporating advancements like CSP bottlenecks in the backbone and neck, and efficient anchor-based detection. YOLOv5 offers a range of model sizes (n, s, m, l, x), allowing users to select a configuration that best fits their computational resources and performance requirements. Its architecture is designed for efficient inference, making it suitable for real-time applications across various hardware platforms.

**Strengths:**

- **Speed and Efficiency:** YOLOv5 is optimized for fast inference, making it ideal for real-time object detection tasks.
- **Ease of Use:** With user-friendly [Python](https://docs.ultralytics.com/usage/python/) and [CLI](https://docs.ultralytics.com/usage/cli/) interfaces, YOLOv5 simplifies training, validation, and deployment.
- **Versatility:** Applicable to a wide range of tasks including [object detection](https://www.ultralytics.com/glossary/object-detection), [image segmentation](https://www.ultralytics.com/glossary/image-segmentation), and [image classification](https://www.ultralytics.com/glossary/image-classification).
- **Scalability:** Offers multiple model sizes to accommodate different computational constraints.

**Weaknesses:**

- **Accuracy:** While highly accurate, it may be surpassed by newer models like YOLOv9 in certain benchmarks.
- **Complexity:** For tasks requiring the absolute highest accuracy, more complex architectures might be considered.

**Use Cases:**

- Real-time video surveillance
- Autonomous vehicles
- Robotics
- Industrial automation
- Quality control in manufacturing ([improving manufacturing with computer vision](https://www.ultralytics.com/blog/improving-manufacturing-with-computer-vision))

[Learn more about YOLOv5](https://docs.ultralytics.com/models/yolov5/){ .md-button }

## YOLOv9

**Authors:** Chien-Yao Wang and Hong-Yuan Mark Liao
**Organization:** Institute of Information Science, Academia Sinica, Taiwan
**Date:** 2024-02-21
**Arxiv:** [https://arxiv.org/abs/2402.13616](https://arxiv.org/abs/2402.13616)
**GitHub:** [https://github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)
**Documentation:** [https://docs.ultralytics.com/models/yolov9/](https://docs.ultralytics.com/models/yolov9/)

YOLOv9 introduces innovative concepts like Programmable Gradient Information (PGI) and Generalized Efficient Layer Aggregation Network (GELAN) to address information loss in deep networks. PGI helps the network learn what it's intended to learn by preserving crucial gradient information, while GELAN optimizes network architecture for better parameter utilization and computational efficiency. This results in a model family that not only achieves higher accuracy but also maintains or improves upon the efficiency of previous YOLO models. YOLOv9 is designed to excel in scenarios where high accuracy and computational efficiency are paramount.

**Strengths:**

- **Enhanced Accuracy:** YOLOv9 achieves state-of-the-art accuracy in real-time object detection, outperforming many previous models, including YOLOv5, on datasets like COCO.
- **Improved Efficiency:** Through GELAN and PGI, YOLOv9 models are designed to be computationally efficient, requiring fewer parameters and FLOPs for comparable or better performance.
- **Information Preservation:** PGI mechanism effectively mitigates information loss, particularly beneficial for deeper and more complex networks.
- **Adaptability:** Offers various model sizes (t, s, m, c, e) to suit different application needs, balancing accuracy and speed.

**Weaknesses:**

- **Training Resources:** Training YOLOv9 models may require more computational resources and time compared to YOLOv5, as noted in the [YOLOv9 documentation](https://docs.ultralytics.com/models/yolov9/).
- **Newer Architecture:** Being a more recent model, the community and ecosystem support might still be developing compared to the more established YOLOv5.

**Use Cases:**

- Applications demanding high accuracy object detection
- Resource-constrained environments where efficiency is critical
- Advanced video analytics
- High-precision industrial inspection
- [AI in traffic management](https://www.ultralytics.com/blog/ai-in-traffic-management-from-congestion-to-coordination) and [smart cities](https://www.ultralytics.com/blog/computer-vision-ai-in-smart-cities)

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

## Performance Comparison Table

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv5n | 640                   | 28.0                 | 73.6                           | 1.12                                | 2.6                | 7.7               |
| YOLOv5s | 640                   | 37.4                 | 120.7                          | 1.92                                | 9.1                | 24.0              |
| YOLOv5m | 640                   | 45.4                 | 233.9                          | 4.03                                | 25.1               | 64.2              |
| YOLOv5l | 640                   | 49.0                 | 408.4                          | 6.61                                | 53.2               | 135.0             |
| YOLOv5x | 640                   | 50.7                 | 763.2                          | 11.89                               | 97.2               | 246.4             |
|         |                       |                      |                                |                                     |                    |                   |
| YOLOv9t | 640                   | 38.3                 | -                              | 2.3                                 | 2.0                | 7.7               |
| YOLOv9s | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e | 640                   | 55.6                 | -                              | 16.77                               | 57.3               | 189.0             |

## Conclusion

Choosing between YOLOv5 and YOLOv9 depends on your specific requirements. If speed and ease of deployment are primary concerns, and a good balance of accuracy is sufficient, YOLOv5 remains an excellent choice. It's well-established and has a robust ecosystem. However, if your application demands the highest possible accuracy and you have the resources for potentially longer training times, YOLOv9 offers superior performance and efficiency due to its advanced architectural innovations.

Consider exploring other models in the YOLO family as well, such as the highly versatile [YOLOv8](https://docs.ultralytics.com/models/yolov8/) and the cutting-edge [YOLO11](https://docs.ultralytics.com/models/yolo11/), to find the perfect fit for your project. You might also be interested in comparing YOLOv9 to other real-time detectors like [RT-DETR](https://docs.ultralytics.com/compare/rtdetr-vs-yolov9/) or [PP-YOLOe](https://docs.ultralytics.com/compare/pp-yoloe-vs-yolov9/).
