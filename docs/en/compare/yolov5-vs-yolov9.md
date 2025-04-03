---
comments: true
description: Compare YOLOv5 and YOLOv9 - performance, architecture, and use cases. Find the best model for real-time object detection and computer vision tasks.
keywords: YOLOv5, YOLOv9, object detection, model comparison, performance metrics, real-time detection, computer vision, Ultralytics, machine learning
---

# YOLOv5 vs YOLOv9: A Detailed Comparison

This page provides a technical comparison between two significant object detection models: Ultralytics YOLOv5 and YOLOv9. Both models are part of the influential YOLO (You Only Look Once) series, known for balancing speed and accuracy in real-time [object detection](https://www.ultralytics.com/glossary/object-detection). This comparison explores their architectural differences, performance metrics, and ideal use cases to help you select the most suitable model for your computer vision projects.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv5", "YOLOv9"]'></canvas>

## Ultralytics YOLOv5: The Established Industry Standard

**Author:** Glenn Jocher  
**Organization:** Ultralytics  
**Date:** 2020-06-26  
**GitHub:** <https://github.com/ultralytics/yolov5>  
**Documentation:** <https://docs.ultralytics.com/models/yolov5/>

Ultralytics YOLOv5 quickly gained popularity after its release due to its remarkable balance of speed, accuracy, and ease of use. Developed entirely in [PyTorch](https://www.ultralytics.com/glossary/pytorch), YOLOv5 features an architecture utilizing CSPDarknet53 as the backbone and PANet for feature aggregation, along with an efficient anchor-based detection head. It offers various model sizes (n, s, m, l, x), allowing users to choose based on their computational resources and performance needs.

**Strengths:**

- **Exceptional Speed and Efficiency:** YOLOv5 is highly optimized for fast inference, making it ideal for real-time applications on various hardware, including [edge devices](https://www.ultralytics.com/blog/edge-ai-and-aiot-upgrade-any-camera-with-ultralytics-yolov8-in-a-no-code-way).
- **Ease of Use:** Ultralytics YOLOv5 is renowned for its streamlined user experience, simple [Python](https://docs.ultralytics.com/usage/python/) and [CLI](https://docs.ultralytics.com/usage/cli/) interfaces, and extensive [documentation](https://docs.ultralytics.com/yolov5/).
- **Well-Maintained Ecosystem:** Benefits from the integrated Ultralytics ecosystem, featuring active development, a large and supportive community ([join on Discord](https://discord.com/invite/ultralytics)), frequent updates, and comprehensive resources like [Ultralytics HUB](https://www.ultralytics.com/hub) for no-code training.
- **Performance Balance:** Achieves a strong trade-off between inference speed and detection accuracy, suitable for diverse real-world deployment scenarios.
- **Versatility:** Supports multiple tasks including [object detection](https://docs.ultralytics.com/tasks/detect/), [instance segmentation](https://docs.ultralytics.com/tasks/segment/), and [image classification](https://docs.ultralytics.com/tasks/classify/).
- **Training Efficiency:** Offers efficient training processes, readily available pre-trained weights, and generally lower memory requirements compared to many other architectures, especially transformer-based models.

**Weaknesses:**

- **Accuracy:** While highly accurate for its time, newer models like YOLOv9 can achieve higher mAP scores on benchmarks like [COCO](https://docs.ultralytics.com/datasets/detect/coco/).
- **Anchor-Based:** Relies on predefined anchor boxes, which might require more tuning for specific datasets compared to anchor-free approaches.

**Use Cases:**

- Real-time video surveillance and [security systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8).
- Deployment on resource-constrained edge devices ([Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/), [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/)).
- Industrial automation and quality control ([improving manufacturing with computer vision](https://www.ultralytics.com/blog/improving-manufacturing-with-computer-vision)).
- Rapid prototyping and development due to its ease of use.

[Learn more about YOLOv5](https://docs.ultralytics.com/models/yolov5/){ .md-button }

## YOLOv9: Advancing Accuracy and Efficiency

**Authors:** Chien-Yao Wang, Hong-Yuan Mark Liao  
**Organization:** Institute of Information Science, Academia Sinica, Taiwan  
**Date:** 2024-02-21  
**Arxiv:** <https://arxiv.org/abs/2402.13616>  
**GitHub:** <https://github.com/WongKinYiu/yolov9>  
**Documentation:** <https://docs.ultralytics.com/models/yolov9/>

YOLOv9 introduces significant architectural innovations, namely Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN). PGI aims to mitigate information loss as data flows through deep networks by providing complete input information for loss function calculation. GELAN is a novel architecture designed for superior parameter utilization and computational efficiency. These advancements allow YOLOv9 to achieve higher accuracy while maintaining efficiency.

**Strengths:**

- **Enhanced Accuracy:** Sets new state-of-the-art results on the COCO dataset for real-time object detectors, surpassing YOLOv5 and other models in mAP.
- **Improved Efficiency:** GELAN and PGI contribute to models that require fewer parameters and computational resources (FLOPs) for comparable or better performance than previous models.
- **Information Preservation:** PGI effectively addresses the information bottleneck problem, crucial for training deeper and more complex networks accurately.

**Weaknesses:**

- **Training Resources:** Training YOLOv9 models can be more resource-intensive and time-consuming compared to Ultralytics YOLOv5, as noted in the [YOLOv9 documentation](https://docs.ultralytics.com/models/yolov9/).
- **Newer Architecture:** As a more recent model from a different research group, its ecosystem, community support, and third-party integrations might be less mature than the well-established Ultralytics YOLOv5.
- **Task Versatility:** Primarily focused on object detection, lacking the built-in support for segmentation, classification, pose estimation etc., found in Ultralytics models like YOLOv5 and [YOLOv8](https://docs.ultralytics.com/models/yolov8/).

**Use Cases:**

- Applications demanding the highest possible object detection accuracy.
- Scenarios where computational efficiency is critical alongside high performance.
- Advanced video analytics and high-precision industrial inspection.
- [AI in traffic management](https://www.ultralytics.com/blog/ai-in-traffic-management-from-congestion-to-coordination) and [smart city](https://www.ultralytics.com/blog/computer-vision-ai-in-smart-cities) applications requiring top-tier detection.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

## Performance Comparison: YOLOv5 vs YOLOv9

The table below provides a comparison of various model variants from YOLOv5 and YOLOv9, highlighting their performance metrics on the COCO dataset. Note that YOLOv9 CPU speeds are not readily available from the source repository.

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv5n | 640                   | 28.0                 | **73.6**                       | **1.12**                            | 2.6                | 7.7               |
| YOLOv5s | 640                   | 37.4                 | 120.7                          | **1.92**                            | 9.1                | 24.0              |
| YOLOv5m | 640                   | 45.4                 | 233.9                          | 4.03                                | 25.1               | 64.2              |
| YOLOv5l | 640                   | 49.0                 | 408.4                          | 6.61                                | 53.2               | 135.0             |
| YOLOv5x | 640                   | 50.7                 | 763.2                          | 11.89                               | 97.2               | 246.4             |
|         |                       |                      |                                |                                     |                    |                   |
| YOLOv9t | 640                   | 38.3                 | -                              | 2.3                                 | **2.0**            | **7.7**           |
| YOLOv9s | 640                   | 46.8                 | -                              | 3.54                                | **7.1**            | 26.4              |
| YOLOv9m | 640                   | 51.4                 | -                              | 6.43                                | **20.0**           | 76.3              |
| YOLOv9c | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e | 640                   | **55.6**             | -                              | 16.77                               | 57.3               | 189.0             |

## Conclusion

Both Ultralytics YOLOv5 and YOLOv9 offer compelling capabilities for object detection.

**Ultralytics YOLOv5** remains an excellent choice, particularly when ease of use, rapid deployment, versatility across tasks (detection, segmentation, classification), and a mature ecosystem are priorities. Its balance of speed and accuracy makes it highly suitable for a wide range of real-time applications and deployment on diverse hardware. The Ultralytics framework provides a seamless experience from training to deployment, backed by strong community support and continuous updates.

**YOLOv9** pushes the boundaries of accuracy and efficiency through novel architectural designs like PGI and GELAN. It is ideal for users who need state-of-the-art detection accuracy, especially if they have the resources for potentially more demanding training. However, it lacks the task versatility and the streamlined, fully integrated ecosystem provided by Ultralytics models.

For users seeking the latest advancements within the Ultralytics ecosystem, consider exploring [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/), [YOLOv10](https://docs.ultralytics.com/models/yolov10/), or the newest [YOLO11](https://docs.ultralytics.com/models/yolo11/), which build upon the strengths of YOLOv5 while incorporating newer techniques and offering enhanced performance and versatility. You can find further comparisons, such as [YOLOv8 vs YOLOv5](https://docs.ultralytics.com/compare/yolov8-vs-yolov5/) or comparisons with other models like [RT-DETR](https://docs.ultralytics.com/models/rtdetr/), in the [Ultralytics documentation](https://docs.ultralytics.com/compare/).
