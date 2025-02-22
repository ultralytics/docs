---
description: Compare YOLO11 and YOLOv9 in architecture, performance, and use cases. Learn which model suits your object detection and computer vision needs.
keywords: YOLO11, YOLOv9, model comparison, object detection, computer vision, Ultralytics, YOLO architecture, YOLO performance, real-time processing
---

# YOLO11 vs YOLOv9: Detailed Technical Comparison

<script async src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO11", "YOLOv9"]'></canvas>

This page provides a detailed technical comparison between two state-of-the-art object detection models: [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) and [YOLOv9](https://docs.ultralytics.com/models/yolov9/). We analyze their architectures, performance metrics, and use cases to assist you in selecting the most suitable model for your computer vision needs.

## Ultralytics YOLO11: The Cutting Edge

[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/), authored by Glenn Jocher and Jing Qiu from Ultralytics and released on 2024-09-27, represents the latest advancement in the YOLO series. Building upon the foundation of previous models like [YOLOv8](https://docs.ultralytics.com/models/yolov8/), YOLO11 is engineered for enhanced accuracy and efficiency across various computer vision tasks including object detection, [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation), [image classification](https://docs.ultralytics.com/tasks/classify/), and [pose estimation](https://docs.ultralytics.com/tasks/pose/).

**Architecture and Features:** YOLO11's architecture is designed for superior feature extraction and faster processing. It achieves higher accuracy with fewer parameters, making it ideal for real-time applications on diverse platforms, from edge devices like [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/) to cloud systems. It maintains task compatibility with YOLOv8, ensuring ease of transition for existing users.

**Performance:** YOLO11 models offer a range of performance profiles. For example, YOLO11m achieves a mAPval50-95 of 51.5 at a 640 image size, with an inference speed of 183.2ms on CPU (ONNX) and 4.7ms on T4 TensorRT10. Its parameter size is 20.1M and FLOPs are 68.0B. This balance positions YOLO11 as a versatile option for applications requiring both accuracy and speed.

**Use Cases:** YOLO11 is particularly well-suited for scenarios demanding high precision and real-time processing, such as [traffic management](https://www.ultralytics.com/blog/optimizingtraffic-management-with-ultralytics-yolo11) in smart cities, [medical image analysis](https://www.ultralytics.com/glossary/medical-image-analysis) for diagnostics, and [quality control](https://www.ultralytics.com/solutions/ai-in-manufacturing) in manufacturing.

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11){ .md-button }

## YOLOv9: Efficiency and Accuracy Innovations

[YOLOv9](https://docs.ultralytics.com/models/yolov9/), developed by Chien-Yao Wang and Hong-Yuan Mark Liao from the Institute of Information Science, Academia Sinica, Taiwan, and introduced on 2024-02-21, brings significant innovations in real-time object detection. YOLOv9 introduces Programmable Gradient Information (PGI) and Generalized Efficient Layer Aggregation Network (GELAN) to address information loss, enhancing both efficiency and accuracy.

**Architecture and Features:** YOLOv9’s architecture focuses on maintaining information integrity throughout the network, crucial for smaller models. The GELAN optimizes computational efficiency, allowing YOLOv9 to achieve state-of-the-art accuracy with fewer parameters and computations.

**Performance:** YOLOv9 demonstrates impressive performance metrics on the COCO dataset. YOLOv9c, for instance, achieves a mAPval50-95 of 53.0 at a 640 size. While specific inference speeds are not directly provided in the document, YOLOv9 prioritizes efficiency, suggesting competitive speeds. YOLOv9c has 25.3M parameters and 102.1B FLOPs.

**Use Cases:** YOLOv9 excels in applications where computational resources are limited but high accuracy is still required. This includes deployment on edge devices, mobile applications, and scenarios where model size and speed are critical, such as [environmental monitoring](https://www.ultralytics.com/blog/ultralytics-yolo11-and-computer-vision-for-environmental-conservation) and [agricultural applications](https://www.ultralytics.com/solutions/ai-in-agriculture).

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

## Model Comparison Table

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLO11n | 640                   | 39.5                 | 56.1                           | 1.5                                 | 2.6                | 6.5               |
| YOLO11s | 640                   | 47.0                 | 90.0                           | 2.5                                 | 9.4                | 21.5              |
| YOLO11m | 640                   | 51.5                 | 183.2                          | 4.7                                 | 20.1               | 68.0              |
| YOLO11l | 640                   | 53.4                 | 238.6                          | 6.2                                 | 25.3               | 86.9              |
| YOLO11x | 640                   | 54.7                 | 462.8                          | 11.3                                | 56.9               | 194.9             |
|         |                       |                      |                                |                                     |                    |                   |
| YOLOv9t | 640                   | 38.3                 | -                              | 2.3                                 | 2.0                | 7.7               |
| YOLOv9s | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e | 640                   | 55.6                 | -                              | 16.77                               | 57.3               | 189.0             |

## Conclusion

Both YOLO11 and YOLOv9 represent significant advancements in object detection. YOLO11 excels in delivering high accuracy and speed, making it suitable for a broad range of demanding applications. YOLOv9, with its focus on efficiency and information preservation, is ideal for resource-constrained environments without sacrificing accuracy.

Users interested in exploring other models might also consider [YOLOv8](https://docs.ultralytics.com/models/yolov8/) for a balance of performance and versatility, or [YOLOv7](https://docs.ultralytics.com/models/yolov7/) for its speed and efficiency in various tasks.