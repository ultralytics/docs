---
comments: true
description: Dive into a detailed comparison of YOLO11 and RTDETRv2. Explore their architecture, strengths, weaknesses, and ideal use cases for object detection.
keywords: YOLO11, RTDETRv2, object detection model, YOLO comparison, real-time detection, vision transformer, ultralytics models, model architecture, performance metrics
---

# YOLO11 vs RTDETRv2: A Detailed Model Comparison for Object Detection

Choosing the right object detection model is crucial for the success of any computer vision project. Ultralytics offers a range of models, including the cutting-edge YOLO11 and the robust RTDETRv2. This page provides a detailed technical comparison to help you make an informed decision based on your specific needs.

<script async src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO11", "RTDETRv2"]'></canvas>

## Architecture and Key Features

**YOLO11** represents the latest evolution in the Ultralytics YOLO series, building upon the foundation of previous models like YOLOv8 and YOLOv9. It is an anchor-free, one-stage detector known for its speed and efficiency. YOLO11 focuses on optimizing both accuracy and inference time, making it suitable for real-time applications and deployment on edge devices. Key improvements in YOLO11 include enhanced feature extraction and a streamlined architecture that reduces parameter count while maintaining or improving accuracy. [Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

**RTDETRv2**, on the other hand, is a real-time object detection model based on the DETR (DEtection TRansformer) architecture. Unlike YOLO models, RTDETRv2 leverages a Vision Transformer backbone, enabling it to capture global context in images more effectively. This architecture often leads to higher accuracy, particularly in complex scenes with overlapping objects. RTDETRv2 is designed for scenarios where accuracy is prioritized without sacrificing too much on inference speed. [Explore RTDETRv2 in Ultralytics Docs](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

## Performance Metrics

The table below summarizes the performance characteristics of YOLO11 and RTDETRv2 models across different sizes, using metrics like mAP (mean Average Precision), inference speed, and model size. These metrics are crucial for understanding the trade-offs between accuracy and computational efficiency for each model. For a deeper understanding of these metrics, refer to our guide on [YOLO Performance Metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/).

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLO11n    | 640                   | 39.5                 | 56.1                           | 1.5                                 | 2.6                | 6.5               |
| YOLO11s    | 640                   | 47.0                 | 90.0                           | 2.5                                 | 9.4                | 21.5              |
| YOLO11m    | 640                   | 51.5                 | 183.2                          | 4.7                                 | 20.1               | 68.0              |
| YOLO11l    | 640                   | 53.4                 | 238.6                          | 6.2                                 | 25.3               | 86.9              |
| YOLO11x    | 640                   | 54.7                 | 462.8                          | 11.3                                | 56.9               | 194.9             |
|            |                       |                      |                                |                                     |                    |                   |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | 54.3                 | -                              | 15.03                               | 76                 | 259               |

## Strengths and Weaknesses

**YOLO11 Strengths:**

- **Speed:** YOLO11 is optimized for real-time inference, offering very fast processing speeds, especially on GPUs and when exported to formats like TensorRT.
- **Efficiency:** With a lower parameter count compared to larger models, YOLO11 is more efficient in terms of computational resources and model size, making it ideal for resource-constrained environments.
- **Ease of Deployment:** The YOLO family is known for its straightforward deployment process across various platforms, including edge devices and cloud servers. Explore different [Model Deployment Options](https://docs.ultralytics.com/guides/model-deployment-options/).

**YOLO11 Weaknesses:**

- **Accuracy Trade-off:** While highly accurate, YOLO11 might slightly lag behind larger, more complex models like RTDETRv2 in mAP, especially on datasets requiring fine-grained detail.

**RTDETRv2 Strengths:**

- **High Accuracy:** Leveraging a Vision Transformer, RTDETRv2 excels in achieving high object detection accuracy, particularly in scenarios with complex scenes and occlusions.
- **Contextual Understanding:** The Transformer architecture enables RTDETRv2 to better understand the global context of an image, improving its ability to differentiate objects and reduce false positives.

**RTDETRv2 Weaknesses:**

- **Speed and Resource Intensity:** RTDETRv2, especially larger variants, tends to be slower and more computationally intensive than YOLO11 due to the complexity of Transformer networks.
- **Model Size:** Transformer-based models often have larger model sizes and parameter counts, which can be a concern for deployment on edge devices with limited memory.

## Ideal Use Cases

**YOLO11 Use Cases:**

- **Real-time Object Detection:** Applications requiring rapid inference, such as autonomous driving, robotics, and real-time video analytics.
- **Edge Deployment:** Scenarios where models need to run efficiently on edge devices with limited computational power, like mobile applications, drones, and embedded systems.
- **High-Speed Applications:** Situations demanding high throughput and low latency, such as high-speed manufacturing quality control or fast-paced surveillance systems.

**RTDETRv2 Use Cases:**

- **High-Accuracy Demanding Applications:** Scenarios where achieving the highest possible accuracy is paramount, such as medical image analysis, detailed satellite imagery analysis, and security systems requiring precise detection.
- **Complex Scene Understanding:** Applications dealing with crowded scenes, occluded objects, or complex backgrounds, where the contextual understanding of Transformers provides an advantage.
- **Cloud-Based Inference:** Deployments where computational resources are less constrained, and the focus is on maximizing detection accuracy, often in cloud environments.

## Choosing Between YOLO11 and RTDETRv2

- **Choose YOLO11 if:** Speed and efficiency are your top priorities, and you need a model that performs well in real-time or on resource-limited devices.
- **Choose RTDETRv2 if:** Accuracy is your primary concern, and you are working with complex scenes where contextual understanding is crucial, and computational resources are less of a constraint.

Both YOLO11 and RTDETRv2 are powerful models within the Ultralytics ecosystem. Depending on your project's specific requirements for speed, accuracy, and deployment environment, one will likely be more suitable than the other. Consider experimenting with both to determine which best fits your needs. You might also be interested in exploring other models like [YOLOv8](https://docs.ultralytics.com/models/yolov8/) or [YOLOv9](https://docs.ultralytics.com/models/yolov9/) to find the optimal balance for your application. For further exploration, visit the [Ultralytics Docs](https://docs.ultralytics.com/guides/) and the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics).