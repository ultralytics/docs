---
comments: true
description: Discover the key differences between Ultralytics YOLOv8 and YOLOv5. Explore their performance, strengths, and use cases for optimal object detection.
keywords: YOLOv8, YOLOv5, object detection, comparison, Ultralytics, performance metrics, computer vision, machine learning, AI models, YOLO models
---

# YOLOv8 vs YOLOv5: A Detailed Comparison

<script async src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv8", "YOLOv5"]'></canvas>

Choosing the right object detection model is crucial for computer vision projects. Ultralytics YOLOv8 and YOLOv5 are both highly popular models, but they cater to different needs and use cases. This page provides a detailed technical comparison to help you make an informed decision.

## YOLOv8: The State-of-the-Art Successor

Ultralytics YOLOv8 is the latest iteration in the YOLO series, building upon the successes of its predecessors like YOLOv5. It introduces architectural improvements and new features aimed at enhancing both accuracy and efficiency.

### Architecture and Key Features

YOLOv8 adopts a more flexible and modular architecture. Key changes include a new backbone network, a streamlined C2f module replacing the C3 module, and an anchor-free detection head. These modifications contribute to improved performance across various model sizes. YOLOv8 also expands its task capabilities beyond object detection to include [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) and [pose estimation](https://docs.ultralytics.com/tasks/pose/).

### Performance Metrics

YOLOv8 demonstrates superior performance compared to YOLOv5, especially in terms of accuracy. The table below shows a performance comparison, highlighting improvements in mAP (mean Average Precision) and inference speed. While maintaining competitive speed, YOLOv8 achieves higher accuracy, making it suitable for applications demanding precision.

### Use Cases and Strengths

YOLOv8 is ideal for cutting-edge applications requiring the highest possible accuracy in object detection, such as [robotics](https://www.ultralytics.com/glossary/robotics), [autonomous vehicles](https://www.ultralytics.com/solutions/ai-in-self-driving), and advanced [security systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8). Its strengths lie in its state-of-the-art accuracy, versatile task support, and ongoing development by Ultralytics.

### Weaknesses

Being the newer model, YOLOv8's ecosystem and community support, while rapidly growing, are still evolving compared to the more mature YOLOv5. Some users might find fewer readily available resources and community-contributed tools initially.

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## YOLOv5: The Proven and Versatile Choice

Ultralytics YOLOv5 is renowned for its speed, ease of use, and extensive community support. It has been widely adopted across various industries and applications due to its excellent balance of performance and accessibility.

### Architecture and Key Features

YOLOv5 is built with a focus on efficiency and speed. Its architecture includes a CSPDarknet53 backbone, PANet feature aggregation network, and a YOLOv3 detection head. YOLOv5 is well-regarded for its speed and relatively smaller model sizes, making it deployable on a wide range of hardware, including edge devices.

### Performance Metrics

YOLOv5 excels in inference speed, offering real-time object detection capabilities. As shown in the comparison table, it provides a range of model sizes, allowing users to choose a configuration that best fits their speed and accuracy requirements. While generally slightly less accurate than YOLOv8, YOLOv5 remains highly competitive and often preferred for speed-critical applications.

### Use Cases and Strengths

YOLOv5 is exceptionally well-suited for real-time applications where speed is paramount, such as [webcam-based detection](https://www.ultralytics.com/blog/object-detection-with-a-pre-trained-ultralytics-yolov8-model), mobile applications, and [edge deployments](https://www.ultralytics.com/glossary/edge-ai). Its strengths include its mature and well-documented codebase, large and active community, and broad hardware compatibility. It's a robust and reliable choice for a wide array of object detection tasks.

### Weaknesses

Compared to YOLOv8, YOLOv5 generally exhibits slightly lower accuracy, particularly on complex datasets. While still highly accurate, users prioritizing absolute maximum precision might find YOLOv8 a better fit.

[Learn more about YOLOv5](https://github.com/ultralytics/yolov5){ .md-button }

## Model Comparison Table

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv8n | 640                   | 37.3                 | 80.4                           | 1.47                                | 3.2                | 8.7               |
| YOLOv8s | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |
|         |                       |                      |                                |                                     |                    |                   |
| YOLOv5n | 640                   | 28.0                 | 73.6                           | 1.12                                | 2.6                | 7.7               |
| YOLOv5s | 640                   | 37.4                 | 120.7                          | 1.92                                | 9.1                | 24.0              |
| YOLOv5m | 640                   | 45.4                 | 233.9                          | 4.03                                | 25.1               | 64.2              |
| YOLOv5l | 640                   | 49.0                 | 408.4                          | 6.61                                | 53.2               | 135.0             |
| YOLOv5x | 640                   | 50.7                 | 763.2                          | 11.89                               | 97.2               | 246.4             |

## Other YOLO Models to Consider

Besides YOLOv8 and YOLOv5, Ultralytics offers a range of other YOLO models, each with unique strengths:

- **YOLOv10:** The latest model focusing on efficiency and eliminating NMS for faster performance. [Explore YOLOv10 documentation](https://docs.ultralytics.com/models/yolov10/).
- **YOLOv9:** Known for its advancements in real-time object detection with innovations like PGI and GELAN. [Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/).
- **YOLOv7:** A breakthrough real-time object detector with a balance of speed and accuracy. [Discover YOLOv7 features](https://docs.ultralytics.com/models/yolov7/).
- **YOLOv6:** A top-tier object detector balancing speed and accuracy, developed by Meituan. [See YOLOv6 details](https://docs.ultralytics.com/models/yolov6/).
- **YOLOv4:** A state-of-the-art real-time object detection model known for its architecture and features. [Explore YOLOv4 documentation](https://docs.ultralytics.com/models/yolov4/).
- **YOLOv3:** An older but still relevant version, with variants like YOLOv3-Ultralytics and YOLOv3u. [Learn about YOLOv3 and its variants](https://docs.ultralytics.com/models/yolov3/).
- **YOLO-NAS:** By Deci AI, this model offers state-of-the-art object detection with quantization support. [Discover YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/).
- **RT-DETR:** Baidu's Vision Transformer-based real-time object detector with high accuracy and adaptable speed. [Explore RT-DETR](https://docs.ultralytics.com/models/rtdetr/).
- **YOLO-World:** For efficient, real-time open-vocabulary object detection. [Learn about YOLO-World](https://docs.ultralytics.com/models/yolo-world/).

By considering these factors and exploring the performance table, you can select the YOLO model that best aligns with your project requirements and achieve optimal object detection results.