---
comments: true
description: Compare YOLOv5 and YOLOv8 for speed, accuracy, and versatility. Learn which Ultralytics model is best for your object detection and vision tasks.
keywords: YOLOv5, YOLOv8, Ultralytics, object detection, computer vision, YOLO models, model comparison, AI, machine learning, deep learning
---

# YOLOv5 vs YOLOv8: A Detailed Comparison

Ultralytics YOLOv5 and YOLOv8 are both cutting-edge, single-stage object detection models renowned for their speed and accuracy. Developed by Ultralytics, these models are widely used in various computer vision applications. This page provides a technical comparison to help users understand the key differences and choose the model best suited for their needs.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv5", "YOLOv8"]'></canvas>

## YOLOv5 Overview

[Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) was introduced in June 2020 by Glenn Jocher of Ultralytics. It quickly gained popularity due to its ease of use, speed, and strong performance. YOLOv5 is implemented in PyTorch and is known for its flexible architecture and efficient training. It offers a range of model sizes (n, s, m, l, x) to accommodate different computational constraints and accuracy requirements. YOLOv5 excels in balancing speed and accuracy, making it suitable for real-time object detection tasks.

**Strengths:**

- **Speed and Efficiency:** YOLOv5 is highly optimized for fast inference, making it ideal for real-time applications.
- **Ease of Use:** With comprehensive [documentation](https://docs.ultralytics.com/models/yolov5/) and a user-friendly interface, YOLOv5 is accessible for both beginners and experienced users.
- **Flexibility:** Offers multiple model sizes to trade-off between speed and accuracy, adaptable to various hardware.
- **Mature and Stable:** Being a more mature model, YOLOv5 has benefited from extensive community testing and refinement.

**Weaknesses:**

- **Accuracy Compared to Newer Models:** While highly accurate, YOLOv5 may be slightly less accurate than its successor, YOLOv8, especially on complex datasets.
- **Feature Set:** Primarily focused on object detection, with segmentation and classification features added later, but not as natively integrated as in YOLOv8.

**Use Cases:**

- **Real-time Object Detection:** Ideal for applications requiring fast processing, such as robotics, autonomous vehicles, and drone vision.
- **Resource-Constrained Environments:** Smaller YOLOv5 models (YOLOv5n, YOLOv5s) are well-suited for edge devices and mobile applications due to their low computational demands.
- **Industrial Inspection:** Speed and reliability make it useful for automated quality control and inspection systems in manufacturing.

[Learn more about YOLOv5](https://docs.ultralytics.com/models/yolov5/){ .md-button }

## YOLOv8 Overview

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), released in January 2023 by Glenn Jocher, Ayush Chaurasia, and Jing Qiu at Ultralytics, is the latest iteration in the YOLO series. Built as a successor to YOLOv5, YOLOv8 is designed to be more versatile and powerful, offering state-of-the-art performance across object detection, image segmentation, and image classification tasks. YOLOv8 introduces architectural improvements and a new task framework, making it more flexible and easier to adapt to different vision tasks.

**Strengths:**

- **State-of-the-Art Performance:** YOLOv8 generally achieves higher accuracy (mAP) compared to YOLOv5, especially in more complex scenarios.
- **Versatility and Task Support:** Natively supports object detection, segmentation, classification, and pose estimation within a unified framework.
- **Architectural Improvements:** Incorporates advanced techniques and optimizations for better accuracy and efficiency.
- **Active Development:** As the latest model, YOLOv8 benefits from ongoing development, improvements, and community support.

**Weaknesses:**

- **Slightly Slower Inference in Some Cases:** Depending on the model size and task, YOLOv8 might be marginally slower than YOLOv5, particularly the smaller variants, though optimizations are continuously being made.
- **Resource Requirements:** To leverage its full potential, especially for larger models and complex tasks, YOLOv8 might require slightly more computational resources than YOLOv5.

**Use Cases:**

- **High-Accuracy Object Detection:** Applications where accuracy is paramount, such as security and surveillance, medical imaging, and detailed video analytics.
- **Multi-Task Vision AI:** Projects requiring a combination of detection, segmentation, or classification tasks, benefiting from YOLOv8's unified framework.
- **Advanced Research and Development:** Ideal for pushing the boundaries of computer vision with a model that incorporates the latest advancements.

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## Performance Metrics Comparison

The table below compares the performance metrics of YOLOv5 and YOLOv8 models on the COCO dataset for object detection.

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
|---------|-----------------------|----------------------|--------------------------------|-------------------------------------|--------------------|-------------------|
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

## Conclusion

Both YOLOv5 and YOLOv8 are powerful object detection models from Ultralytics. YOLOv5 remains a robust choice for applications prioritizing speed and efficiency, especially in resource-limited scenarios. YOLOv8, being the newer model, offers improved accuracy and broader task versatility, making it suitable for more demanding applications and research-oriented projects. The choice between YOLOv5 and YOLOv8 depends on the specific requirements of the project, balancing factors like accuracy, speed, and available computational resources.

Users interested in exploring other models may also consider [YOLOv7](https://docs.ultralytics.com/models/yolov7/), [YOLOv9](https://docs.ultralytics.com/models/yolov9/), [YOLO10](https://docs.ultralytics.com/models/yolov10/) and the latest [YOLO11](https://docs.ultralytics.com/models/yolo11/) for further advancements in object detection technology.
