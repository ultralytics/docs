---
comments: true
description: Compare YOLOv5 and YOLOv8 for speed, accuracy, and versatility. Learn which Ultralytics model is best for your object detection and vision tasks.
keywords: YOLOv5, YOLOv8, Ultralytics, object detection, computer vision, YOLO models, model comparison, AI, machine learning, deep learning
---

# YOLOv5 vs YOLOv8: A Detailed Comparison

Comparing Ultralytics YOLOv5 and Ultralytics YOLOv8 for object detection reveals significant advancements and distinct strengths in each model. Both models, developed by Ultralytics, are renowned for their speed and accuracy, but cater to different user needs and priorities in the field of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv). This page provides a technical comparison to help users make informed decisions based on their project requirements.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv5", "YOLOv8"]'></canvas>

## YOLOv5: The Established Industry Standard

**Author**: Glenn Jocher  
**Organization**: Ultralytics  
**Date**: 2020-06-26  
**GitHub Link**: <https://github.com/ultralytics/yolov5>  
**Docs Link**: <https://docs.ultralytics.com/models/yolov5/>

[Ultralytics YOLOv5](https://docs.ultralytics.com/models/yolov5/), released in June 2020, quickly became an industry favorite due to its exceptional balance of speed and accuracy. Built on [PyTorch](https://pytorch.org/), it is known for its **ease of use** and rapid deployment capabilities, making it a go-to choice for many developers and researchers.

### Architecture and Key Features

YOLOv5 features a CSPDarknet53 backbone, PANet path aggregation network, and a YOLOv5 detection head. It utilizes an anchor-based approach. The model is lauded for its efficient architecture that allows for fast training and inference. It's available in various sizes (n, s, m, l, x), offering scalability and adaptability to different hardware and performance requirements. Ultralytics provides extensive [documentation](https://docs.ultralytics.com/yolov5/) and a streamlined user experience, simplifying training and deployment.

### Strengths

- **Exceptional Speed:** YOLOv5 is highly optimized for speed, making it ideal for real-time object detection applications.
- **Ease of Use and Deployment:** Known for its simplicity, YOLOv5 is easy to train, deploy, and integrate. The Ultralytics ecosystem provides excellent support, tutorials, and pre-trained weights.
- **Mature Ecosystem:** Benefits from a large, active community and extensive resources, which is beneficial for troubleshooting and development. [Ultralytics HUB](https://www.ultralytics.com/hub) further enhances model management.
- **Training Efficiency:** Efficient training process with readily available [pre-trained checkpoints](https://github.com/ultralytics/yolov5/releases).

### Weaknesses

- **Lower Accuracy Compared to YOLOv8:** Generally, YOLOv5 models may have slightly lower accuracy compared to newer YOLOv8 variants, especially for complex tasks.
- **Limited Task Support:** Primarily focused on [object detection](https://www.ultralytics.com/glossary/object-detection), although segmentation capabilities were added later.

### Ideal Use Cases

YOLOv5 is optimally used in scenarios where speed and efficiency are paramount:

- **Real-time Video Surveillance:** Ideal for applications needing rapid object detection in video streams, like [security systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8).
- **Edge Computing Devices:** Smaller YOLOv5 models excel on devices with limited computational power like [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/) or [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/).
- **Mobile Applications:** Suitable for mobile apps where fast inference and smaller model sizes are crucial.

[Learn more about YOLOv5](https://docs.ultralytics.com/models/yolov5/){ .md-button }

## YOLOv8: The Cutting-Edge Solution

**Authors**: Glenn Jocher, Ayush Chaurasia, and Jing Qiu  
**Organization**: Ultralytics  
**Date**: 2023-01-10  
**GitHub Link**: <https://github.com/ultralytics/ultralytics>  
**Docs Link**: <https://docs.ultralytics.com/models/yolov8/>

[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/), introduced in January 2023, represents the latest iteration in the Ultralytics YOLO series at the time of its release. It is designed as a versatile framework supporting a full range of vision AI tasks, including detection, segmentation, classification, pose estimation, and oriented bounding boxes (OBB).

### Architecture and Key Features

YOLOv8 incorporates a new backbone network, a refined anchor-free detection head, and a novel loss function. Its flexible, modular architecture allows for easier adaptation and customization. This design contributes to improved accuracy and efficiency across various tasks. Ultralytics emphasizes **ease of use** with YOLOv8, providing clear [documentation](https://docs.ultralytics.com/models/yolov8/), a simple API, and integration with [Ultralytics HUB](https://www.ultralytics.com/hub) for streamlined workflows.

### Strengths

- **Improved Accuracy and Speed:** YOLOv8 generally offers better accuracy (mAP) and competitive speed compared to YOLOv5, achieving a strong **performance balance**.
- **Versatility:** Supports a wide array of vision tasks beyond object detection, making it a comprehensive tool.
- **User-Friendly Design:** Built with ease of use in mind, simplifying training, validation, and deployment via [Python](https://docs.ultralytics.com/usage/python/) or [CLI](https://docs.ultralytics.com/usage/cli/).
- **Active Development:** As a newer model within the **well-maintained Ultralytics ecosystem**, it benefits from ongoing updates and strong community support.

### Weaknesses

- **Computational Demand:** Larger YOLOv8 models require more computational resources, potentially limiting use in highly constrained environments.
- **Newer Architecture:** Being newer, the volume of third-party resources might be less extensive compared to the long-established YOLOv5, though Ultralytics' support is robust.

### Ideal Use Cases

YOLOv8 is well-suited for applications requiring high accuracy and versatility:

- **Advanced Robotics:** For complex object recognition and scene understanding.
- **High-Resolution Image Analysis:** Excels in detailed analysis where fine-grained detection is crucial.
- **Multi-Task Vision Systems:** Ideal for systems needing simultaneous detection, [segmentation](https://docs.ultralytics.com/tasks/segment/), or [pose estimation](https://docs.ultralytics.com/tasks/pose/).

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## Performance Comparison: YOLOv5 vs YOLOv8

The table below provides a detailed comparison of performance metrics for various YOLOv5 and YOLOv8 model sizes, evaluated on the [COCO dataset](https://docs.ultralytics.com/datasets/detect/coco/). YOLOv8 models generally show higher mAP<sup>val</sup> 50-95 scores compared to their YOLOv5 counterparts, indicating improved accuracy. While CPU inference times can be higher for YOLOv8, GPU speeds (TensorRT) remain highly competitive, showcasing the model's efficiency on optimized hardware. YOLOv8 achieves this higher accuracy often with comparable or even fewer parameters and FLOPs in some cases (e.g., YOLOv8l vs YOLOv5l).

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv5n | 640                   | 28.0                 | **73.6**                       | **1.12**                            | **2.6**            | **7.7**           |
| YOLOv5s | 640                   | 37.4                 | **120.7**                      | 1.92                                | 9.1                | 24.0              |
| YOLOv5m | 640                   | 45.4                 | **233.9**                      | 4.03                                | 25.1               | 64.2              |
| YOLOv5l | 640                   | 49.0                 | 408.4                          | 6.61                                | 53.2               | 135.0             |
| YOLOv5x | 640                   | 50.7                 | 763.2                          | 11.89                               | 97.2               | 246.4             |
|         |                       |                      |                                |                                     |                    |                   |
| YOLOv8n | 640                   | 37.3                 | 80.4                           | 1.47                                | 3.2                | 8.7               |
| YOLOv8s | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x | 640                   | **53.9**             | 479.1                          | 14.37                               | 68.2               | 257.8             |

## Training Methodologies

Both YOLOv5 and YOLOv8 leverage PyTorch for training and benefit from Ultralytics' streamlined training pipelines. Users can easily train models using the provided CLI or Python interfaces with minimal setup. Key advantages include:

- **Efficient Training:** Optimized training scripts and readily available [pre-trained weights](https://github.com/ultralytics/assets/releases) significantly reduce training time.
- **Ease of Use:** Comprehensive documentation ([YOLOv5 Docs](https://docs.ultralytics.com/models/yolov5/), [YOLOv8 Docs](https://docs.ultralytics.com/models/yolov8/)) and simple APIs make custom training straightforward.
- **Data Augmentation:** Both models incorporate effective [data augmentation](https://www.ultralytics.com/glossary/data-augmentation) techniques.
- **Lower Memory Requirements:** Compared to architectures like Transformers, Ultralytics YOLO models typically require less CUDA memory during training and inference, making them accessible on a wider range of hardware.
- **Ultralytics Ecosystem:** Integration with tools like [Ultralytics HUB](https://www.ultralytics.com/hub) and logging platforms ([TensorBoard](https://docs.ultralytics.com/integrations/tensorboard/), [Comet](https://docs.ultralytics.com/integrations/comet/)) simplifies experiment tracking and management.

## Conclusion

Both YOLOv5 and YOLOv8 are powerful object detection models developed by Ultralytics, offering excellent performance and ease of use.

- **YOLOv5** remains a strong contender, particularly for applications prioritizing maximum inference speed and leveraging its mature ecosystem. It's an excellent choice for deployment on resource-constrained devices.
- **YOLOv8** represents the next generation, offering superior accuracy and enhanced versatility across multiple vision tasks (detection, segmentation, pose, etc.). Its anchor-free architecture and advanced features make it ideal for new projects seeking state-of-the-art performance and flexibility.

Ultralytics continues to innovate, ensuring both models are well-supported, easy to use, and provide a great balance of speed and accuracy suitable for diverse real-world scenarios.

For users exploring other options, Ultralytics also offers models like [YOLOv9](https://docs.ultralytics.com/models/yolov9/), [YOLOv10](https://docs.ultralytics.com/models/yolov10/), and the latest [YOLO11](https://docs.ultralytics.com/models/yolo11/), each providing unique advantages. Further comparisons are available in the [Ultralytics documentation](https://docs.ultralytics.com/compare/).
