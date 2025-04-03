---
comments: true
description: Discover key differences between YOLOv8 and YOLOv5. Compare speed, accuracy, use cases, and more to choose the ideal model for your computer vision needs.
keywords: YOLOv8, YOLOv5, object detection, YOLO comparison, computer vision, model comparison, speed, accuracy, Ultralytics, deep learning
---

# YOLOv8 vs YOLOv5: A Detailed Comparison

Comparing Ultralytics YOLOv8 and YOLOv5 for object detection reveals significant advancements and distinct strengths in each model. Both models, developed by Ultralytics, are renowned for their speed and accuracy, but cater to different user needs and priorities in the field of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv). This page provides a technical comparison to help users make informed decisions based on their project requirements, highlighting the advantages of the Ultralytics ecosystem.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv8", "YOLOv5"]'></canvas>

## YOLOv8: The Cutting-Edge Solution

**Authors**: Glenn Jocher, Ayush Chaurasia, and Jing Qiu  
**Organization**: Ultralytics  
**Date**: 2023-01-10  
**GitHub Link**: <https://github.com/ultralytics/ultralytics>  
**Docs Link**: <https://docs.ultralytics.com/models/yolov8/>

[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) represents the latest iteration in the YOLO series, designed as a versatile framework supporting a full range of vision AI tasks, including [object detection](https://docs.ultralytics.com/tasks/detect/), [instance segmentation](https://docs.ultralytics.com/tasks/segment/), [image classification](https://docs.ultralytics.com/tasks/classify/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), and oriented bounding boxes (OBB). YOLOv8 incorporates new architectural features like an anchor-free detection head and a refined C2f neck for enhanced performance and flexibility.

**Strengths:**

- **Improved Accuracy and Speed:** YOLOv8 generally offers better accuracy (higher mAP) with competitive speed compared to YOLOv5, especially in larger model variants, providing a strong performance balance.
- **Versatility:** Supports a wide array of vision tasks beyond object detection, making it a comprehensive tool for diverse applications within a single, unified framework.
- **Ease of Use:** Ultralytics emphasizes a streamlined user experience with YOLOv8, providing a simple [Python API](https://docs.ultralytics.com/usage/python/) and [CLI](https://docs.ultralytics.com/usage/cli/), extensive [documentation](https://docs.ultralytics.com/models/yolov8/), and readily available pre-trained weights for efficient training and deployment.
- **Well-Maintained Ecosystem:** As the latest model, YOLOv8 benefits from active development, frequent updates, strong community support via [GitHub](https://github.com/ultralytics/ultralytics) and [Discord](https://discord.com/invite/ultralytics), and integration with [Ultralytics HUB](https://www.ultralytics.com/hub) for streamlined model management and [MLOps](https://www.ultralytics.com/glossary/machine-learning-operations-mlops).
- **Memory Efficiency:** Optimized for lower memory usage during training and inference compared to many other architectures, particularly transformer-based models.

**Weaknesses:**

- **Computational Demand:** Larger YOLOv8 models require more computational resources, which might be a limitation for severely resource-constrained environments.

**Ideal Use Cases:**
YOLOv8 is well-suited for applications requiring high accuracy and versatility:

- **Advanced Robotics:** For complex object recognition and scene understanding.
- **High-Resolution Image Analysis:** Excels in detailed analysis where fine-grained detection is crucial.
- **Multi-Task Vision Systems:** Ideal for systems needing simultaneous detection, segmentation, and pose estimation.

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## YOLOv5: The Established Industry Standard

**Authors**: Glenn Jocher  
**Organization**: Ultralytics  
**Date**: 2020-06-26  
**GitHub Link**: <https://github.com/ultralytics/yolov5>  
**Docs Link**: <https://docs.ultralytics.com/models/yolov5/>

[Ultralytics YOLOv5](https://docs.ultralytics.com/models/yolov5/) quickly became an industry favorite after its release due to its exceptional balance of speed and accuracy. Built on [PyTorch](https://pytorch.org/), it's known for its ease of use and rapid deployment capabilities, featuring a CSPDarknet53 backbone and PANet neck.

**Strengths:**

- **Exceptional Speed:** YOLOv5 is highly optimized for speed, making it suitable for real-time applications.
- **Ease of Use and Deployment:** Known for its simplicity, YOLOv5 is easy to train and deploy, supported by extensive tutorials and documentation within the Ultralytics ecosystem.
- **Mature Ecosystem:** Benefits from a large, active community, offering abundant resources, pre-trained models, and support.
- **Versatile Model Sizes:** Offers various sizes (n, s, m, l, x) for scalability across different hardware.
- **Training Efficiency:** Efficient training process with readily available pre-trained weights and lower memory requirements compared to many alternatives.

**Weaknesses:**

- **Lower Accuracy Compared to YOLOv8:** Generally, YOLOv5 models have slightly lower mAP compared to corresponding YOLOv8 models.
- **Limited Task Support Initially:** Primarily focused on object detection, although segmentation capabilities were added later.

**Ideal Use Cases:**
YOLOv5 excels where speed and efficiency are paramount:

- **Real-time Video Surveillance:** Ideal for rapid object detection in video streams.
- **Edge Computing Devices:** Smaller YOLOv5 models are excellent for deployment on devices with limited computational power like [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/).
- **Mobile Applications:** Suitable where fast inference and smaller model sizes are crucial.

[Learn more about YOLOv5](https://docs.ultralytics.com/models/yolov5/){ .md-button }

## Performance Comparison

The table below provides a detailed comparison of performance metrics for various YOLOv8 and YOLOv5 models on the COCO val2017 dataset. YOLOv8 models generally achieve higher mAP<sup>val</sup> scores than their YOLOv5 counterparts with comparable or better speeds on GPU, showcasing the advancements in the newer architecture. YOLOv5 maintains an edge in CPU inference speed for some smaller models.

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv8n | 640                   | 37.3                 | 80.4                           | 1.47                                | 3.2                | 8.7               |
| YOLOv8s | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x | 640                   | **53.9**             | 479.1                          | 14.37                               | 68.2               | 257.8             |
|         |                       |                      |                                |                                     |                    |                   |
| YOLOv5n | 640                   | 28.0                 | **73.6**                       | **1.12**                            | **2.6**            | **7.7**           |
| YOLOv5s | 640                   | 37.4                 | 120.7                          | 1.92                                | 9.1                | 24.0              |
| YOLOv5m | 640                   | 45.4                 | 233.9                          | 4.03                                | 25.1               | 64.2              |
| YOLOv5l | 640                   | 49.0                 | 408.4                          | 6.61                                | 53.2               | 135.0             |
| YOLOv5x | 640                   | 50.7                 | 763.2                          | 11.89                               | 97.2               | 246.4             |

## Other Ultralytics Models

Beyond YOLOv8 and YOLOv5, Ultralytics offers a range of state-of-the-art models. Consider exploring [YOLOv7](https://docs.ultralytics.com/models/yolov7/), [YOLOv9](https://docs.ultralytics.com/models/yolov9/), [YOLOv10](https://docs.ultralytics.com/models/yolov10/), and the latest [YOLO11](https://docs.ultralytics.com/models/yolo11/) for different performance characteristics and features. Each model benefits from the robust Ultralytics ecosystem, ensuring ease of use, efficient training, and deployment flexibility.
