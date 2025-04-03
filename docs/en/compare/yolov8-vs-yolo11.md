---
comments: true
description: Compare YOLOv8 and YOLO11 for object detection. Explore their performance, architecture, and best-use cases to find the right model for your needs.
keywords: YOLOv8, YOLO11, object detection, Ultralytics, YOLO comparison, machine learning, computer vision, inference speed, model accuracy
---

# YOLOv8 vs YOLO11: A Technical Comparison

Comparing Ultralytics YOLOv8 and the newer Ultralytics YOLO11 reveals the rapid evolution within the YOLO family for real-time object detection and other computer vision tasks. Both models, developed by Ultralytics, offer state-of-the-art performance but present distinct advantages depending on the specific application needs. This page provides an in-depth technical comparison, analyzing their architectures, performance metrics, and ideal use cases to guide developers and researchers.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv8", "YOLO11"]'></canvas>

## Ultralytics YOLOv8

**Authors:** Glenn Jocher, Ayush Chaurasia, and Jing Qiu  
**Organization:** Ultralytics  
**Date:** 2023-01-10  
**GitHub:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)  
**Docs:** [https://docs.ultralytics.com/models/yolov8/](https://docs.ultralytics.com/models/yolov8/)

Ultralytics YOLOv8 marked a significant step forward in the YOLO series, establishing itself as a highly versatile and powerful framework. It builds upon previous YOLO successes, introducing architectural refinements like a refined CSPDarknet backbone, a C2f neck for better feature fusion, and an anchor-free, decoupled head. This design enhances both performance and flexibility across a wide range of vision AI tasks, including [object detection](https://docs.ultralytics.com/tasks/detect/), [instance segmentation](https://docs.ultralytics.com/tasks/segment/), [image classification](https://docs.ultralytics.com/tasks/classify/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), and oriented bounding boxes (OBB).

**Performance Metrics:**
YOLOv8 delivers excellent performance, balancing speed and accuracy. The YOLOv8x model achieves 53.9% mAP<sup>val</sup>50-95 on the COCO dataset, while the lightweight YOLOv8n reaches 37.3% mAP<sup>val</sup>50-95, suitable for edge devices. Detailed benchmarks are available in the [YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/#performance-metrics).

**Strengths:**

- **Versatility:** Supports multiple vision tasks within a single, unified framework.
- **Performance Balance:** Achieves a strong trade-off between inference speed and accuracy, suitable for diverse real-world deployments.
- **Ease of Use:** Offers a streamlined user experience through a simple [Python API](https://docs.ultralytics.com/usage/python/) and [CLI](https://docs.ultralytics.com/usage/cli/), backed by extensive [documentation](https://docs.ultralytics.com/).
- **Well-Maintained Ecosystem:** Benefits from continuous development, a strong open-source community, frequent updates, and integration with [Ultralytics HUB](https://docs.ultralytics.com/hub/) for MLOps workflows.
- **Training Efficiency:** Efficient training processes with readily available pre-trained weights simplify custom model development.
- **Memory Efficiency:** Requires less CUDA memory for training and inference compared to many other architectures, especially transformer-based models.

**Weaknesses:**

- Larger models (e.g., YOLOv8x) demand significant computational resources.
- May require further optimization for deployment on extremely resource-constrained hardware.

**Ideal Use Cases:**
YOLOv8 is ideal for applications demanding high accuracy and the flexibility to handle multiple tasks, such as advanced [robotics](https://www.ultralytics.com/glossary/robotics), [security systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8), [smart city](https://www.ultralytics.com/blog/computer-vision-ai-in-smart-cities) infrastructure, and projects requiring rapid development cycles.

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## Ultralytics YOLO11

**Authors:** Glenn Jocher and Jing Qiu  
**Organization:** Ultralytics  
**Date:** 2024-09-27  
**GitHub:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)  
**Docs:** [https://docs.ultralytics.com/models/yolo11/](https://docs.ultralytics.com/models/yolo11/)

YOLO11 represents the latest evolution in the Ultralytics YOLO series, engineered for enhanced efficiency and accuracy. Building on the robust foundation of YOLOv8, YOLO11 introduces further architectural refinements, particularly focusing on optimizing the backbone, neck, and head structures to reduce parameters and computational cost (FLOPs) while boosting performance. It maintains the anchor-free approach and supports the same wide range of vision tasks as YOLOv8, making it a cutting-edge choice for demanding applications.

**Performance Metrics:**
YOLO11 demonstrates improved performance, especially in terms of speed and efficiency. YOLO11n achieves 39.5% mAP<sup>val</sup>50-95 with fewer parameters (2.6M vs. 3.2M) and faster CPU inference (56.1ms vs. 80.4ms) compared to YOLOv8n. The larger YOLO11x model pushes accuracy further to 54.7% mAP<sup>val</sup>50-95 while being more computationally efficient than YOLOv8x (194.9 BFLOPs vs. 257.8 BFLOPs). See the [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/#performance-metrics) for full details.

**Strengths:**

- **Superior Speed and Efficiency:** Offers faster inference speeds, particularly on CPU, and reduced computational load (FLOPs) compared to YOLOv8.
- **State-of-the-Art Accuracy:** Achieves higher mAP scores across various model sizes.
- **Optimized Architecture:** Refined design makes it highly suitable for deployment on [edge devices](https://www.ultralytics.com/glossary/edge-ai) and resource-constrained environments.
- **Latest Advancements:** Incorporates the most recent research and engineering improvements from Ultralytics.
- **Ease of Use & Ecosystem:** Inherits the user-friendly nature and comprehensive ecosystem of Ultralytics, including extensive documentation, simple APIs, and [Ultralytics HUB](https://docs.ultralytics.com/hub/) integration.
- **Versatility & Efficiency:** Supports multiple tasks efficiently with lower memory requirements than many alternatives.

**Weaknesses:**

- As a newer model, the community knowledge base and third-party integrations might still be growing compared to the more established YOLOv8.
- Like other one-stage detectors, may face challenges detecting extremely small objects compared to specialized two-stage detectors.

**Ideal Use Cases:**
YOLO11 excels in scenarios where inference speed and efficiency are critical without compromising accuracy. This includes real-time video analysis, [robotics](https://www.ultralytics.com/blog/integrating-computer-vision-in-robotics-with-ultalytics-yolo11), [autonomous systems](https://www.ultralytics.com/solutions/ai-in-automotive), and deployment on hardware with limited computational power, such as [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/) or [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/).

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

## Performance Comparison

The table below provides a detailed comparison of object detection performance on the COCO val2017 dataset for various YOLOv8 and YOLO11 model sizes. YOLO11 generally shows improved mAP and faster CPU speeds with fewer parameters and FLOPs compared to its YOLOv8 counterparts.

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv8n | 640                   | 37.3                 | 80.4                           | **1.47**                            | 3.2                | 8.7               |
| YOLOv8s | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |
|         |                       |                      |                                |                                     |                    |                   |
| YOLO11n | 640                   | **39.5**             | **56.1**                       | 1.5                                 | **2.6**            | **6.5**           |
| YOLO11s | 640                   | **47.0**             | **90.0**                       | **2.5**                             | **9.4**            | **21.5**          |
| YOLO11m | 640                   | **51.5**             | **183.2**                      | **4.7**                             | **20.1**           | **68.0**          |
| YOLO11l | 640                   | **53.4**             | **238.6**                      | **6.2**                             | **25.3**           | **86.9**          |
| YOLO11x | 640                   | **54.7**             | **462.8**                      | **11.3**                            | **56.9**           | **194.9**         |

## Other Models

Ultralytics offers a wide range of models catering to different needs. Besides YOLOv8 and YOLO11, users might explore:

- [Ultralytics YOLOv5](https://docs.ultralytics.com/models/yolov5/): A highly popular and stable predecessor, known for its reliability and large community support.
- [Ultralytics YOLOv9](https://docs.ultralytics.com/models/yolov9/): Introduced innovations like Programmable Gradient Information (PGI).
- [Ultralytics YOLOv10](https://docs.ultralytics.com/models/yolov10/): Focused on further efficiency improvements, particularly eliminating NMS post-processing.

Comparisons with other state-of-the-art models like [RT-DETR](https://docs.ultralytics.com/models/rtdetr/), [DAMO-YOLO](https://docs.ultralytics.com/compare/damo-yolo-vs-yolov8/), and [PP-YOLOE](https://docs.ultralytics.com/compare/pp-yoloe-vs-yolov9/) are also available within the [Ultralytics documentation](https://docs.ultralytics.com/compare/) to help select the optimal model for any computer vision challenge.
