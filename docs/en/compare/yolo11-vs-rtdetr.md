---
comments: true
description: Compare RTDETRv2's accuracy with YOLO11's speed in this detailed analysis of top object detection models. Decide the best fit for your projects.
keywords: RTDETRv2, YOLO11, object detection, Ultralytics, Vision Transformer, YOLO, computer vision, real-time detection, model comparison
---

# YOLO11 vs RTDETRv2: A Technical Comparison for Object Detection

Choosing the right object detection model is crucial for computer vision projects. Ultralytics offers a range of models, including the highly efficient Ultralytics YOLO series, while other research groups contribute models like RT-DETR. This page provides a detailed technical comparison between **Ultralytics YOLO11** and **RTDETRv2**, analyzing their architectures, performance, and ideal use cases to help you make an informed decision.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO11", "RTDETRv2"]'></canvas>

## Ultralytics YOLO11: Efficient and Versatile Object Detection

**Ultralytics YOLO11** is the latest iteration in the renowned YOLO series, developed by Glenn Jocher and Jing Qiu at Ultralytics and released on 2024-09-27. It represents the cutting edge in real-time object detection, balancing speed and accuracy effectively.

- **Authors:** Glenn Jocher, Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com/)
- **Date:** 2024-09-27
- **GitHub:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Docs:** [https://docs.ultralytics.com/models/yolo11/](https://docs.ultralytics.com/models/yolo11/)

### Architecture and Key Features

YOLO11 builds upon the successful single-stage architecture of its predecessors like [YOLOv8](https://docs.ultralytics.com/models/yolov8/), incorporating refinements for enhanced performance. It is designed for **efficiency**, making it suitable for deployment across various hardware, including resource-constrained [edge devices](https://www.ultralytics.com/glossary/edge-ai).

A key advantage of YOLO11 lies within the **Ultralytics ecosystem**. It offers a streamlined user experience with a simple API, extensive [documentation](https://docs.ultralytics.com/), and readily available pre-trained weights, facilitating **efficient training** and deployment. The ecosystem benefits from active development, strong community support, and frequent updates. Furthermore, YOLO11 is highly **versatile**, supporting tasks like detection, segmentation, classification, pose estimation, and OBB out-of-the-box, a breadth of capability often lacking in competing models. Compared to transformer-based models like RTDETRv2, YOLO11 typically requires significantly **less memory** during training and inference.

### Strengths and Weaknesses

**Strengths:**

- **Exceptional Speed:** Optimized for real-time inference on both CPU and GPU.
- **High Efficiency:** Lower memory and computational requirements compared to transformer models.
- **Performance Balance:** Excellent trade-off between speed and accuracy across various model sizes.
- **Versatility:** Supports multiple vision tasks within a single framework.
- **Ease of Use:** Simple API, comprehensive documentation, and integrated ecosystem ([Ultralytics HUB](https://www.ultralytics.com/hub)).
- **Training Efficiency:** Faster training times and lower resource needs.

**Weaknesses:**

- Smaller variants (like YOLO11n) prioritize speed, potentially offering slightly lower accuracy than the largest RTDETRv2 models in complex scenarios.

### Ideal Use Cases

YOLO11 excels in applications demanding real-time performance and efficiency, such as [autonomous vehicles](https://www.ultralytics.com/solutions/ai-in-automotive), robotics, [security systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8), and deployment on edge devices. Its versatility makes it suitable for projects requiring multiple vision tasks.

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

## RTDETRv2: High Accuracy Real-Time Detection

**RTDETRv2** (Real-Time Detection Transformer v2) is an object detection model developed by researchers at Baidu, released on 2023-04-17. It leverages a transformer architecture to achieve high accuracy.

- **Authors:** Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, Yi Liu
- **Organization:** Baidu
- **Date:** 2023-04-17
- **Arxiv:** [https://arxiv.org/abs/2304.08069](https://arxiv.org/abs/2304.08069)
- **GitHub:** [https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)
- **Docs:** [https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme)

### Architecture and Key Features

RTDETRv2 utilizes a [Vision Transformer (ViT)](https://www.ultralytics.com/glossary/vision-transformer-vit) backbone combined with CNN features. The transformer allows it to capture global image context using [self-attention mechanisms](https://www.ultralytics.com/glossary/self-attention), aiming for high detection accuracy, particularly in complex scenes.

### Strengths and Weaknesses

**Strengths:**

- **High Accuracy:** Achieves strong mAP scores, especially larger variants on complex datasets.
- **Global Context Understanding:** Transformer architecture effectively models long-range dependencies.

**Weaknesses:**

- **Higher Complexity:** Transformer models are generally more complex and computationally intensive.
- **Larger Model Size:** Higher parameter counts and FLOPs compared to equivalent YOLO11 models.
- **Slower Inference:** Significantly slower on CPU (metrics often unreported) and generally slower on GPU compared to YOLO11, especially smaller variants.
- **Higher Memory Requirements:** Training and inference demand more memory resources.
- **Limited Versatility:** Primarily focused on object detection, lacking built-in support for other tasks like segmentation or pose estimation found in YOLO11.
- **Less Mature Ecosystem:** May lack the extensive documentation, community support, and ease-of-use features of the Ultralytics ecosystem.

### Ideal Use Cases

RTDETRv2 is suited for applications where achieving the absolute highest accuracy is the primary goal, and sufficient computational resources (especially powerful GPUs) are available. Examples include detailed [medical image analysis](https://www.ultralytics.com/glossary/medical-image-analysis) or high-resolution [satellite imagery analysis](https://www.ultralytics.com/blog/using-computer-vision-to-analyse-satellite-imagery).

[Learn more about RTDETRv2](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme){ .md-button }

## Performance Comparison: YOLO11 vs RTDETRv2

The table below compares various YOLO11 and RTDETRv2 models based on performance metrics on the COCO dataset.

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLO11n    | 640                   | 39.5                 | **56.1**                       | **1.5**                             | **2.6**            | **6.5**           |
| YOLO11s    | 640                   | 47.0                 | **90.0**                       | **2.5**                             | **9.4**            | **21.5**          |
| YOLO11m    | 640                   | 51.5                 | **183.2**                      | **4.7**                             | 20.1               | 68.0              |
| YOLO11l    | 640                   | 53.4                 | **238.6**                      | 6.2                                 | 25.3               | 86.9              |
| YOLO11x    | 640                   | **54.7**             | **462.8**                      | 11.3                                | 56.9               | 194.9             |
|            |                       |                      |                                |                                     |                    |                   |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | 54.3                 | -                              | 15.03                               | **76**             | **259**           |

**Analysis:** Ultralytics YOLO11 demonstrates superior speed, particularly on CPU where RTDETRv2 lacks reported metrics, and significantly faster TensorRT inference times, especially for smaller models (YOLO11n/s). YOLO11x achieves a higher mAP than RTDETRv2-x while being faster on TensorRT and more efficient (fewer FLOPs/params). Overall, YOLO11 offers a better balance of speed, accuracy, and efficiency, making it more practical for diverse deployment scenarios.

## Conclusion

Both Ultralytics YOLO11 and RTDETRv2 are powerful object detection models, but they cater to different priorities.

**Ultralytics YOLO11** is highly recommended for most users due to its exceptional **speed**, **efficiency**, **versatility**, and **ease of use**. Its well-maintained ecosystem, lower memory requirements, and excellent performance balance make it ideal for real-world applications, from edge devices to cloud servers.

**RTDETRv2** offers high accuracy, particularly in complex scenes, but comes with higher computational costs, slower inference speeds, and greater memory demands. It's a viable option for niche applications where maximum accuracy is paramount and ample resources are available, but users should be prepared for its increased complexity and resource intensity compared to the streamlined experience offered by YOLO11.

For users exploring other options, Ultralytics provides a rich model zoo including [YOLOv10](https://docs.ultralytics.com/models/yolov10/), [YOLOv9](https://docs.ultralytics.com/models/yolov9/), [YOLOv8](https://docs.ultralytics.com/models/yolov8/), and specialized models like [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/) and [YOLO-World](https://docs.ultralytics.com/models/yolo-world/).
