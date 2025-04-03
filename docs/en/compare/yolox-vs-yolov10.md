---
comments: true
description: Compare YOLOv10 and YOLOX for object detection. Explore architecture, benchmarks, and use cases to choose the best real-time detection model for your needs.
keywords: YOLOv10, YOLOX, object detection, Ultralytics, real-time, model comparison, benchmark, computer vision, deep learning, AI
---

# YOLOX vs YOLOv10: Technical Comparison

Selecting the optimal object detection model is essential for balancing accuracy, speed, and computational demands in computer vision projects. This page provides a detailed technical comparison between **YOLOX** and **YOLOv10**, two significant models in the object detection landscape. We will analyze their architectures, performance metrics, and ideal use cases to help you choose the best fit for your needs.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOX", "YOLOv10"]'></canvas>

## YOLOX: High-Performance Anchor-Free Detector

YOLOX is an anchor-free object detection model developed by Megvii, aiming to simplify the YOLO design while achieving high performance. It was introduced as an alternative approach within the YOLO family, focusing on bridging research and industrial application needs.

**Technical Details:**

- **Authors:** Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun
- **Organization:** Megvii
- **Date:** 2021-07-18
- **Arxiv Link:** <https://arxiv.org/abs/2107.08430>
- **GitHub Link:** <https://github.com/Megvii-BaseDetection/YOLOX>
- **Docs Link:** <https://yolox.readthedocs.io/en/latest/>

### Architecture and Key Features

YOLOX implements several key architectural changes compared to earlier YOLO models:

- **Anchor-Free Design:** Eliminates predefined [anchor boxes](https://www.ultralytics.com/glossary/anchor-based-detectors), simplifying the detection pipeline and reducing hyperparameters, which can improve generalization.
- **Decoupled Head:** Uses separate heads for classification and localization tasks, potentially improving convergence speed and accuracy compared to coupled heads used in some earlier models.
- **Advanced Training Strategies:** Incorporates techniques like SimOTA (Simplified Optimal Transport Assignment) for dynamic label assignment and strong [data augmentation](https://docs.ultralytics.com/integrations/albumentations/) methods like MixUp.

### Performance Metrics

YOLOX models offer a competitive balance between accuracy ([mAP](https://docs.ultralytics.com/guides/yolo-performance-metrics/)) and speed, with various model sizes available (Nano, Tiny, S, M, L, X) to cater to different resource constraints. See the table below for specific metrics.

### Strengths and Weaknesses

**Strengths:**

- **High Accuracy:** Achieves strong mAP scores, especially with larger variants like YOLOX-x.
- **Anchor-Free Simplicity:** Reduces complexity related to anchor box configuration.
- **Established Model:** Has been available since 2021, with community resources and deployment examples.

**Weaknesses:**

- **Inference Speed:** While efficient, it can be slower than highly optimized models like YOLOv10, particularly the smaller variants.
- **External Ecosystem:** Not natively integrated into the Ultralytics ecosystem, potentially requiring more effort for deployment and integration with tools like [Ultralytics HUB](https://www.ultralytics.com/hub).
- **Task Versatility:** Primarily focused on object detection, lacking built-in support for other tasks like segmentation or pose estimation found in newer Ultralytics models.

### Use Cases

YOLOX is suitable for:

- **General Object Detection:** Applications needing a solid balance between accuracy and speed, such as [security systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8).
- **Research:** As a baseline for exploring anchor-free detection methods.
- **Industrial Applications:** Tasks like [quality control](https://www.ultralytics.com/solutions/ai-in-manufacturing) where accuracy is critical.

[Learn more about YOLOX](https://yolox.readthedocs.io/en/latest/){ .md-button }

## YOLOv10: Cutting-Edge Real-Time End-to-End Detector

[Ultralytics YOLOv10](https://docs.ultralytics.com/models/yolov10/), developed by researchers at Tsinghua University, represents a significant advancement in real-time object detection by focusing on end-to-end efficiency. It addresses post-processing bottlenecks and optimizes the architecture for superior speed and performance.

**Technical Details:**

- **Authors:** Ao Wang, Hui Chen, Lihao Liu, et al.
- **Organization:** Tsinghua University
- **Date:** 2024-05-23
- **Arxiv Link:** <https://arxiv.org/abs/2405.14458>
- **GitHub Link:** <https://github.com/THU-MIG/yolov10>
- **Docs Link:** <https://docs.ultralytics.com/models/yolov10/>

### Architecture and Key Features

YOLOv10 introduces innovations for enhanced efficiency:

- **NMS-Free Training:** Employs consistent dual assignments to eliminate the need for [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) during inference, reducing latency and enabling true end-to-end deployment.
- **Holistic Efficiency-Accuracy Design:** Optimizes various model components comprehensively, reducing computational redundancy and enhancing capability.
- **Lightweight Architecture:** Focuses on parameter efficiency and FLOPs reduction, leading to faster inference speeds suitable for diverse hardware, including [edge devices](https://www.ultralytics.com/glossary/edge-ai).

### Performance Metrics

YOLOv10 excels in speed and efficiency while maintaining high accuracy. As shown in the table, YOLOv10 models often achieve better latency and parameter efficiency compared to YOLOX models at similar mAP levels. For instance, YOLOv10-S matches YOLOX-m's mAP with significantly fewer parameters and faster speed.

### Strengths and Weaknesses

**Strengths:**

- **Exceptional Speed and Efficiency:** Optimized for real-time, low-latency inference.
- **NMS-Free Inference:** Simplifies deployment and speeds up post-processing.
- **State-of-the-Art Performance:** Achieves excellent mAP scores across various model scales (n, s, m, b, l, x).
- **Ultralytics Ecosystem Integration:** Benefits from seamless integration with the Ultralytics [Python package](https://docs.ultralytics.com/usage/python/), extensive [documentation](https://docs.ultralytics.com/), and active maintenance.
- **Ease of Use:** Offers a streamlined user experience typical of Ultralytics models.
- **Training Efficiency:** Efficient training process with readily available pre-trained weights and lower memory requirements compared to many alternatives.

**Weaknesses:**

- **Relatively Newer:** As a more recent model, the breadth of community examples might still be growing compared to older models like YOLOX.

### Use Cases

YOLOv10 is ideal for demanding real-time applications:

- **Edge AI:** Deployment on resource-constrained devices like [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/) and [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/).
- **Real-Time Systems:** Autonomous vehicles, [robotics](https://www.ultralytics.com/glossary/robotics), high-speed video analytics, and [surveillance](https://www.ultralytics.com/blog/shattering-the-surveillance-status-quo-with-vision-ai).
- **High-Throughput Processing:** Industrial inspection and applications requiring rapid analysis.

[Learn more about YOLOv10](https://docs.ultralytics.com/models/yolov10/){ .md-button }

## Performance Comparison: YOLOX vs YOLOv10

| Model     | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| --------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOXnano | 416                   | 25.8                 | -                              | -                                   | **0.91**           | **1.08**          |
| YOLOXtiny | 416                   | 32.8                 | -                              | -                                   | 5.06               | 6.45              |
| YOLOXs    | 640                   | 40.5                 | -                              | 2.56                                | 9.0                | 26.8              |
| YOLOXm    | 640                   | 46.9                 | -                              | 5.43                                | 25.3               | 73.8              |
| YOLOXl    | 640                   | 49.7                 | -                              | 9.04                                | 54.2               | 155.6             |
| YOLOXx    | 640                   | 51.1                 | -                              | 16.1                                | 99.1               | 281.9             |
|           |                       |                      |                                |                                     |                    |                   |
| YOLOv10n  | 640                   | 39.5                 | -                              | **1.56**                            | 2.3                | 6.7               |
| YOLOv10s  | 640                   | 46.7                 | -                              | 2.66                                | 7.2                | 21.6              |
| YOLOv10m  | 640                   | 51.3                 | -                              | 5.48                                | 15.4               | 59.1              |
| YOLOv10b  | 640                   | 52.7                 | -                              | 6.54                                | 24.4               | 92.0              |
| YOLOv10l  | 640                   | 53.3                 | -                              | 8.33                                | 29.5               | 120.3             |
| YOLOv10x  | 640                   | **54.4**             | -                              | 12.2                                | 56.9               | 160.4             |

The table highlights YOLOv10's advancements in efficiency. YOLOv10 models generally offer lower latency (faster T4 TensorRT speed) and require fewer parameters and FLOPs compared to YOLOX models at similar or even better mAP levels. For example, YOLOv10-L achieves a higher mAP (53.3 vs 49.7) than YOLOX-l with significantly fewer parameters (29.5M vs 54.2M) and FLOPs (120.3B vs 155.6B), along with faster inference speed. YOLOv10x pushes the state-of-the-art mAP further while remaining more efficient than YOLOXx. This improved performance-efficiency balance makes YOLOv10 a compelling choice, especially when leveraging the streamlined [Ultralytics ecosystem](https://docs.ultralytics.com/) for training and deployment.

## Conclusion

Both YOLOX and YOLOv10 are strong object detection models. YOLOX provides a reliable anchor-free approach with good performance. However, YOLOv10 demonstrates superior efficiency, achieving state-of-the-art speed and accuracy trade-offs, particularly due to its NMS-free design and architectural optimizations. For developers seeking the best real-time performance, lower computational cost, and seamless integration within a well-maintained ecosystem, **YOLOv10 is generally the recommended choice**. Its integration with Ultralytics tools simplifies workflows from training to deployment.

For further exploration, consider comparing these models with other state-of-the-art options like [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/), [YOLOv9](https://docs.ultralytics.com/models/yolov9/), and the latest [YOLO11](https://docs.ultralytics.com/models/yolo11/). You might also find comparisons like [YOLOv10 vs YOLO11](https://docs.ultralytics.com/compare/yolo11-vs-yolov10/) and [YOLOX vs YOLOv8](https://docs.ultralytics.com/compare/yolov8-vs-yolox/) informative.
