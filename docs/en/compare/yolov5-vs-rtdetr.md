---
comments: true
description: Compare YOLOv5 and RTDETRv2 for object detection. Explore their architectures, performance metrics, strengths, and best use cases in computer vision.
keywords: YOLOv5, RTDETRv2, object detection, model comparison, Ultralytics, computer vision, machine learning, real-time detection, Vision Transformers, AI models
---

# YOLOv5 vs RTDETRv2: A Detailed Model Comparison

Choosing the right object detection model is crucial for computer vision projects. Ultralytics offers a suite of models tailored for various needs, including the highly efficient [Ultralytics YOLOv5](https://docs.ultralytics.com/models/yolov5/) and comparing it against other architectures like RTDETRv2. This page provides a technical comparison between YOLOv5 and RTDETRv2, highlighting their architectural differences, performance metrics, training methodologies, and ideal applications.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv5", "RTDETRv2"]'></canvas>

## YOLOv5: Speed and Efficiency

**Author:** Glenn Jocher  
**Organization:** Ultralytics  
**Date:** 2020-06-26  
**GitHub:** [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)  
**Docs:** [https://docs.ultralytics.com/models/yolov5/](https://docs.ultralytics.com/models/yolov5/)

Ultralytics YOLOv5 is a widely adopted one-stage object detector celebrated for its exceptional **inference speed** and **operational efficiency**. Developed by Ultralytics, it has become a benchmark for real-time object detection tasks.

### Architecture

YOLOv5 employs a CNN-based architecture optimized for speed:

- **Backbone:** CSPDarknet53 for efficient feature extraction.
- **Neck:** PANet for effective feature fusion across scales.
- **Head:** YOLOv5 detection head for bounding box prediction and classification.
  It is available in multiple sizes (n, s, m, l, x), allowing users to select the best trade-off between speed and accuracy for their specific needs.

### Strengths

YOLOv5 offers significant advantages, particularly for developers seeking practical deployment:

- **Ease of Use:** Features a streamlined user experience with a simple API, extensive [documentation](https://docs.ultralytics.com/models/yolov5/), and numerous [tutorials](https://docs.ultralytics.com/yolov5/#tutorials).
- **Well-Maintained Ecosystem:** Benefits from the integrated [Ultralytics ecosystem](https://docs.ultralytics.com/), including active development, strong community support via [GitHub](https://github.com/ultralytics/yolov5) and Discord, frequent updates, and platforms like [Ultralytics HUB](https://www.ultralytics.com/hub) for no-code training and deployment.
- **Performance Balance:** Achieves a strong balance between inference speed and accuracy, making it suitable for diverse real-world scenarios.
- **Memory Requirements:** Typically requires lower memory (especially CUDA memory during training) compared to transformer-based models like RTDETRv2.
- **Training Efficiency:** Offers efficient training processes, faster convergence, and readily available [pre-trained weights](https://github.com/ultralytics/yolov5#pretrained-checkpoints) on datasets like [COCO](https://docs.ultralytics.com/datasets/detect/coco/).
- **Versatility:** While primarily focused on detection, the YOLOv5 repository also supports [instance segmentation](https://docs.ultralytics.com/tasks/segment/) and [image classification](https://docs.ultralytics.com/tasks/classify/) tasks.

### Weaknesses

- **Accuracy:** While highly accurate, larger, more complex models like RTDETRv2-x may achieve slightly higher mAP on challenging datasets, albeit at the cost of speed and resources.

### Ideal Use Cases

YOLOv5 excels in:

- Real-time object detection: Video surveillance, [security alarm systems](https://docs.ultralytics.com/guides/security-alarm-system/), and [AI in traffic management](https://www.ultralytics.com/blog/ai-in-traffic-management-from-congestion-to-coordination).
- Edge computing: Deployment on resource-constrained devices like [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/) and [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/).
- Mobile applications: Lightweight models suitable for mobile deployment.
- Rapid prototyping: Quick setup and training for various [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) tasks.

[Learn more about YOLOv5](https://docs.ultralytics.com/models/yolov5/){ .md-button }

## RTDETRv2: High Accuracy Real-Time Detection

**Authors:** Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu  
**Organization:** Baidu  
**Date:** 2023-04-17 (Initial RT-DETR), 2024-07-24 (RT-DETRv2 improvements)  
**Arxiv:** [https://arxiv.org/abs/2304.08069](https://arxiv.org/abs/2304.08069), [https://arxiv.org/abs/2407.17140](https://arxiv.org/abs/2407.17140)  
**GitHub:** [https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)  
**Docs:** [https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme)

RTDETRv2 (Real-Time Detection Transformer v2) is a state-of-the-art object detector that leverages the power of Vision Transformers (ViT) to achieve high accuracy while maintaining real-time performance.

### Architecture

RTDETRv2 utilizes a hybrid approach:

- **Backbone:** Typically a CNN (like ResNet variants) for initial feature extraction.
- **Encoder-Decoder:** A [Transformer](https://www.ultralytics.com/glossary/transformer)-based encoder-decoder structure that uses [self-attention mechanisms](https://www.ultralytics.com/glossary/self-attention) to capture global context within the image features. This allows the model to better understand relationships between distant objects and complex scenes.

### Strengths

- **High Accuracy:** The transformer architecture enables RTDETRv2 to achieve excellent mAP scores, particularly on complex datasets with dense or small objects.
- **Real-Time Capability:** Optimized to provide competitive inference speeds, especially when accelerated using tools like [NVIDIA TensorRT](https://docs.ultralytics.com/integrations/tensorrt/).
- **Robust Feature Extraction:** Effectively captures global context, leading to better performance in challenging scenarios like occlusion.

### Weaknesses

- **Computational Cost:** Generally has a higher parameter count and FLOPs compared to YOLOv5, requiring more significant computational resources (GPU memory, processing power).
- **Training Complexity:** Training transformer-based models can be more resource-intensive and potentially slower than training CNN-based models like YOLOv5.
- **Inference Speed:** While real-time capable on powerful hardware, it may be slower than the fastest YOLOv5 variants, especially on CPUs or less powerful edge devices.
- **Ecosystem:** Lacks the extensive, unified ecosystem, tooling (like Ultralytics HUB), and broad community support provided by Ultralytics for YOLO models.

### Ideal Use Cases

RTDETRv2 is best suited for applications where **accuracy is paramount** and sufficient computational resources are available:

- Autonomous driving: Precise perception for [AI in self-driving cars](https://www.ultralytics.com/solutions/ai-in-automotive).
- Medical imaging: Detailed anomaly detection in [AI in Healthcare](https://www.ultralytics.com/solutions/ai-in-healthcare).
- High-resolution image analysis: Analyzing satellite imagery or industrial inspection data ([improving manufacturing with computer vision](https://www.ultralytics.com/blog/improving-manufacturing-with-computer-vision)).
- Complex scene understanding: Scenarios with heavy occlusion or numerous small objects.

[Learn more about RTDETRv2](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme){ .md-button }

## Performance Comparison: YOLOv5 vs RTDETRv2

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv5n    | 640                   | 28.0                 | **73.6**                       | **1.12**                            | **2.6**            | **7.7**           |
| YOLOv5s    | 640                   | 37.4                 | 120.7                          | 1.92                                | 9.1                | 24.0              |
| YOLOv5m    | 640                   | 45.4                 | 233.9                          | 4.03                                | 25.1               | 64.2              |
| YOLOv5l    | 640                   | 49.0                 | 408.4                          | 6.61                                | 53.2               | 135.0             |
| YOLOv5x    | 640                   | 50.7                 | 763.2                          | 11.89                               | 97.2               | 246.4             |
|            |                       |                      |                                |                                     |                    |                   |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | **54.3**             | -                              | 15.03                               | 76                 | 259               |

The table highlights the trade-offs:

- **YOLOv5** models (especially n/s/m) offer significantly faster inference speeds on both CPU and GPU (TensorRT) with much lower parameter counts and FLOPs, making them ideal for resource-constrained environments.
- **RTDETRv2** models achieve higher peak mAP scores (especially l/x variants) but come with increased latency and computational requirements. Notably, RTDETRv2-s/m offer competitive accuracy to YOLOv5l/x with potentially faster TensorRT speeds but lack reported CPU performance.

## Training and Ecosystem

**Ultralytics YOLOv5** stands out for its ease of training and comprehensive ecosystem. Training is straightforward using the provided CLI or Python API, backed by extensive documentation and tutorials. The Ultralytics ecosystem offers tools like Ultralytics HUB for simplified training and deployment, active community support, and seamless integrations with tools like [Weights & Biases](https://docs.ultralytics.com/integrations/weights-biases/) and [ClearML](https://docs.ultralytics.com/integrations/clearml/). Furthermore, YOLOv5's CNN architecture generally requires less GPU memory and trains faster than transformer models.

**RTDETRv2**, while powerful, involves training a more complex transformer architecture. This typically demands more substantial computational resources (especially high GPU memory) and potentially longer training times. While the [GitHub repository](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch) provides training scripts, the surrounding ecosystem and support structure are less extensive than those offered by Ultralytics.

## Conclusion

Both YOLOv5 and RTDETRv2 are capable object detection models, but they cater to different priorities.

- **Ultralytics YOLOv5** is the recommended choice for applications demanding **high speed, efficiency, ease of use, and deployment versatility**, especially on edge devices or where computational resources are limited. Its robust ecosystem and lower training requirements make it highly accessible for developers and researchers.
- **RTDETRv2** is suitable when **maximum accuracy** is the absolute priority, and sufficient computational resources (including powerful GPUs for training and inference) are available.

For most practical applications, YOLOv5 provides an excellent and often superior balance of performance, speed, and usability, backed by the strong support and tooling of the Ultralytics ecosystem.

## Explore Other Models

If you are exploring alternatives, consider other models within the Ultralytics ecosystem:

- **[YOLOv8](https://docs.ultralytics.com/models/yolov8/):** A successor to YOLOv5, offering improved accuracy and speed across various tasks including detection, segmentation, pose, and tracking.
- **[YOLOv10](https://docs.ultralytics.com/models/yolov10/):** Features innovations like NMS-free training for further efficiency gains.
- **[YOLO11](https://docs.ultralytics.com/models/yolo11/):** The latest generation from Ultralytics, pushing the boundaries of performance and efficiency.

Comparing models like [YOLOv8 vs RTDETRv2](https://docs.ultralytics.com/compare/yolov8-vs-rtdetr/) or [YOLOv10 vs RTDETRv2](https://docs.ultralytics.com/compare/yolov10-vs-rtdetr/) can provide further insights into the best fit for your project.
