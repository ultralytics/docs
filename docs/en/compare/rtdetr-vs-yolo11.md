---
comments: true
description: Explore the technical comparison of RTDETRv2 and YOLO11. Discover strengths, weaknesses, and ideal use cases to choose the best detection model.
keywords: RTDETRv2, YOLO11, object detection, model comparison, computer vision, real-time detection, accuracy, performance metrics, Ultralytics
---

# RTDETRv2 vs YOLO11: A Technical Comparison for Object Detection

Choosing the right object detection model is crucial for computer vision projects. Ultralytics offers a range of models, including the efficient YOLO series and the high-accuracy RT-DETR series. This page provides a detailed technical comparison between **RTDETRv2** and **YOLO11**, two state-of-the-art models for object detection, to help you make an informed decision.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["RTDETRv2", "YOLO11"]'></canvas>

## RTDETRv2: High Accuracy Real-Time Detection

**RTDETRv2** (Real-Time Detection Transformer v2) is a cutting-edge object detection model known for its high accuracy and real-time capabilities.

- **Authors:** Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu
- **Organization:** Baidu
- **Date:** 2023-04-17 (Initial RT-DETR), 2024-07-24 (RTDETRv2 improvements)
- **Arxiv Link:** [https://arxiv.org/abs/2304.08069](https://arxiv.org/abs/2304.08069), [https://arxiv.org/abs/2407.17140](https://arxiv.org/abs/2407.17140)
- **GitHub Link:** [https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)
- **Docs Link:** [https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme)

### Architecture and Key Features

RTDETRv2 employs a transformer-based architecture, specifically a [Vision Transformer (ViT)](https://www.ultralytics.com/glossary/vision-transformer-vit), enabling it to capture global context within images. This leads to improved accuracy, especially in complex scenes. Unlike traditional Convolutional Neural Networks (CNNs), ViTs leverage [self-attention mechanisms](https://www.ultralytics.com/glossary/self-attention) to weigh the importance of different image regions, enhancing feature extraction. This architecture allows RTDETRv2 to achieve state-of-the-art accuracy while maintaining competitive inference speeds, often utilizing a hybrid approach combining CNNs for feature extraction and transformers for context modeling.

### Performance Metrics

As indicated in the comparison table below, RTDETRv2 models offer impressive mAP scores, particularly the larger variants like RTDETRv2-x, which achieves a mAP<sup>val</sup>50-95 of 54.3. Inference speeds on TensorRT are respectable, making it suitable for real-time applications when deployed on capable hardware like NVIDIA T4 GPUs, which are optimized for [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/).

### Strengths and Weaknesses

**Strengths:**

- **High Accuracy:** Transformer-based architecture enables superior object detection accuracy, crucial for applications like [medical image analysis](https://www.ultralytics.com/glossary/medical-image-analysis).
- **Real-Time Performance:** Achieves competitive inference speeds, especially with hardware acceleration.
- **Robust Feature Extraction:** Vision Transformers effectively capture global context and intricate details.

**Weaknesses:**

- **Larger Model Size:** Models like RTDETRv2-x have a larger parameter count and FLOPs compared to smaller YOLO models, requiring more computational resources and CUDA memory during training and inference.
- **Inference Speed:** While real-time capable, inference speed might be slower than the fastest YOLO models on resource-constrained devices, particularly on CPU.
- **Complexity:** Transformer models can be more complex to train and tune compared to CNN-based YOLO models.

### Ideal Use Cases

RTDETRv2 is ideally suited for applications where high accuracy is paramount and sufficient computational resources are available. These include:

- **Autonomous Vehicles:** For reliable and precise perception of the environment, essential for [AI in self-driving cars](https://www.ultralytics.com/solutions/ai-in-automotive).
- **Robotics:** Enabling robots to accurately interact with objects in complex settings, a key aspect of [AI's Role in Robotics](https://www.ultralytics.com/blog/from-algorithms-to-automation-ais-role-in-robotics).
- **Medical Imaging:** For precise detection of anomalies in medical images, aiding in diagnostics, improving [AI in Healthcare](https://www.ultralytics.com/solutions/ai-in-healthcare).
- **High-Resolution Image Analysis:** Applications requiring detailed analysis of large images, such as [satellite image analysis](https://www.ultralytics.com/blog/using-computer-vision-to-analyse-satellite-imagery) or industrial inspection.

[Learn more about RTDETRv2](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

## YOLO11: Efficient and Versatile Object Detection

**Ultralytics YOLO11** represents the latest iteration in the renowned Ultralytics YOLO series, known for its exceptional speed, efficiency, and ease of use.

- **Authors:** Glenn Jocher and Jing Qiu
- **Organization:** Ultralytics
- **Date:** 2024-09-27
- **GitHub Link:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Docs Link:** [https://docs.ultralytics.com/models/yolo11/](https://docs.ultralytics.com/models/yolo11/)

### Architecture and Key Features

YOLO11 continues the single-stage detection paradigm, prioritizing inference speed without significantly compromising accuracy. It incorporates architectural improvements and optimizations to achieve an excellent balance between speed and precision, building upon predecessors like [YOLOv8](https://docs.ultralytics.com/models/yolov8/). Ultralytics YOLO models are designed for efficient processing, making them highly suitable for real-time applications across diverse hardware platforms. Key advantages include:

- **Ease of Use:** Streamlined user experience with a simple API and extensive [documentation](https://docs.ultralytics.com/).
- **Well-Maintained Ecosystem:** Benefits from the integrated [Ultralytics ecosystem](https://www.ultralytics.com/), including active development, strong community support via [GitHub Issues](https://github.com/ultralytics/ultralytics/issues) and [Discord](https://discord.com/invite/ultralytics), frequent updates, and resources like [Ultralytics HUB](https://www.ultralytics.com/hub).
- **Versatility:** Supports multiple tasks beyond detection, including [segmentation](https://docs.ultralytics.com/tasks/segment/), [classification](https://docs.ultralytics.com/tasks/classify/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), and [OBB](https://docs.ultralytics.com/tasks/obb/).

### Performance Metrics

The performance table highlights YOLO11's strength in speed and efficiency. Models like YOLO11n and YOLO11s achieve impressive inference times on both CPU and GPU, making them excellent choices for latency-sensitive applications and edge deployments. While achieving competitive [mean average precision (mAP)](https://www.ultralytics.com/glossary/mean-average-precision-map), YOLO11 models generally have significantly lower parameter counts and FLOPs compared to RTDETRv2, leading to lower memory requirements.

### Strengths and Weaknesses

**Strengths:**

- **Exceptional Speed:** Famous for fast inference speeds, crucial for real-time applications.
- **High Efficiency:** Computationally efficient, enabling deployment on resource-limited devices ([Edge AI](https://www.ultralytics.com/glossary/edge-ai)).
- **Versatile Application:** Suitable for a broad range of tasks and deployment scenarios.
- **Small Model Size:** Memory-efficient due to reduced parameter count, especially smaller variants.
- **Ease of Use & Ecosystem:** Simple API, extensive docs, active community, and integrated tools like Ultralytics HUB simplify development.
- **Training Efficiency:** Efficient training process with readily available pre-trained weights and lower CUDA memory usage compared to transformer models.

**Weaknesses:**

- **Accuracy Trade-off:** In scenarios demanding the absolute highest accuracy, particularly with complex or overlapping objects, larger models like RTDETRv2 might offer marginally better performance, albeit at a higher computational cost.

### Ideal Use Cases

YOLO11's speed, efficiency, and versatility make it ideal for:

- **Real-time Surveillance:** Rapid object detection in security systems ([security alarm system projects with Ultralytics YOLOv8](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8)).
- **Edge AI:** Deployment on mobile, embedded, and IoT devices ([Edge AI and AIoT Upgrade Any Camera with Ultralytics YOLOv8 in a No-Code Way](https://www.ultralytics.com/blog/edge-ai-and-aiot-upgrade-any-camera-with-ultralytics-yolov8-in-a-no-code-way)).
- **Retail Analytics:** Real-time customer and inventory analysis ([AI for Smarter Retail Inventory Management](https://www.ultralytics.com/blog/ai-for-smarter-retail-inventory-management)).
- **Traffic Management:** Efficient vehicle detection and traffic flow analysis ([optimizing traffic management with Ultralytics YOLO11](https://www.ultralytics.com/blog/optimizingtraffic-management-with-ultralytics-yolo11)).

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

## Performance Comparison: RTDETRv2 vs YOLO11

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | 54.3                 | -                              | 15.03                               | 76                 | 259               |
|            |                       |                      |                                |                                     |                    |                   |
| YOLO11n    | 640                   | 39.5                 | **56.1**                       | **1.5**                             | **2.6**            | **6.5**           |
| YOLO11s    | 640                   | 47.0                 | 90.0                           | 2.5                                 | 9.4                | 21.5              |
| YOLO11m    | 640                   | 51.5                 | 183.2                          | 4.7                                 | 20.1               | 68.0              |
| YOLO11l    | 640                   | 53.4                 | 238.6                          | 6.2                                 | 25.3               | 86.9              |
| YOLO11x    | 640                   | **54.7**             | 462.8                          | 11.3                                | 56.9               | 194.9             |

## Conclusion

Both RTDETRv2 and YOLO11 are powerful object detection models, each excelling in different areas. **RTDETRv2** is the preferred choice when top-tier accuracy is the absolute priority and computational resources are readily available. Its transformer architecture allows for nuanced understanding of complex scenes.

**Ultralytics YOLO11**, however, shines in scenarios demanding real-time performance, high efficiency, and ease of deployment, particularly on resource-constrained platforms. Its excellent balance of speed and accuracy, coupled with lower memory requirements, faster training times, multi-task versatility, and the robust Ultralytics ecosystem, makes it a highly practical and developer-friendly choice for a vast array of real-world applications. For most users, YOLO11 offers a superior blend of performance, efficiency, and usability.

## Explore Other Models

For users seeking other options, Ultralytics offers a diverse model zoo, including:

- **[YOLOv10](https://docs.ultralytics.com/models/yolov10/):** Another highly efficient YOLO model focusing on NMS-free design.
- **[YOLOv9](https://docs.ultralytics.com/models/yolov9/) and [YOLOv8](https://docs.ultralytics.com/models/yolov8/):** Previous state-of-the-art YOLO models offering strong performance benchmarks.
- **[YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/):** Models designed with Neural Architecture Search for optimal performance.
- **[MobileSAM](https://docs.ultralytics.com/models/mobile-sam/) and [FastSAM](https://docs.ultralytics.com/models/fast-sam/):** Efficient models for instance segmentation tasks.

Choosing between RTDETRv2, YOLO11, or other Ultralytics models depends on the specific requirements of your computer vision project, balancing accuracy, speed, resource constraints, and ease of development. Refer to the [Ultralytics Documentation](https://docs.ultralytics.com/models/) and [GitHub repository](https://github.com/ultralytics/ultralytics) for detailed information and implementation guides.
