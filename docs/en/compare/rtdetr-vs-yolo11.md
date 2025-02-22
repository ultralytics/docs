---
comments: true
description: Compare RTDETRv2 and YOLO11 for object detection. Analyze key features, accuracy, speed, and use cases to find the best model for your needs.
keywords: RTDETRv2,YOLO11,object detection,Ultralytics,Vision Transformer,YOLO models,model comparison,real-time detection,computer vision
---

# RTDETRv2 vs YOLO11: A Technical Comparison for Object Detection

Choosing the right object detection model is crucial for computer vision projects. Ultralytics offers a range of models, including the efficient YOLO series and the high-accuracy RT-DETR series. This page provides a detailed technical comparison between **RTDETRv2** and **YOLO11**, two state-of-the-art models for object detection, to help you make an informed decision.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["RTDETRv2", "YOLO11"]'></canvas>

## RTDETRv2: High Accuracy Real-Time Detection

**RTDETRv2** ([Real-Time Detection Transformer v2](https://docs.ultralytics.com/models/rtdetr/)) is a cutting-edge object detection model known for its high accuracy and real-time capabilities. Built upon a Vision Transformer (ViT) architecture, RTDETRv2 excels in tasks requiring precise object localization and classification.

### Architecture and Key Features

RTDETRv2 employs a transformer-based architecture, enabling it to capture global context within images, leading to improved accuracy, especially in complex scenes. Unlike traditional Convolutional Neural Networks (CNNs), Vision Transformers leverage self-attention mechanisms to weigh the importance of different image regions, enhancing feature extraction. This architecture allows RTDETRv2 to achieve state-of-the-art accuracy while maintaining competitive inference speeds.

### Performance Metrics

As indicated in the comparison table below, RTDETRv2 models offer impressive mAP scores, particularly the larger variants like RTDETRv2-x, which achieves a mAP<sup>val</sup><sub>50-95</sub> of 54.3. Inference speeds on TensorRT are also respectable, making it suitable for real-time applications when deployed on capable hardware like NVIDIA T4 GPUs.

### Strengths and Weaknesses

**Strengths:**

- **High Accuracy:** Transformer-based architecture enables superior object detection accuracy.
- **Real-Time Performance:** Achieves competitive inference speeds, especially with hardware acceleration.
- **Robust Feature Extraction:** Vision Transformers effectively capture global context and intricate details.

**Weaknesses:**

- **Larger Model Size:** Models like RTDETRv2-x have a larger parameter count and FLOPs compared to smaller YOLO models, requiring more computational resources.
- **Inference Speed:** While real-time capable, inference speed might be slower than the fastest YOLO models on resource-constrained devices.

### Ideal Use Cases

RTDETRv2 is ideally suited for applications where high accuracy is paramount and sufficient computational resources are available. These include:

- **Autonomous Vehicles:** For reliable and precise perception of the environment. [AI in self-driving cars](https://www.ultralytics.com/solutions/ai-in-self-driving)
- **Robotics:** Enabling robots to accurately interact with and manipulate objects in complex settings. [From Algorithms to Automation: AI's Role in Robotics](https://www.ultralytics.com/blog/from-algorithms-to-automation-ais-role-in-robotics)
- **Medical Imaging:** For precise detection of anomalies in medical images, aiding in diagnostics. [AI in Healthcare](https://www.ultralytics.com/solutions/ai-in-healthcare)
- **High-Resolution Image Analysis:** Applications requiring detailed analysis of large images, such as satellite imagery or industrial inspection. [Using Computer Vision to Analyse Satellite Imagery](https://www.ultralytics.com/blog/using-computer-vision-to-analyse-satellite-imagery)

[Learn more about RTDETRv2](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

## YOLO11: Efficient and Versatile Object Detection

**YOLO11** ([You Only Look Once 11](https://docs.ultralytics.com/models/yolo11/)) represents the latest iteration in the renowned Ultralytics YOLO series, known for its speed and efficiency. YOLO11 builds upon previous versions, offering enhanced accuracy and performance while maintaining its real-time edge.

### Architecture and Key Features

YOLO11 continues the single-stage detection paradigm, prioritizing inference speed without significantly compromising accuracy. It incorporates architectural improvements and optimizations to achieve a better balance between speed and precision compared to its predecessors like [YOLOv8](https://docs.ultralytics.com/models/yolov8/). YOLO models are designed for efficient processing, making them highly suitable for real-time applications across diverse hardware platforms.

### Performance Metrics

The performance table highlights YOLO11's strength in speed. Models like YOLO11n and YOLO11s achieve impressive inference times on both CPU and GPU, making them excellent choices for latency-sensitive applications and edge deployments. While slightly lower in mAP compared to the larger RTDETRv2 models, YOLO11 still delivers competitive accuracy for a wide range of object detection tasks.

### Strengths and Weaknesses

**Strengths:**

- **Exceptional Speed:** YOLO models are renowned for their fast inference speeds, crucial for real-time applications.
- **Efficiency:** YOLO11 models are computationally efficient, allowing deployment on resource-constrained devices.
- **Versatility:** Suitable for a broad spectrum of object detection tasks and deployment scenarios.
- **Small Model Size:** Smaller YOLO11 variants have significantly fewer parameters, making them memory-efficient.

**Weaknesses:**

- **Accuracy Trade-off:** In scenarios demanding the absolute highest accuracy, particularly with complex or overlapping objects, larger models like RTDETRv2 might offer superior performance.

### Ideal Use Cases

YOLO11's speed and efficiency make it ideal for applications with real-time processing requirements and deployments on edge devices. Key use cases include:

- **Real-time Video Surveillance:** For efficient and rapid detection of objects in security systems. [Security Alarm System Projects with Ultralytics YOLOv8](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8)
- **Edge AI Applications:** Deployment on mobile devices, embedded systems, and IoT devices for on-device processing. [Edge AI and AIoT Upgrade Any Camera with Ultralytics YOLOv8 in a No-Code Way](https://www.ultralytics.com/blog/edge-ai-and-aiot-upgrade-any-camera-with-ultralytics-yolov8-in-a-no-code-way)
- **Robotics and Automation:** For real-time perception in robotic systems and automated processes. [AI in Robotics](https://www.ultralytics.com/glossary/robotics)
- **Retail Analytics:** For real-time customer and inventory analysis in retail environments. [AI for Smarter Retail Inventory Management](https://www.ultralytics.com/blog/ai-for-smarter-retail-inventory-management)
- **Traffic Management:** For real-time vehicle detection and traffic flow analysis. [Optimizing Traffic Management with Ultralytics YOLO11](https://www.ultralytics.com/blog/optimizingtraffic-management-with-ultralytics-yolo11)

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

## Model Comparison Table

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | 54.3                 | -                              | 15.03                               | 76                 | 259               |
|            |                       |                      |                                |                                     |                    |                   |
| YOLO11n    | 640                   | 39.5                 | 56.1                           | 1.5                                 | 2.6                | 6.5               |
| YOLO11s    | 640                   | 47.0                 | 90.0                           | 2.5                                 | 9.4                | 21.5              |
| YOLO11m    | 640                   | 51.5                 | 183.2                          | 4.7                                 | 20.1               | 68.0              |
| YOLO11l    | 640                   | 53.4                 | 238.6                          | 6.2                                 | 25.3               | 86.9              |
| YOLO11x    | 640                   | 54.7                 | 462.8                          | 11.3                                | 56.9               | 194.9             |

## Conclusion

Both RTDETRv2 and YOLO11 are powerful object detection models, each catering to different needs. **RTDETRv2** is the preferred choice when top-tier accuracy is the priority and computational resources are available. **YOLO11**, on the other hand, shines in scenarios demanding real-time performance, efficiency, and deployment on resource-constrained platforms.

For users seeking other options, Ultralytics offers a diverse model zoo, including:

- **YOLOv8 and YOLOv9:** Previous generations of YOLO models, providing a range of speed-accuracy trade-offs. [Ultralytics YOLOv8 Turns One: A Year of Breakthroughs and Innovations](https://www.ultralytics.com/blog/ultralytics-yolov8-turns-one-a-year-of-breakthroughs-and-innovations) and [YOLOv9](https://docs.ultralytics.com/models/yolov9/)
- **YOLO-NAS:** Models designed with Neural Architecture Search for optimal performance. [YOLO-NAS by Deci AI - a State-of-the-Art Object Detection Model](https://docs.ultralytics.com/models/yolo-nas/)
- **FastSAM and MobileSAM:** For real-time instance segmentation tasks. [FastSAM](https://docs.ultralytics.com/models/fast-sam/) and [MobileSAM](https://docs.ultralytics.com/models/mobile-sam/)

Choosing between RTDETRv2 and YOLO11, or other Ultralytics models, depends on the specific requirements of your computer vision project, balancing accuracy, speed, and resource constraints. Refer to the [Ultralytics Documentation](https://docs.ultralytics.com/models/) and [GitHub repository](https://github.com/ultralytics/ultralytics) for detailed information and implementation guides.
