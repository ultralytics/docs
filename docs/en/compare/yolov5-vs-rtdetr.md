---
comments: true
description: Compare YOLOv5 and RTDETRv2 for object detection. Explore their architectures, performance metrics, strengths, and best use cases in computer vision.
keywords: YOLOv5, RTDETRv2, object detection, model comparison, Ultralytics, computer vision, machine learning, real-time detection, Vision Transformers, AI models
---

# YOLOv5 vs RTDETRv2: A Detailed Model Comparison

Choosing the right object detection model is crucial for computer vision projects. Ultralytics YOLO offers a suite of models tailored for various needs. This page provides a technical comparison between [Ultralytics YOLOv5](https://docs.ultralytics.com/models/yolov5/) and [RT-DETR v2](https://docs.ultralytics.com/models/rtdetr/), highlighting their architectural differences, performance metrics, and ideal applications.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv5", "RTDETRv2"]'></canvas>

## YOLOv5: Speed and Efficiency

[Ultralytics YOLOv5](https://docs.ultralytics.com/models/yolov5/), created by Glenn Jocher from Ultralytics and released on 2020-06-26 ([GitHub](https://github.com/ultralytics/yolov5)), is a highly popular one-stage object detector known for its speed and efficiency. Its architecture is based on:

- **Backbone:** CSPDarknet53 for feature extraction.
- **Neck:** PANet for feature fusion.
- **Head:** YOLOv5 head for detection.

YOLOv5 comes in various sizes (n, s, m, l, x), offering a trade-off between speed and accuracy.

**Strengths:**

- **Speed:** YOLOv5 excels in inference speed, making it suitable for real-time applications like [security alarm systems](https://docs.ultralytics.com/guides/security-alarm-system/) and [vision-based eye tracking](https://docs.ultralytics.com/guides/vision-eye/).
- **Efficiency:** Models are relatively small and require less computational resources, ideal for [edge deployments on devices like Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/) and [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/).
- **Versatility:** Adaptable to various hardware and software environments, with seamless [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/) and [OpenVINO](https://docs.ultralytics.com/integrations/openvino/) support.
- **Ease of Use:** Well-documented and easy to implement with Ultralytics [Python package](https://pypi.org/project/ultralytics/) and [Ultralytics HUB](https://www.ultralytics.com/hub).

**Weaknesses:**

- **Accuracy:** While highly accurate, larger models like RTDETRv2 may achieve higher mAP, especially on complex datasets.

**Use Cases:**

- Real-time object detection in video surveillance and [computer vision for theft prevention](https://www.ultralytics.com/blog/computer-vision-for-theft-prevention-enhancing-security).
- Mobile and edge deployments for [smart cities applications](https://www.ultralytics.com/blog/computer-vision-ai-in-smart-cities) and [AI in agriculture](https://www.ultralytics.com/solutions/ai-in-agriculture).
- Applications requiring rapid processing, such as robotics ([ROS Quickstart](https://docs.ultralytics.com/guides/ros-quickstart/)) and [autonomous vehicles](https://www.ultralytics.com/solutions/ai-in-self-driving).

[Learn more about YOLOv5](https://docs.ultralytics.com/models/yolov5/){ .md-button }

## RTDETRv2: High Accuracy Real-Time Detection

**RTDETRv2** (Real-Time Detection Transformer v2), authored by Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu from Baidu and published on 2023-04-17 ([Arxiv](https://arxiv.org/abs/2304.08069), [GitHub](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)), is a cutting-edge object detection model known for its high accuracy and real-time capabilities. Built upon a Vision Transformer (ViT) architecture, RTDETRv2 excels in tasks requiring precise object localization and classification.

### Architecture and Key Features

RTDETRv2 employs a transformer-based architecture, enabling it to capture global context within images, leading to improved accuracy, especially in complex scenes. Unlike traditional Convolutional Neural Networks (CNNs), Vision Transformers leverage [self-attention mechanisms](https://www.ultralytics.com/glossary/self-attention) to weigh the importance of different image regions, enhancing feature extraction. This architecture allows RTDETRv2 to achieve state-of-the-art accuracy while maintaining competitive inference speeds, especially when using an [inference engine](https://www.ultralytics.com/glossary/inference-engine) like [TensorRT](https://www.ultralytics.com/glossary/tensorrt).

### Strengths and Weaknesses

**Strengths:**

- **High Accuracy:** Transformer-based architecture enables superior object detection accuracy, critical for applications like [medical image analysis](https://www.ultralytics.com/glossary/medical-image-analysis) and [satellite image analysis](https://www.ultralytics.com/blog/using-computer-vision-to-analyse-satellite-imagery).
- **Real-Time Performance:** Achieves competitive inference speeds, particularly with hardware acceleration like NVIDIA T4 GPUs, making it suitable for real-time applications in [AI in aviation](https://www.ultralytics.com/blog/ai-in-aviation-a-runway-to-smarter-airports) and [AI in traffic management](https://www.ultralytics.com/blog/ai-in-traffic-management-from-congestion-to-coordination).
- **Robust Feature Extraction:** Vision Transformers effectively capture global context and intricate details, beneficial in scenarios with complex backgrounds or occlusions.

**Weaknesses:**

- **Larger Model Size:** Models like RTDETRv2-x have a larger parameter count and FLOPs compared to smaller YOLO models, requiring more computational resources and larger [batch sizes](https://www.ultralytics.com/glossary/batch-size).
- **Inference Speed:** While real-time capable, inference speed might be slower than the fastest YOLO models on resource-constrained devices.

### Ideal Use Cases

RTDETRv2 is ideally suited for applications where high accuracy is paramount and sufficient computational resources are available. These include:

- **Autonomous Vehicles:** For reliable and precise perception of the environment in [AI in self-driving cars](https://www.ultralytics.com/solutions/ai-in-self-driving).
- **Robotics:** Enabling robots to accurately interact with and manipulate objects in complex settings, as explored in [AI's role in robotics](https://www.ultralytics.com/blog/from-algorithms-to-automation-ais-role-in-robotics).
- **Medical Imaging:** For precise detection of anomalies in medical images, aiding in diagnostics and [AI in Healthcare](https://www.ultralytics.com/solutions/ai-in-healthcare).
- **High-Resolution Image Analysis:** Applications requiring detailed analysis of large images, such as [using computer vision to analyse satellite imagery](https://www.ultralytics.com/blog/using-computer-vision-to-analyse-satellite-imagery) or industrial inspection for [improving manufacturing with computer vision](https://www.ultralytics.com/blog/improving-manufacturing-with-computer-vision).

[Learn more about RTDETRv2](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

| Model      | size<sup>(pixels) | mAP<sup>val50-95 | Speed<sup>CPU ONNX<sup>(ms) | Speed<sup>T4 TensorRT10<sup>(ms) | params<sup>(M) | FLOPs<sup>(B) |
|------------|-------------------|------------------|-----------------------------|----------------------------------|----------------|---------------|
| YOLOv5n    | 640               | 28.0             | 73.6                        | 1.12                             | 2.6            | 7.7           |
| YOLOv5s    | 640               | 37.4             | 120.7                       | 1.92                             | 9.1            | 24.0          |
| YOLOv5m    | 640               | 45.4             | 233.9                       | 4.03                             | 25.1           | 64.2          |
| YOLOv5l    | 640               | 49.0             | 408.4                       | 6.61                             | 53.2           | 135.0         |
| YOLOv5x    | 640               | 50.7             | 763.2                       | 11.89                            | 97.2           | 246.4         |
|            |                   |                  |                             |                                  |                |               |
| RTDETRv2-s | 640               | 48.1             | -                           | 5.03                             | 20             | 60            |
| RTDETRv2-m | 640               | 51.9             | -                           | 7.51                             | 36             | 100           |
| RTDETRv2-l | 640               | 53.4             | -                           | 9.76                             | 42             | 136           |
| RTDETRv2-x | 640               | 54.3             | -                           | 15.03                            | 76             | 259           |

## Conclusion

Both RTDETRv2 and YOLOv5 are powerful object detection models, each catering to different needs. **RTDETRv2** is the preferred choice when top-tier accuracy is the priority and computational resources are available. **YOLOv5**, on the other hand, shines in scenarios demanding real-time performance, efficiency, and deployment on resource-constrained platforms.

For users seeking other options, Ultralytics offers a diverse model zoo, including:

- **YOLOv8 and YOLOv9:** Previous generations of YOLO models, providing a range of speed-accuracy trade-offs. Learn more about [Ultralytics YOLOv8's first year](https://www.ultralytics.com/blog/ultralytics-yolov8-turns-one-a-year-of-breakthroughs-and-innovations) and [YOLOv9](https://docs.ultralytics.com/models/yolov9/).
- **YOLO11:** The latest model in the YOLO series, offering enhanced capabilities in object detection. Explore the features of [YOLO11](https://docs.ultralytics.com/models/yolo11/).
- **YOLO-NAS:** Models designed with Neural Architecture Search for optimal performance. See details on [YOLO-NAS by Deci AI](https://docs.ultralytics.com/models/yolo-nas/).
- **FastSAM and MobileSAM:** For real-time instance segmentation tasks. Investigate [FastSAM](https://docs.ultralytics.com/models/fast-sam/) and [MobileSAM](https://docs.ultralytics.com/models/mobile-sam/).

Choosing between RTDETRv2 and YOLOv5, or other Ultralytics models, depends on the specific requirements of your computer vision project, balancing accuracy, speed, and resource constraints. Refer to the [Ultralytics Documentation](https://docs.ultralytics.com/models/) and [GitHub repository](https://github.com/ultralytics/ultralytics) for detailed information and implementation guides.
