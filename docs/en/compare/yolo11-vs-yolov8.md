---
description: Compare YOLO11 and YOLOv8 architectures, performance, use cases, and benchmarks. Discover which YOLO model fits your object detection needs.
keywords: YOLO11, YOLOv8, object detection, model comparison, performance benchmarks, YOLO series, computer vision, Ultralytics YOLO, YOLO architecture
---

# YOLO11 vs YOLOv8: Detailed Comparison

When selecting a computer vision model, particularly for object detection, it's essential to understand the strengths and weaknesses of different architectures. This page offers a detailed technical comparison between Ultralytics YOLO11 and YOLOv8, two state-of-the-art models designed for object detection tasks. We will analyze their architectural nuances, performance benchmarks, and suitable applications to guide you in making an informed decision.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO11", "YOLOv8"]'></canvas>

## Ultralytics YOLO11

Ultralytics YOLO11, authored by Glenn Jocher and Jing Qiu from Ultralytics and released on 2024-09-27, represents the newest evolution in the YOLO series. It is engineered for enhanced accuracy and efficiency in object detection and other vision tasks. Building on previous YOLO models, YOLO11 introduces architectural refinements aimed at improving detection precision while maintaining real-time performance.

**Architecture and Key Features:**
YOLO11 incorporates advancements in network structure to optimize feature extraction and processing. It achieves higher accuracy with potentially fewer parameters compared to predecessors like YOLOv8. This results in faster inference speeds and reduced computational demands, making it suitable for deployment across diverse platforms from edge devices to cloud infrastructure. YOLO11 supports a range of vision tasks including object detection, instance segmentation, image classification, pose estimation and oriented bounding boxes (OBB). For comprehensive details, refer to the [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/).

**Performance Metrics and Benchmarks:**
YOLO11 demonstrates strong performance metrics on the COCO dataset. For example, YOLO11m achieves a mAPval50-95 of 51.5% at a 640 image size. The model variants range from YOLO11n for applications prioritizing speed to YOLO11x for maximum accuracy. Inference speeds vary across model sizes, with YOLO11n achieving faster speeds suitable for real-time applications. Detailed performance metrics can be found in the [Ultralytics YOLO Docs](https://docs.ultralytics.com/).

**Use Cases:**
YOLO11's accuracy and efficiency make it ideal for applications requiring precise and fast object detection, such as:

- **Robotics**: Enabling navigation and interaction in dynamic environments.
- **Security Systems**: Enhancing advanced security systems for intrusion detection and monitoring as explored in [security alarm system projects](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8).
- **Retail Analytics**: Improving inventory management and customer behavior analysis for [AI in retail](https://www.ultralytics.com/blog/achieving-retail-efficiency-with-ai).
- **Industrial Automation**: Supporting quality control and defect detection in manufacturing, relevant to [AI in manufacturing solutions](https://www.ultralytics.com/solutions/ai-in-manufacturing).

**Strengths:**

- **High Accuracy**: Achieves state-of-the-art mAP scores.
- **Efficient Inference**: Offers fast processing speeds for real-time applications.
- **Versatile Tasks**: Supports multiple computer vision tasks beyond detection.
- **Scalability**: Performs well across different hardware environments.

**Weaknesses:**

- Larger models may require more computational resources, as is common with high-accuracy models.
- Further optimization might be needed for deployment on extremely resource-constrained edge devices.

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11){ .md-button }

## Ultralytics YOLOv8

Ultralytics YOLOv8, developed by Glenn Jocher, Ayush Chaurasia, and Jing Qiu at Ultralytics and released on 2023-01-10, is a highly versatile and powerful model in the YOLO series. It builds upon the strengths of previous versions, offering architectural improvements and new capabilities across various vision tasks. YOLOv8 is designed for ease of use and flexibility, making it accessible for both new and experienced users in the field of computer vision.

**Architecture and Key Features:**
YOLOv8 introduces a refined architecture that enhances performance and flexibility. It supports a comprehensive suite of vision AI tasks, including detection, segmentation, classification, pose estimation, and oriented bounding boxes. YOLOv8’s architecture is designed for streamlined workflows, emphasizing ease of training, validation, and deployment. Detailed information on its architecture and features can be found in the [YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/).

**Performance Metrics and Benchmarks:**
YOLOv8 provides excellent mAP scores while maintaining fast inference speeds. For instance, YOLOv8m achieves a mAPval50-95 of 50.2% at a 640 image size. Like YOLO11, YOLOv8 is available in different sizes (n, s, m, l, x) to suit various performance needs, balancing speed and accuracy. For a deeper understanding of its performance metrics, see the [YOLO performance metrics guide](https://docs.ultralytics.com/guides/yolo-performance-metrics/).

**Use Cases:**
YOLOv8's versatility makes it suitable for a wide array of applications, such as:

- **Real-time Object Detection**: Ideal for applications requiring rapid object detection in [security systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8) and [smart cities](https://www.ultralytics.com/blog/computer-vision-ai-in-smart-cities).
- **Pose Estimation**: Enabling advanced pose estimation tasks as detailed in [pose estimation with YOLOv8](https://www.ultralytics.com/blog/pose-estimation-with-ultralytics-yolov8).
- **Instance Segmentation**: Facilitating precise instance segmentation for applications in [healthcare](https://www.ultralytics.com/solutions/ai-in-healthcare) and [manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing).
- **Rapid Prototyping**: Its ease of use and extensive documentation make it excellent for rapid project development.

**Strengths:**

- **State-of-the-Art Performance**: Delivers excellent mAP and fast inference speeds.
- **Versatility**: Supports a wide range of vision tasks.
- **Ease of Use**: User-friendly tools and comprehensive documentation.
- **Strong Community**: Benefits from a large open-source community and integration with Ultralytics HUB.

**Weaknesses:**

- Larger models require significant computational resources.
- For extremely latency-sensitive applications, even smaller variants might need further optimization.

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLO11n | 640                   | 39.5                 | 56.1                           | 1.5                                 | 2.6                | 6.5               |
| YOLO11s | 640                   | 47.0                 | 90.0                           | 2.5                                 | 9.4                | 21.5              |
| YOLO11m | 640                   | 51.5                 | 183.2                          | 4.7                                 | 20.1               | 68.0              |
| YOLO11l | 640                   | 53.4                 | 238.6                          | 6.2                                 | 25.3               | 86.9              |
| YOLO11x | 640                   | 54.7                 | 462.8                          | 11.3                                | 56.9               | 194.9             |
|         |                       |                      |                                |                                     |                    |                   |
| YOLOv8n | 640                   | 37.3                 | 80.4                           | 1.47                                | 3.2                | 8.7               |
| YOLOv8s | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |

For users interested in exploring other models, Ultralytics offers a range of YOLO models including [YOLOv5](https://docs.ultralytics.com/models/yolov5/), [YOLOv7](https://docs.ultralytics.com/models/yolov7/), [YOLOv9](https://docs.ultralytics.com/models/yolov9/), and [YOLOv10](https://docs.ultralytics.com/models/yolov10/), each with its own unique characteristics and performance profiles. Further comparisons with other models like [RT-DETR](https://docs.ultralytics.com/models/rtdetr/), [DAMO-YOLO](https://docs.ultralytics.com/compare/damo-yolo-vs-yolov8/), and [PP-YOLOE](https://docs.ultralytics.com/compare/pp-yoloe-vs-yolov9/) are also available in the Ultralytics documentation to help you choose the best model for your specific needs.
