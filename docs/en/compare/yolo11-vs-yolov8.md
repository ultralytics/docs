---
comments: true
description: Compare YOLO11 and YOLOv8 architectures, performance, use cases, and benchmarks. Discover which YOLO model fits your object detection needs.
keywords: YOLO11, YOLOv8, object detection, model comparison, performance benchmarks, YOLO series, computer vision, Ultralytics YOLO, YOLO architecture
---

# YOLO11 vs YOLOv8: Detailed Comparison

When selecting a computer vision model, particularly for [object detection](https://www.ultralytics.com/glossary/object-detection), understanding the strengths and weaknesses of different architectures is essential. This page offers a detailed technical comparison between Ultralytics YOLO11 and [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/), two state-of-the-art models designed for object detection and other vision tasks. We will analyze their architectural nuances, performance benchmarks, and suitable applications to guide you in making an informed decision.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO11", "YOLOv8"]'></canvas>

## Ultralytics YOLO11

**Authors:** Glenn Jocher, Jing Qiu  
**Organization:** [Ultralytics](https://www.ultralytics.com/)  
**Date:** 2024-09-27  
**GitHub:** <https://github.com/ultralytics/ultralytics>  
**Docs:** <https://docs.ultralytics.com/models/yolo11/>

[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/), represents the newest evolution in the YOLO series. It is engineered for enhanced accuracy and efficiency in object detection and other vision tasks like [instance segmentation](https://docs.ultralytics.com/tasks/segment/), [image classification](https://docs.ultralytics.com/tasks/classify/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), and oriented bounding boxes (OBB). Building on previous YOLO models, YOLO11 introduces architectural refinements aimed at improving detection precision while maintaining real-time performance.

**Architecture and Key Features:**
YOLO11 incorporates advancements in network structure to optimize feature extraction and processing. It achieves higher accuracy with potentially fewer parameters compared to predecessors like YOLOv8, as seen in the performance table below. This results in faster inference speeds and reduced computational demands, making it suitable for deployment across diverse platforms from edge devices like [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/) to cloud infrastructure. YOLO11 benefits from the well-maintained Ultralytics ecosystem, offering efficient training processes, readily available pre-trained weights, and lower memory usage compared to many other model types.

**Performance Metrics and Benchmarks:**
YOLO11 demonstrates strong performance metrics on the [COCO dataset](https://docs.ultralytics.com/datasets/detect/coco/). For example, YOLO11m achieves a mAP<sup>val</sup>50-95 of 51.5% at a 640 image size, surpassing YOLOv8m. The model variants range from YOLO11n for applications prioritizing speed to YOLO11x for maximum accuracy. Inference speeds are notably faster on CPU compared to YOLOv8, offering a significant advantage in CPU-bound scenarios.

**Strengths:**

- **High Accuracy:** Achieves state-of-the-art mAP scores, often exceeding YOLOv8 at similar sizes.
- **Efficient Inference:** Offers faster processing speeds, especially on CPU, for real-time applications.
- **Versatile Tasks:** Supports multiple computer vision tasks within a single framework.
- **Scalability:** Performs well across different hardware environments, with efficient memory usage.
- **Ease of Use:** Benefits from the streamlined Ultralytics API, extensive [documentation](https://docs.ultralytics.com/models/yolo11/), and active community support.

**Weaknesses:**

- Larger models (e.g., YOLO11x) require significant computational resources, similar to other high-accuracy models.
- Being newer, it might have fewer third-party integrations initially compared to the more established YOLOv8.

**Use Cases:**
YOLO11's accuracy and efficiency make it ideal for applications requiring precise and fast object detection, such as:

- **Robotics**: Enabling navigation and interaction in dynamic environments.
- **Security Systems**: Enhancing advanced [security systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8) for intrusion detection and monitoring.
- **Retail Analytics**: Improving inventory management and customer behavior analysis for [AI in retail](https://www.ultralytics.com/blog/achieving-retail-efficiency-with-ai).
- **Industrial Automation**: Supporting quality control and defect detection in [manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing).

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

## Ultralytics YOLOv8

**Authors:** Glenn Jocher, Ayush Chaurasia, Jing Qiu  
**Organization:** [Ultralytics](https://www.ultralytics.com/)  
**Date:** 2023-01-10  
**GitHub:** <https://github.com/ultralytics/ultralytics>  
**Docs:** <https://docs.ultralytics.com/models/yolov8/>

[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) is a highly versatile and powerful model in the YOLO series. It builds upon the strengths of previous versions, offering architectural improvements and new capabilities across various vision tasks. YOLOv8 is designed for ease of use and flexibility, making it accessible for both new and experienced users in the field of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv).

**Architecture and Key Features:**
YOLOv8 introduced a refined architecture featuring an anchor-free detection head and a C2f module in the neck, enhancing performance and flexibility. It supports a comprehensive suite of vision AI tasks, including detection, segmentation, classification, pose estimation, and OBB. YOLOv8's architecture is designed for streamlined workflows, emphasizing ease of training, validation, and deployment within the robust Ultralytics ecosystem.

**Performance Metrics and Benchmarks:**
YOLOv8 provides excellent mAP scores while maintaining fast inference speeds. For instance, YOLOv8m achieves a mAP<sup>val</sup>50-95 of 50.2% at a 640 image size. Like YOLO11, YOLOv8 is available in different sizes (n, s, m, l, x) to suit various performance needs, balancing speed and accuracy effectively. It offers a strong performance balance suitable for diverse real-world deployment scenarios.

**Strengths:**

- **State-of-the-Art Performance:** Delivers excellent mAP and fast inference speeds.
- **Versatility:** Supports a wide range of vision tasks.
- **Ease of Use:** User-friendly tools, simple [Python](https://docs.ultralytics.com/usage/python/) and [CLI](https://docs.ultralytics.com/usage/cli/) interfaces, and comprehensive documentation.
- **Well-Maintained Ecosystem:** Benefits from a large open-source community, frequent updates, extensive resources, and integration with [Ultralytics HUB](https://www.ultralytics.com/hub).
- **Training Efficiency:** Efficient training processes and readily available pre-trained weights.

**Weaknesses:**

- Larger models require significant computational resources.
- While fast, CPU inference speeds are generally slower than YOLO11 for corresponding model sizes.

**Use Cases:**
YOLOv8's versatility makes it suitable for a wide array of applications, such as:

- **Real-time Object Detection**: Ideal for applications requiring rapid object detection in [security systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8) and [smart cities](https://www.ultralytics.com/blog/computer-vision-ai-in-smart-cities).
- **Pose Estimation**: Enabling advanced pose estimation tasks as detailed in [pose estimation with YOLOv8](https://www.ultralytics.com/blog/pose-estimation-with-ultralytics-yolov8).
- **Instance Segmentation**: Facilitating precise instance segmentation for applications in [healthcare](https://www.ultralytics.com/solutions/ai-in-healthcare) and [manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing).
- **Rapid Prototyping**: Its ease of use and extensive documentation make it excellent for rapid project development.

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## Performance Comparison

The table below provides a detailed comparison of object detection performance for different YOLO11 and YOLOv8 variants on the COCO val2017 dataset. YOLO11 generally shows improved mAP and faster CPU inference speeds with fewer parameters and FLOPs compared to its YOLOv8 counterparts.

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLO11n | 640                   | **39.5**             | **56.1**                       | 1.5                                 | **2.6**            | **6.5**           |
| YOLO11s | 640                   | **47.0**             | **90.0**                       | **2.5**                             | **9.4**            | **21.5**          |
| YOLO11m | 640                   | **51.5**             | **183.2**                      | **4.7**                             | **20.1**           | **68.0**          |
| YOLO11l | 640                   | **53.4**             | **238.6**                      | **6.2**                             | **25.3**           | **86.9**          |
| YOLO11x | 640                   | **54.7**             | **462.8**                      | **11.3**                            | **56.9**           | **194.9**         |
|         |                       |                      |                                |                                     |                    |                   |
| YOLOv8n | 640                   | 37.3                 | 80.4                           | **1.47**                            | 3.2                | 8.7               |
| YOLOv8s | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |

For users interested in exploring other models, Ultralytics offers a range of YOLO models including [YOLOv5](https://docs.ultralytics.com/models/yolov5/), [YOLOv7](https://docs.ultralytics.com/models/yolov7/), [YOLOv9](https://docs.ultralytics.com/models/yolov9/), and [YOLOv10](https://docs.ultralytics.com/models/yolov10/), each with its own unique characteristics and performance profiles. Further comparisons with other models like [RT-DETR](https://docs.ultralytics.com/models/rtdetr/), [DAMO-YOLO](https://docs.ultralytics.com/compare/damo-yolo-vs-yolov8/), and [PP-YOLOE](https://docs.ultralytics.com/compare/pp-yoloe-vs-yolov9/) are also available in the [Ultralytics documentation](https://docs.ultralytics.com/compare/) to help you choose the best model for your specific needs.
