---
comments: true
description: Compare YOLO11 and YOLOv8 architectures, performance, use cases, and benchmarks. Discover which YOLO model fits your object detection needs.
keywords: YOLO11, YOLOv8, object detection, model comparison, performance benchmarks, YOLO series, computer vision, Ultralytics YOLO, YOLO architecture
---

# YOLO11 vs YOLOv8: Detailed Comparison

When selecting a computer vision model, particularly for [object detection](https://docs.ultralytics.com/tasks/detect/), understanding the strengths and weaknesses of different architectures is essential. This page offers a detailed technical comparison between [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) and [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/), two state-of-the-art models designed for object detection and other vision tasks. We will analyze their architectural nuances, performance benchmarks, and suitable applications to guide you in making an informed decision for your next [AI](https://www.ultralytics.com/glossary/artificial-intelligence-ai) project.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO11", "YOLOv8"]'></canvas>

## Ultralytics YOLO11

**Authors:** Glenn Jocher, Jing Qiu  
**Organization:** [Ultralytics](https://www.ultralytics.com/)  
**Date:** 2024-09-27  
**GitHub:** <https://github.com/ultralytics/ultralytics>  
**Docs:** <https://docs.ultralytics.com/models/yolo11/>

Ultralytics YOLO11 represents the newest evolution in the YOLO series, engineered for enhanced accuracy and efficiency. Building on the robust foundation of previous YOLO models, YOLO11 introduces architectural refinements aimed at improving detection precision while maintaining exceptional [real-time performance](https://www.ultralytics.com/glossary/real-time-inference). It is a highly versatile model, supporting a wide range of tasks including [instance segmentation](https://docs.ultralytics.com/tasks/segment/), [image classification](https://docs.ultralytics.com/tasks/classify/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), and oriented bounding boxes (OBB).

### Architecture and Key Features

YOLO11 incorporates advancements in network structure to optimize feature extraction and processing. It achieves higher accuracy with fewer parameters and FLOPs compared to its predecessors like YOLOv8, as shown in the performance table below. This efficiency translates to faster inference speeds and reduced computational demands, making it suitable for deployment across diverse platforms, from [edge devices](https://docs.ultralytics.com/guides/nvidia-jetson/) to powerful cloud infrastructure. A key advantage of YOLO11 is its seamless integration into the well-maintained Ultralytics ecosystem, which provides efficient [training processes](https://docs.ultralytics.com/modes/train/), readily available pre-trained weights, and lower memory usage compared to many other model types.

### Strengths

- **Superior Accuracy:** Achieves state-of-the-art [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) scores, consistently outperforming YOLOv8 at similar model sizes.
- **Highly Efficient Inference:** Offers significantly faster processing speeds, especially on CPU, which is critical for real-time applications in resource-constrained environments.
- **Multi-Task Versatility:** A single, unified framework supports multiple [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) tasks, simplifying development workflows.
- **Optimized and Scalable:** Performs well across different hardware with efficient memory usage and a smaller computational footprint.
- **Ease of Use:** Benefits from the streamlined Ultralytics API, extensive [documentation](https://docs.ultralytics.com/models/yolo11/), and active community support on [GitHub](https://github.com/ultralytics/ultralytics) and [Discord](https://discord.com/invite/ultralytics).

### Weaknesses

- As a newer model, it may initially have fewer third-party integrations compared to the more established YOLOv8.
- The largest models (e.g., YOLO11x) still require substantial computational resources, a common trait for high-accuracy detectors.

### Use Cases

YOLO11's exceptional balance of accuracy and efficiency makes it the ideal choice for applications demanding precise and fast object detection, such as:

- **Robotics**: Enabling navigation and object interaction in dynamic environments for [autonomous systems](https://www.ultralytics.com/glossary/autonomous-vehicles).
- **Security Systems**: Enhancing advanced [security systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8) for intrusion detection and real-time monitoring.
- **Retail Analytics**: Improving inventory management and customer behavior analysis for [AI in retail](https://www.ultralytics.com/blog/achieving-retail-efficiency-with-ai).
- **Industrial Automation**: Supporting quality control and defect detection in [manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing).

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

## Ultralytics YOLOv8

**Authors:** Glenn Jocher, Ayush Chaurasia, and Jing Qiu  
**Organization:** [Ultralytics](https://www.ultralytics.com/)  
**Date:** 2023-01-10  
**GitHub:** <https://github.com/ultralytics/ultralytics>  
**Docs:** <https://docs.ultralytics.com/models/yolov8/>

Ultralytics YOLOv8 set a new standard for real-time object detection upon its release, quickly becoming one of the most popular vision models in the world. It introduced key architectural changes, such as an anchor-free detection head and the C2f backbone module, which provided a significant leap in performance over previous versions. Like YOLO11, YOLOv8 is a versatile, multi-task model that has been extensively validated across countless real-world applications.

### Architecture and Key Features

YOLOv8's design focuses on a strong balance between speed and accuracy. Its anchor-free approach reduces the number of box predictions, simplifying the post-processing pipeline and improving inference speed. The model is highly scalable, with variants ranging from the lightweight 'n' (nano) version for mobile and [edge AI](https://www.ultralytics.com/glossary/edge-ai) to the powerful 'x' (extra-large) version for maximum accuracy. YOLOv8 is fully integrated into the Ultralytics ecosystem, benefiting from a simple API, comprehensive guides, and tools like the [Ultralytics HUB](https://www.ultralytics.com/hub) for no-code training and deployment.

### Strengths

- **Proven Performance:** A highly reliable and widely adopted model that delivers strong results across all supported tasks.
- **Excellent Speed-Accuracy Trade-off:** Offers a fantastic balance that made it a go-to choice for a wide variety of applications.
- **Mature Ecosystem:** Benefits from extensive community support, tutorials, and third-party integrations built up since its release.
- **Versatility:** Supports the same wide range of vision tasks as YOLO11, making it a powerful all-in-one solution.

### Weaknesses

- While still a top performer, it is generally surpassed by YOLO11 in both accuracy and CPU inference speed across all model sizes.
- Larger models have a higher parameter and FLOP count compared to their YOLO11 counterparts, leading to greater computational requirements.

### Use Cases

YOLOv8 remains a formidable and highly relevant model, excelling in applications where it has been widely deployed and tested:

- **Agriculture**: Used for crop monitoring, pest detection, and yield estimation in [smart farming](https://www.ultralytics.com/blog/sowing-success-ai-in-agriculture).
- **Healthcare**: Assists in [medical image analysis](https://www.ultralytics.com/glossary/medical-image-analysis) for tasks like detecting cells or anomalies.
- **Environmental Monitoring**: Deployed for [wildlife tracking](https://www.ultralytics.com/blog/ai-in-wildlife-conservation) and monitoring environmental changes.
- **Smart Cities**: Powers applications like [traffic management](https://www.ultralytics.com/blog/ai-in-traffic-management-from-congestion-to-coordination) and public safety monitoring.

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## Performance Head-to-Head: YOLO11 vs. YOLOv8

The primary distinction between YOLO11 and YOLOv8 lies in their performance metrics. YOLO11 consistently delivers higher accuracy (mAP) with a more efficient architecture, resulting in fewer parameters and FLOPs. This architectural optimization is particularly evident in CPU inference speeds, where YOLO11 models are substantially faster than their YOLOv8 equivalents. While YOLOv8n has a slight edge in GPU latency, YOLO11 models from 's' to 'x' are faster on GPU as well, making YOLO11 the superior choice for most new projects.

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

## Architectural Evolution and Ecosystem

YOLO11 is a direct evolution of YOLOv8, building upon its successful design principles while introducing targeted optimizations. Both models share the same core philosophy of being fast, accurate, and easy to use. They are developed and maintained within the unified [Ultralytics](https://github.com/ultralytics/ultralytics) repository, ensuring a consistent and streamlined user experience.

This shared ecosystem is a major advantage for developers. Migrating a project from YOLOv8 to YOLO11 is straightforward, allowing teams to leverage the performance gains of the newer model with minimal code changes. The ecosystem provides:

- **A simple and consistent API** for [training](https://docs.ultralytics.com/modes/train/), [validation](https://docs.ultralytics.com/modes/val/), and [prediction](https://docs.ultralytics.com/modes/predict/).
- **Extensive documentation** with numerous guides and examples.
- **Efficient training workflows** with readily available pre-trained weights on datasets like [COCO](https://docs.ultralytics.com/datasets/detect/coco/).
- **Lower memory requirements** during training and inference compared to other model types like [Transformers](https://www.ultralytics.com/glossary/transformer).
- **A vibrant open-source community** for support and collaboration.

## Conclusion: Which Model Should You Choose?

For new projects or those requiring the best possible performance, **YOLO11 is the clear choice**. It offers superior accuracy and faster inference speeds, particularly on CPUs, with a more efficient architecture. Its advancements make it the new state-of-the-art for real-time object detection.

**YOLOv8 remains an excellent and highly reliable model**. It is a great option for existing projects that are already optimized for its architecture or in scenarios where its extensive track record and vast number of third-party integrations are a key consideration.

Ultimately, both models represent the pinnacle of real-time object detection, and the choice depends on your specific project needs. However, with its clear performance advantages and seamless integration into the Ultralytics ecosystem, YOLO11 is poised to become the new standard for developers and researchers.

## Explore Other Models

While YOLO11 and YOLOv8 are leading choices, the field of computer vision is constantly evolving. You may also be interested in comparing them with other powerful models available in the Ultralytics ecosystem, such as [YOLOv10](https://docs.ultralytics.com/models/yolov10/), [YOLOv9](https://docs.ultralytics.com/models/yolov9/), and the transformer-based [RT-DETR](https://docs.ultralytics.com/models/rtdetr/). Explore our full range of [model comparisons](https://docs.ultralytics.com/compare/) to find the perfect fit for your project.
