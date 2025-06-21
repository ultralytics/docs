---
comments: true
description: Compare YOLOv8 and YOLO11 for object detection. Explore their performance, architecture, and best-use cases to find the right model for your needs.
keywords: YOLOv8, YOLO11, object detection, Ultralytics, YOLO comparison, machine learning, computer vision, inference speed, model accuracy
---

# YOLOv8 vs YOLO11: A Detailed Technical Comparison

When selecting a computer vision model, particularly for [object detection](https://docs.ultralytics.com/tasks/detect/), understanding the strengths and weaknesses of different architectures is essential. This page offers a detailed technical comparison between [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) and [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/), two state-of-the-art models from [Ultralytics](https://www.ultralytics.com) designed for object detection and other vision tasks. We will analyze their architectural nuances, performance benchmarks, and suitable applications to guide you in making an informed decision for your project.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv8", "YOLO11"]'></canvas>

## Ultralytics YOLOv8

**Authors:** Glenn Jocher, Ayush Chaurasia, and Jing Qiu  
**Organization:** [Ultralytics](https://www.ultralytics.com/)  
**Date:** 2023-01-10  
**GitHub:** <https://github.com/ultralytics/ultralytics>  
**Docs:** <https://docs.ultralytics.com/models/yolov8/>

Released in early 2023, YOLOv8 quickly became a benchmark for real-time object detection, offering a significant leap in performance over previous versions. It introduced an anchor-free detection mechanism and a new CSPDarknet53-based backbone, which improved both accuracy and speed. YOLOv8 is a highly versatile model, supporting a full range of vision AI tasks, including detection, [instance segmentation](https://docs.ultralytics.com/tasks/segment/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), and [image classification](https://docs.ultralytics.com/tasks/classify/).

### Architecture and Key Features

YOLOv8's architecture is built for efficiency and flexibility. Its anchor-free head reduces the number of box predictions, simplifying the post-processing pipeline and speeding up inference. The model was designed as a comprehensive framework, not just a single model, providing a unified platform for training models for various tasks. This integration into the Ultralytics ecosystem means users benefit from a streamlined workflow, from training to deployment, backed by extensive [documentation](https://docs.ultralytics.com/models/yolov8/) and a robust set of tools.

### Strengths

- **Proven Performance:** A highly reliable and widely adopted model that has set industry standards for performance and speed.
- **Task Versatility:** A single, unified framework capable of handling detection, segmentation, classification, and pose estimation.
- **Mature Ecosystem:** Benefits from a vast number of community tutorials, third-party integrations, and widespread deployment in production environments.
- **Ease of Use:** Features a simple [Python API](https://docs.ultralytics.com/usage/python/) and [CLI](https://docs.ultralytics.com/usage/cli/), making it accessible to both beginners and experts.

### Weaknesses

- While still a top performer, its accuracy and speed have been surpassed by its successor, YOLO11, especially in CPU-bound scenarios.
- Larger models (YOLOv8l, YOLOv8x) can be computationally intensive, requiring significant GPU resources for real-time performance.

### Use Cases

YOLOv8 remains an excellent choice for a wide range of applications, especially where stability and a mature ecosystem are valued. It excels in:

- **Industrial Automation:** For quality control and defect detection in [manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing).
- **Security Systems:** Powering advanced [security systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8) for real-time monitoring and intrusion detection.
- **Retail Analytics:** Improving [inventory management](https://www.ultralytics.com/blog/ai-for-smarter-retail-inventory-management) and analyzing customer behavior.

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## Ultralytics YOLO11

**Authors:** Glenn Jocher and Jing Qiu  
**Organization:** [Ultralytics](https://www.ultralytics.com/)  
**Date:** 2024-09-27  
**GitHub:** <https://github.com/ultralytics/ultralytics>  
**Docs:** <https://docs.ultralytics.com/models/yolo11/>

YOLO11 is the latest evolution in the Ultralytics YOLO series, engineered for superior accuracy and efficiency. Building on the strong foundation of YOLOv8, YOLO11 introduces architectural refinements that optimize feature extraction and processing. This results in higher detection precision with fewer parameters and faster inference speeds, particularly on CPUs. Like its predecessor, YOLO11 is a multi-task model supporting detection, segmentation, classification, pose estimation, and oriented bounding boxes (OBB) within the same streamlined framework.

### Architecture and Key Features

YOLO11 refines the network structure to achieve a better balance between computational cost and performance. It achieves higher accuracy with a lower parameter count and fewer FLOPs compared to YOLOv8, as shown in the performance table below. This efficiency makes it highly suitable for deployment across a wide range of hardware, from resource-constrained [edge devices](https://docs.ultralytics.com/guides/nvidia-jetson/) to powerful cloud servers. A key advantage of YOLO11 is its seamless integration into the **well-maintained Ultralytics ecosystem**, which ensures an excellent user experience, **efficient training** processes with readily available pre-trained weights, and **lower memory usage** during training and inference.

### Strengths

- **State-of-the-Art Accuracy:** Delivers higher mAP scores than YOLOv8 across all model sizes, setting a new standard for object detection.
- **Enhanced Efficiency:** Offers significantly faster inference speeds, especially on CPU, while requiring fewer parameters and FLOPs.
- **Performance Balance:** Provides an exceptional trade-off between speed and accuracy, making it ideal for diverse real-world applications.
- **Scalability and Versatility:** Performs well on various hardware and supports multiple computer vision tasks within a single, easy-to-use framework.
- **Well-Maintained Ecosystem:** Benefits from active development, strong community support via [GitHub](https://github.com/ultralytics/ultralytics/issues) and [Discord](https://discord.com/invite/ultralytics), and frequent updates.

### Weaknesses

- Being a newer model, it may initially have fewer third-party integrations compared to the more established YOLOv8.
- The largest models (e.g., YOLO11x) still require substantial computational power for training and deployment, a common trait of high-accuracy detectors.

### Use Cases

YOLO11 is the recommended choice for new projects that demand the highest levels of accuracy and real-time performance. Its efficiency makes it ideal for:

- **Robotics:** Enabling precise navigation and object interaction in [autonomous systems](https://www.ultralytics.com/solutions/ai-in-automotive).
- **Healthcare:** Assisting in [medical image analysis](https://www.ultralytics.com/glossary/medical-image-analysis) for applications like tumor detection.
- **Smart Cities:** Powering intelligent [traffic management](https://www.ultralytics.com/blog/optimizingtraffic-management-with-ultralytics-yolo11) and public safety systems.

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

## Performance Head-to-Head: YOLOv8 vs. YOLO11

The primary distinction between YOLOv8 and YOLO11 lies in performance. YOLO11 consistently outperforms YOLOv8 by delivering higher accuracy (mAP) with greater efficiency (fewer parameters and faster speeds). For instance, YOLO11l achieves a higher mAP (53.4) than YOLOv8l (52.9) with nearly 42% fewer parameters and is significantly faster on CPU. This trend holds across all model variants, making YOLO11 a more powerful and efficient successor.

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

## Conclusion and Recommendation

Both YOLOv8 and YOLO11 are exceptional models, but they serve slightly different needs.

- **YOLOv8** is a robust and mature model, making it a safe bet for projects that are already built upon it or that rely heavily on its extensive ecosystem of existing third-party tools and tutorials. It remains a formidable choice for a wide array of computer vision tasks.

- **YOLO11** is the clear winner in terms of performance and efficiency. It represents the cutting edge of real-time object detection. For any new project, YOLO11 is the recommended starting point. Its superior accuracy, faster inference speeds (especially on CPU), and more efficient architecture provide a significant advantage and future-proof your application. The continuous support and development within the Ultralytics ecosystem further solidify its position as the premier choice for developers and researchers.

For those interested in exploring other models, Ultralytics also supports a range of architectures, including the foundational [YOLOv5](https://docs.ultralytics.com/models/yolov5/), the recent [YOLOv9](https://docs.ultralytics.com/models/yolov9/), and transformer-based models like [RT-DETR](https://docs.ultralytics.com/models/rtdetr/). You can find more comparisons on our [model comparison page](https://docs.ultralytics.com/compare/).
