---
comments: true
description: Compare YOLO11 and YOLOv9 for object detection. Explore innovations, benchmarks, and use cases to select the best model for your tasks.
keywords: YOLO11, YOLOv9, object detection, model comparison, benchmarks, Ultralytics, real-time processing, machine learning, computer vision
---

# YOLOv9 vs YOLO11: A Technical Comparison

This page provides a detailed technical comparison between two state-of-the-art object detection models: [YOLOv9](https://docs.ultralytics.com/models/yolov9/) and [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/). We analyze their architectures, performance metrics, and use cases to assist you in selecting the most suitable model for your computer vision needs, highlighting the strengths of the Ultralytics ecosystem.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv9", "YOLO11"]'></canvas>

## YOLOv9: Addressing Information Loss

YOLOv9 represents a significant development in real-time object detection, focusing on overcoming information loss in deep neural networks.

**Technical Details:**

- **Authors:** Chien-Yao Wang, Hong-Yuan Mark Liao
- **Organization:** Institute of Information Science, Academia Sinica, Taiwan
- **Date:** 2024-02-21
- **Arxiv Link:** <https://arxiv.org/abs/2402.13616>
- **GitHub Link:** <https://github.com/WongKinYiu/yolov9>
- **Docs Link:** <https://docs.ultralytics.com/models/yolov9/>

**Architecture and Key Features:**
YOLOv9 introduces Programmable Gradient Information (PGI) to ensure deep networks retain crucial data across layers, mitigating information bottlenecks. It also utilizes the Generalized Efficient Layer Aggregation Network (GELAN), an optimized architecture designed for superior parameter utilization and computational efficiency. These innovations aim to improve accuracy without significantly increasing computational demands.

**Performance Metrics:**
YOLOv9 models demonstrate strong performance on the [COCO dataset](https://docs.ultralytics.com/datasets/detect/coco/). For instance, YOLOv9-E achieves a high mAP<sup>val</sup> 50-95 of 55.6% with 57.3M parameters. While specific CPU inference speeds are not readily available in the source repository, GPU speeds show competitive performance.

**Strengths:**

- **High Accuracy:** Achieves state-of-the-art mAP scores, particularly with larger models.
- **Innovative Architecture:** PGI and GELAN address fundamental challenges in deep learning information flow.
- **Efficiency:** Aims for a good balance between accuracy and computational cost compared to some prior models.

**Weaknesses:**

- **Ecosystem Integration:** May require more effort to integrate into streamlined workflows compared to models developed within the Ultralytics ecosystem.
- **Task Versatility:** Primarily focused on object detection and segmentation, potentially lacking the broader task support found in Ultralytics models.
- **Training Resources:** Training YOLOv9 models can be resource-intensive and potentially slower than comparable Ultralytics models.

**Ideal Use Cases:**
YOLOv9 is well-suited for applications where maximizing accuracy is critical, such as:

- Advanced Driver-Assistance Systems (ADAS).
- Detailed analysis of high-resolution imagery.
- Scenarios where its specific architectural innovations offer advantages.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

## Ultralytics YOLO11: The Cutting Edge

[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/), the latest iteration from Ultralytics, builds upon the success of models like [YOLOv8](https://docs.ultralytics.com/models/yolov8/), focusing on enhancing speed, accuracy, and usability across a wide range of vision tasks.

**Technical Details:**

- **Authors:** Glenn Jocher, Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com/)
- **Date:** 2024-09-27
- **GitHub Link:** <https://github.com/ultralytics/ultralytics>
- **Docs Link:** <https://docs.ultralytics.com/models/yolo11/>

**Architecture and Key Features:**
YOLO11 features a refined architecture optimized for efficient feature extraction and faster processing. It achieves high accuracy with fewer parameters compared to many predecessors, enhancing real-time performance. Crucially, YOLO11 benefits from the robust Ultralytics ecosystem:

- **Ease of Use:** Offers a streamlined user experience with a simple [Python API](https://docs.ultralytics.com/usage/python/) and extensive [documentation](https://docs.ultralytics.com/models/yolo11/).
- **Well-Maintained Ecosystem:** Integrates seamlessly with [Ultralytics HUB](https://www.ultralytics.com/hub) for no-code training and deployment, benefits from active development, strong community support, and frequent updates.
- **Performance Balance:** Delivers a strong trade-off between speed and accuracy, suitable for diverse real-world scenarios.
- **Memory Efficiency:** Typically requires lower memory usage during training and inference compared to other model types like transformers.
- **Versatility:** Supports multiple tasks including [object detection](https://docs.ultralytics.com/tasks/detect/), [instance segmentation](https://docs.ultralytics.com/tasks/segment/), [image classification](https://docs.ultralytics.com/tasks/classify/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), and [oriented bounding boxes (OBB)](https://docs.ultralytics.com/tasks/obb/).
- **Training Efficiency:** Offers efficient training processes with readily available pre-trained weights.

**Performance Metrics:**
YOLO11 models provide a spectrum of options. YOLO11n achieves 39.5 mAP<sup>val</sup> 50-95 with only 2.6M parameters and impressive speeds (56.1ms CPU ONNX, 1.5ms T4 TensorRT10). YOLO11x reaches 54.7 mAP<sup>val</sup> 50-95, offering high accuracy.

**Strengths:**

- **Superior Speed & Efficiency:** Excellent inference speeds, especially on CPU and for smaller models on GPU.
- **High Accuracy:** Competitive mAP scores across model sizes.
- **Multi-Task Support:** Highly versatile for various computer vision applications.
- **User-Friendly:** Easy to train, validate, and deploy within the Ultralytics framework.
- **Optimized for Deployment:** Performs well across diverse hardware, from edge devices ([NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/), [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/)) to cloud infrastructure.

**Weaknesses:**

- **Speed-Accuracy Trade-off:** Smaller models prioritize speed, potentially sacrificing some accuracy compared to the largest variants.
- **One-Stage Detector Limits:** Like most YOLO models, may face challenges with extremely small objects compared to specialized two-stage detectors.

**Ideal Use Cases:**
YOLO11 excels in applications demanding high speed and efficiency without compromising accuracy:

- Real-time video analytics ([security](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8), traffic management).
- Robotics and autonomous systems.
- Industrial automation ([quality control](https://www.ultralytics.com/solutions/ai-in-manufacturing)).
- Edge computing applications.

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

## Performance Comparison: YOLOv9 vs YOLO11

The following table compares the performance of YOLOv9 and YOLO11 models on the COCO dataset based on available metrics.

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv9t | 640                   | 38.3                 | -                              | 2.3                                 | **2.0**            | 7.7               |
| YOLOv9s | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e | 640                   | **55.6**             | -                              | 16.77                               | 57.3               | 189.0             |
|         |                       |                      |                                |                                     |                    |                   |
| YOLO11n | 640                   | 39.5                 | **56.1**                       | **1.5**                             | 2.6                | **6.5**           |
| YOLO11s | 640                   | 47.0                 | 90.0                           | 2.5                                 | 9.4                | 21.5              |
| YOLO11m | 640                   | 51.5                 | 183.2                          | 4.7                                 | 20.1               | 68.0              |
| YOLO11l | 640                   | 53.4                 | 238.6                          | 6.2                                 | 25.3               | 86.9              |
| YOLO11x | 640                   | 54.7                 | 462.8                          | 11.3                                | 56.9               | 194.9             |

YOLOv9 demonstrates strong peak accuracy (mAP) with its larger variants (YOLOv9e). However, Ultralytics YOLO11 excels in inference speed, particularly the nano (n) model on both CPU and GPU (T4 TensorRT10), while maintaining highly competitive accuracy across all sizes. YOLO11n also boasts the lowest FLOPs. YOLO11 models generally offer a better balance of speed, accuracy, and parameter efficiency, especially when considering the ease of use and comprehensive support within the Ultralytics ecosystem. The availability of CPU speed metrics for YOLO11 further highlights its suitability for diverse deployment scenarios.

## Conclusion

Both YOLOv9 and Ultralytics YOLO11 represent significant advancements in object detection. YOLOv9 introduces innovative architectural concepts like PGI and GELAN to push accuracy boundaries. Ultralytics YOLO11, while also achieving excellent accuracy, prioritizes a balance of speed, efficiency, and versatility, backed by a mature, user-friendly ecosystem. For developers seeking state-of-the-art performance combined with ease of use, extensive documentation, multi-task capabilities, and seamless integration for training and deployment, YOLO11 is often the preferred choice.

For users interested in exploring other models, Ultralytics offers a range of options including [YOLOv5](https://docs.ultralytics.com/models/yolov5/), [YOLOv8](https://docs.ultralytics.com/models/yolov8/), and [YOLOv10](https://docs.ultralytics.com/models/yolov10/). Comparisons with other architectures like [RT-DETR](https://docs.ultralytics.com/models/rtdetr/) are also available in the [Ultralytics documentation](https://docs.ultralytics.com/compare/).
