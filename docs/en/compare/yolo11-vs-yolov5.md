---
comments: true
description: Explore the comprehensive comparison between YOLO11 and YOLOv5. Learn about their architectures, performance metrics, use cases, and strengths.
keywords: YOLO11 vs YOLOv5,Yolo comparison,Yolo models,object detection,Yolo performance,Yolo benchmarks,Ultralytics,Yolo architecture
---

# YOLO11 vs YOLOv5: A Detailed Comparison

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO11", "YOLOv5"]'></canvas>

This page provides a technical comparison between two popular object detection models: YOLO11 and YOLOv5, both developed by Ultralytics. We will analyze their architectures, performance metrics, and use cases to help you choose the right model for your computer vision needs.

## YOLO11

**Authors**: Glenn Jocher and Jing Qiu
**Organization**: Ultralytics
**Date**: 2024-09-27
**GitHub Link**: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
**Docs Link**: [https://docs.ultralytics.com/models/yolo11/](https://docs.ultralytics.com/models/yolo11/)

YOLO11 is the latest state-of-the-art object detection model from Ultralytics, building upon previous YOLO versions to offer enhanced performance and flexibility. It is designed for speed and accuracy across various vision tasks, including object detection, segmentation, classification, and pose estimation.

### Architecture and Key Features

YOLO11 introduces several architectural improvements, focusing on efficiency and higher accuracy. While specific architectural details are continuously evolving, YOLO11 generally emphasizes refined backbone networks and optimized head designs to boost detection capabilities without significantly increasing computational cost. It leverages advancements in network architecture to achieve better feature extraction and more efficient processing. For more in-depth architectural insights, refer to the [Ultralytics documentation](https://docs.ultralytics.com/).

### Performance Metrics

YOLO11 demonstrates superior performance compared to YOLOv5 in terms of accuracy, especially when considering the trade-off with speed. The model achieves a higher mAP (mean Average Precision) while maintaining competitive inference speeds, making it suitable for applications requiring high detection accuracy.

| Model   | size (pixels) | mAP<sup>val</sup><sub>50-95</sub> | Speed<sup>CPU ONNX</sup> (ms) | Speed<sup>T4 TensorRT10</sup> (ms) | params (M) | FLOPs (B) |
| ------- | ------------- | --------------------------------- | ----------------------------- | ---------------------------------- | ---------- | --------- |
| YOLO11n | 640           | 39.5                              | 56.1                          | 1.5                                | 2.6        | 6.5       |
| YOLO11s | 640           | 47.0                              | 90.0                          | 2.5                                | 9.4        | 21.5      |
| YOLO11m | 640           | 51.5                              | 183.2                         | 4.7                                | 20.1       | 68.0      |
| YOLO11l | 640           | 53.4                              | 238.6                         | 6.2                                | 25.3       | 86.9      |
| YOLO11x | 640           | 54.7                              | 462.8                         | 11.3                               | 56.9       | 194.9     |

### Use Cases

YOLO11 is ideally suited for applications that demand the highest levels of accuracy in object detection. These include:

- **High-precision applications**: Medical imaging, satellite image analysis, and quality control in manufacturing where accuracy is paramount. See how YOLO11 can be used for [tumor detection in medical imaging](https://www.ultralytics.com/blog/using-yolo11-for-tumor-detection-in-medical-imaging).
- **Complex scene understanding**: Scenarios with dense objects or challenging environments where robust detection is necessary.
- **Advanced analytics**: Applications requiring detailed and reliable object detection for subsequent analysis and decision-making.

**Strengths:**

- **Higher Accuracy**: Generally achieves better mAP than YOLOv5.
- **State-of-the-art**: Represents the latest advancements in the YOLO series.
- **Versatile**: Supports multiple vision tasks beyond object detection.

**Weaknesses:**

- **Potentially Slower**: Larger models may have slower inference speeds compared to YOLOv5, especially the smaller variants.
- **Newer Model**: Being newer, the community and available resources might be still growing compared to the more established YOLOv5.

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

## YOLOv5

**Authors**: Glenn Jocher
**Organization**: Ultralytics
**Date**: 2020-06-26
**GitHub Link**: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
**Docs Link**: [https://docs.ultralytics.com/models/yolov5/](https://docs.ultralytics.com/models/yolov5/)

YOLOv5 is a widely adopted and highly optimized object detection model known for its excellent balance of speed and accuracy. It has been a popular choice for researchers and practitioners due to its ease of use and strong performance in real-world applications.

### Architecture and Key Features

YOLOv5 utilizes a streamlined architecture that prioritizes efficiency and speed. It employs techniques like CSPBottleneck layers and focuses on optimizing the network for fast inference. YOLOv5 is available in various sizes (n, s, m, l, x), offering flexibility to choose models based on specific speed and accuracy requirements. Further architectural details can be found in the [YOLOv5 documentation](https://docs.ultralytics.com/yolov5/).

### Performance Metrics

YOLOv5 excels in providing a good balance between accuracy and speed. It offers fast inference times, making it suitable for real-time object detection tasks. While slightly less accurate than YOLO11 in some benchmarks, its speed and efficiency are highly valued in many applications.

| Model   | size (pixels) | mAP<sup>val</sup><sub>50-95</sub> | Speed<sup>CPU ONNX</sup> (ms) | Speed<sup>T4 TensorRT10</sup> (ms) | params (M) | FLOPs (B) |
| ------- | ------------- | --------------------------------- | ----------------------------- | ---------------------------------- | ---------- | --------- |
| YOLOv5n | 640           | 28.0                              | 73.6                          | 1.12                               | 2.6        | 7.7       |
| YOLOv5s | 640           | 37.4                              | 120.7                         | 1.92                               | 9.1        | 24.0      |
| YOLOv5m | 640           | 45.4                              | 233.9                         | 4.03                               | 25.1       | 64.2      |
| YOLOv5l | 640           | 49.0                              | 408.4                         | 6.61                               | 53.2       | 135.0     |
| YOLOv5x | 640           | 50.7                              | 763.2                         | 11.89                              | 97.2       | 246.4     |

### Use Cases

YOLOv5 is well-suited for applications where real-time performance and efficiency are critical:

- **Real-time detection**: Applications like video surveillance, autonomous driving, and robotics where fast inference is crucial. Explore [AI in robotics](https://www.ultralytics.com/blog/from-algorithms-to-automation-ais-role-in-robotics).
- **Edge devices**: Deployment on resource-constrained devices due to its efficient design and smaller model sizes. Learn about [deploying YOLOv5 on NVIDIA Jetson](https://docs.ultralytics.com/yolov5/tutorials/running_on_jetson_nano/).
- **High-throughput processing**: Scenarios requiring processing large volumes of images or video streams quickly.

**Strengths:**

- **High Speed**: Known for its fast inference speed, especially the smaller models.
- **Efficiency**: Optimized for resource-constrained environments.
- **Mature and Well-Documented**: Extensive community support and resources available.

**Weaknesses:**

- **Lower Accuracy**: Generally slightly lower mAP compared to YOLO11 for larger models.
- **Architecture**: While efficient, its architecture may not fully leverage the latest advancements in deep learning compared to newer models.

[Learn more about YOLOv5](https://docs.ultralytics.com/models/yolov5/){ .md-button }

## Comparison Table

| Model   | size<br><sup>(pixels) | mAP<sup>val</sup><sub>50-95</sub> | Speed<br><sup>CPU ONNX</sup> (ms) | Speed<br><sup>T4 TensorRT10</sup> (ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | --------------------------------- | --------------------------------- | -------------------------------------- | ------------------ | ----------------- |
| YOLO11n | 640                   | 39.5                              | 56.1                              | 1.5                                    | 2.6                | 6.5               |
| YOLO11s | 640                   | 47.0                              | 90.0                              | 2.5                                    | 9.4                | 21.5              |
| YOLO11m | 640                   | 51.5                              | 183.2                             | 4.7                                    | 20.1               | 68.0              |
| YOLO11l | 640                   | 53.4                              | 238.6                             | 6.2                                    | 25.3               | 86.9              |
| YOLO11x | 640                   | 54.7                              | 462.8                             | 11.3                                   | 56.9               | 194.9             |
|         |                       |                                   |                                   |                                        |                    |                   |
| YOLOv5n | 640                   | 28.0                              | 73.6                              | 1.12                                   | 2.6                | 7.7               |
| YOLOv5s | 640                   | 37.4                              | 120.7                             | 1.92                                   | 9.1                | 24.0              |
| YOLOv5m | 640                   | 45.4                              | 233.9                             | 4.03                                   | 25.1               | 64.2              |
| YOLOv5l | 640                   | 49.0                              | 408.4                             | 6.61                                   | 53.2               | 135.0             |
| YOLOv5x | 640                   | 50.7                              | 763.2                             | 11.89                                  | 97.2               | 246.4             |

For users interested in other models, Ultralytics also offers YOLOv8, YOLOv7, and YOLOv6, each with its own strengths and optimizations. Explore the [Ultralytics Models documentation](https://docs.ultralytics.com/models/) for more comparisons.
