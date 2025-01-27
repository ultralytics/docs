---
comments: true
description: Technical comparison of YOLOv5 and RT-DETR for object detection, focusing on architecture, performance, use cases, mAP, inference speed, and model size.
keywords: YOLOv5, RT-DETR, object detection, model comparison, computer vision, Ultralytics, performance, architecture, use cases, mAP, inference speed, model size
---

# YOLOv5 vs RT-DETR v2: A Detailed Model Comparison

Choosing the right object detection model is crucial for computer vision projects. Ultralytics YOLO offers a suite of models tailored for various needs. This page provides a technical comparison between [Ultralytics YOLOv5](https://docs.ultralytics.com/models/yolov5) and [RT-DETR v2](https://docs.ultralytics.com/models/rtdetr), highlighting their architectural differences, performance metrics, and ideal applications.

<script async src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv5", "RTDETRv2"]'></canvas>

## YOLOv5: Speed and Efficiency

[Ultralytics YOLOv5](https://docs.ultralytics.com/models/yolov5/) is a highly popular one-stage object detector known for its speed and efficiency. Its architecture is based on:

- **Backbone:** CSPDarknet53 for feature extraction.
- **Neck:** PANet for feature fusion.
- **Head:** YOLOv5 head for detection.

YOLOv5 comes in various sizes (n, s, m, l, x), offering a trade-off between speed and accuracy.

**Strengths:**

- **Speed:** YOLOv5 excels in inference speed, making it suitable for real-time applications.
- **Efficiency:** Models are relatively small and require less computational resources.
- **Versatility:** Adaptable to various hardware, including edge devices.
- **Ease of Use:** Well-documented and easy to implement with Ultralytics [Python package](https://pypi.org/project/ultralytics/) and [Ultralytics HUB](https://docs.ultralytics.com/hub/).

**Weaknesses:**

- **Accuracy:** While highly accurate, larger models like RT-DETR v2 may achieve higher mAP, especially on complex datasets.

**Use Cases:**

- Real-time object detection in video surveillance ([security alarm systems](https://docs.ultralytics.com/guides/security-alarm-system/)).
- Mobile and edge deployments ([Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/), [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/)).
- Applications requiring rapid processing, such as robotics ([ROS Quickstart](https://docs.ultralytics.com/guides/ros-quickstart/)) and [autonomous vehicles](https://www.ultralytics.com/solutions/ai-in-self-driving).

[Learn more about YOLOv5](https://docs.ultralytics.com/models/yolov5/){ .md-button }

## RT-DETR v2: Accuracy with Transformer Efficiency

[RT-DETR v2](https://docs.ultralytics.com/models/rtdetr/) represents a shift towards Transformer-based architectures for real-time object detection. It leverages:

- **Backbone:** Hybrid backbone combining CNNs and Transformers for efficient feature extraction.
- **Decoder:** Transformer decoder inspired by DETR (DEtection TRansformer) for direct set prediction, eliminating the need for Non-Maximum Suppression (NMS) in the model architecture.

RT-DETR v2 also offers different sizes (s, m, l, x) to balance accuracy and speed.

**Strengths:**

- **Accuracy:** RT-DETR v2 achieves state-of-the-art accuracy, particularly the larger models, due to its transformer-based architecture which excels at capturing global context.
- **Robustness:** DETR-style models are known for their robustness and ability to handle complex scenes.
- **NMS-free:** Simplifies the pipeline and potentially improves latency by removing the NMS post-processing step from the model itself.

**Weaknesses:**

- **Speed:** While optimized for real-time, RT-DETR v2 may be slightly slower in inference speed compared to smaller YOLOv5 models, especially on CPU.
- **Model Size:** Transformer-based models can be larger than traditional CNN-based models.

**Use Cases:**

- Applications prioritizing high accuracy object detection.
- Complex scene understanding and detailed image analysis.
- Scenarios where robustness to occlusion and cluttered backgrounds is important.
- Industrial inspection and quality control ([AI in Manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing)).
- Medical image analysis ([medical image analysis](https://www.ultralytics.com/glossary/medical-image-analysis)).

[Learn more about RT-DETR v2](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

## Model Comparison Table

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv5n    | 640                   | 28.0                 | 73.6                           | 1.12                                | 2.6                | 7.7               |
| YOLOv5s    | 640                   | 37.4                 | 120.7                          | 1.92                                | 9.1                | 24.0              |
| YOLOv5m    | 640                   | 45.4                 | 233.9                          | 4.03                                | 25.1               | 64.2              |
| YOLOv5l    | 640                   | 49.0                 | 408.4                          | 6.61                                | 53.2               | 135.0             |
| YOLOv5x    | 640                   | 50.7                 | 763.2                          | 11.89                               | 97.2               | 246.4             |
|            |                       |                      |                                |                                     |                    |                   |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | 54.3                 | -                              | 15.03                               | 76                 | 259               |

## Conclusion

Both YOLOv5 and RT-DETR v2 are powerful object detection models, each with its strengths. YOLOv5 is ideal when speed and efficiency are paramount, while RT-DETR v2 shines in scenarios demanding the highest accuracy. The choice between them depends on the specific requirements of your project.

Users might also be interested in exploring other Ultralytics YOLO models such as [YOLOv8](https://docs.ultralytics.com/models/yolov8/), [YOLOv10](https://docs.ultralytics.com/models/yolov10/), [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/), [YOLOv7](https://docs.ultralytics.com/models/yolov7/), [YOLOv9](https://docs.ultralytics.com/models/yolov9/) and [YOLOv6](https://docs.ultralytics.com/models/yolov6/) for different performance characteristics and architectural innovations.

For further details, refer to the official [Ultralytics Documentation](https://docs.ultralytics.com/models/) and the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics).
