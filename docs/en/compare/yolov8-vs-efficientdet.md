---
comments: true
description: Discover the key differences between YOLOv8 and EfficientDet for object detection. Compare performance, accuracy, speed, use cases, and scalability.
keywords: YOLOv8, EfficientDet, object detection, model comparison, Ultralytics, real-time detection, accuracy, scalability, AI models, computer vision
---

# Model Comparison: YOLOv8 vs EfficientDet for Object Detection

<script async src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv8", "EfficientDet"]'></canvas>

Choosing the right object detection model is crucial for computer vision projects. Ultralytics YOLOv8 and EfficientDet are both popular choices, but they cater to different needs. This page provides a detailed technical comparison to help you make an informed decision.

## Architecture

**Ultralytics YOLOv8** is a state-of-the-art, single-stage object detector, known for its speed and efficiency. It builds upon previous YOLO versions with architectural improvements like a new backbone network, anchor-free detection head, and a refined loss function. YOLOv8 is designed for real-time object detection across various applications. You can find more details in the [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/).

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

**EfficientDet**, developed by Google, is also a highly efficient object detection model, but with a focus on scalability and accuracy across different model sizes. EfficientDet utilizes a bi-directional feature pyramid network and compound scaling to optimize performance and parameter efficiency. More information can be found in the original [EfficientDet paper](https://arxiv.org/abs/1911.09070).

## Performance Metrics

The table below summarizes the performance of different variants of YOLOv8 and EfficientDet models. Key metrics include mAP (mean Average Precision), inference speed, and model size.

| Model           | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| --------------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv8n         | 640                   | 37.3                 | 80.4                           | 1.47                                | 3.2                | 8.7               |
| YOLOv8s         | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m         | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l         | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x         | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |
|                 |                       |                      |                                |                                     |                    |                   |
| EfficientDet-d0 | 640                   | 34.6                 | 10.2                           | 3.92                                | 3.9                | 2.54              |
| EfficientDet-d1 | 640                   | 40.5                 | 13.5                           | 7.31                                | 6.6                | 6.1               |
| EfficientDet-d2 | 640                   | 43.0                 | 17.7                           | 10.92                               | 8.1                | 11.0              |
| EfficientDet-d3 | 640                   | 47.5                 | 28.0                           | 19.59                               | 12.0               | 24.9              |
| EfficientDet-d4 | 640                   | 49.7                 | 42.8                           | 33.55                               | 20.7               | 55.2              |
| EfficientDet-d5 | 640                   | 51.5                 | 72.5                           | 67.86                               | 33.7               | 130.0             |
| EfficientDet-d6 | 640                   | 52.6                 | 92.8                           | 89.29                               | 51.9               | 226.0             |
| EfficientDet-d7 | 640                   | 53.7                 | 122.0                          | 128.07                              | 51.9               | 325.0             |

**YOLOv8 Strengths:**

- **Speed:** YOLOv8 excels in inference speed, making it suitable for real-time applications.
- **Ease of Use:** Ultralytics provides excellent documentation and a user-friendly Python package, simplifying model training and deployment. See the [Ultralytics YOLO Docs](https://docs.ultralytics.com/guides/).
- **Versatility:** YOLOv8 supports various tasks beyond object detection, including [segmentation](https://docs.ultralytics.com/tasks/segment/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), and [classification](https://docs.ultralytics.com/tasks/classify/).
- **Active Community & Support:** Benefit from a large and active open-source community and comprehensive [Ultralytics Support](https://www.ultralytics.com/support).

**YOLOv8 Weaknesses:**

- **Accuracy vs. Larger EfficientDet:** While highly accurate, larger EfficientDet models (D5-D7) can achieve slightly higher mAP in some benchmarks, though at a speed tradeoff.

**EfficientDet Strengths:**

- **Scalability and Efficiency:** EfficientDet offers a range of models (D0-D7) allowing users to choose the best trade-off between accuracy and computational cost.
- **Compound Scaling:** The compound scaling method efficiently scales up model dimensions for improved performance with increased model size.
- **Accuracy for Larger Models:** EfficientDet-D7 achieves very high accuracy, competitive with more complex two-stage detectors.

**EfficientDet Weaknesses:**

- **Speed:** Generally slower inference speeds compared to YOLOv8, especially the larger EfficientDet variants.
- **Complexity:** Implementation and fine-tuning might be slightly more complex than YOLOv8 for some users.

## Use Cases

**YOLOv8 Ideal Use Cases:**

- **Real-time Object Detection:** Applications requiring high-speed inference, such as [security alarm systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8), [autonomous vehicles](https://www.ultralytics.com/solutions/ai-in-self-driving), and [robotic systems](https://www.ultralytics.com/glossary/robotics).
- **Edge Deployment:** Efficient deployment on edge devices like [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/) and [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/) due to its speed and smaller model sizes.
- **Applications Across Industries:** Versatile enough for use in [agriculture](https://www.ultralytics.com/solutions/ai-in-agriculture), [manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing), [healthcare](https://www.ultralytics.com/solutions/ai-in-healthcare), and [smart cities](https://www.ultralytics.com/blog/computer-vision-ai-in-smart-cities).

**EfficientDet Ideal Use Cases:**

- **High-Accuracy Requirements:** Scenarios where maximizing detection accuracy is paramount, even if it means sacrificing some speed.
- **Resource-Constrained Environments (Smaller Models):** EfficientDet-D0 and D1 offer good performance with relatively small model sizes, suitable for resource-limited devices.
- **Research and Development:** A good choice for research purposes where exploring different accuracy/speed trade-offs is essential.

## Conclusion

Both YOLOv8 and EfficientDet are excellent object detection models. YOLOv8 is generally preferred for applications prioritizing speed and ease of use, while EfficientDet shines when accuracy and scalability across different model sizes are the primary concerns. Your choice will depend on the specific requirements of your project and the balance you need to strike between speed and accuracy.

For users interested in other models, Ultralytics also offers [YOLOv5](https://docs.ultralytics.com/models/yolov5/), [YOLOv7](https://docs.ultralytics.com/models/yolov7/), [YOLOv9](https://docs.ultralytics.com/models/yolov9/) and [YOLOv10](https://docs.ultralytics.com/models/yolov10/), along with models like [RT-DETR](https://docs.ultralytics.com/models/rtdetr/) and [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/).

| Model           | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| --------------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv8n         | 640                   | 37.3                 | 80.4                           | 1.47                                | 3.2                | 8.7               |
| YOLOv8s         | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m         | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l         | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x         | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |
|                 |                       |                      |                                |                                     |                    |                   |
| EfficientDet-d0 | 640                   | 34.6                 | 10.2                           | 3.92                                | 3.9                | 2.54              |
| EfficientDet-d1 | 640                   | 40.5                 | 13.5                           | 7.31                                | 6.6                | 6.1               |
| EfficientDet-d2 | 640                   | 43.0                 | 17.7                           | 10.92                               | 8.1                | 11.0              |
| EfficientDet-d3 | 640                   | 47.5                 | 28.0                           | 19.59                               | 12.0               | 24.9              |
| EfficientDet-d4 | 640                   | 49.7                 | 42.8                           | 33.55                               | 20.7               | 55.2              |
| EfficientDet-d5 | 640                   | 51.5                 | 72.5                           | 67.86                               | 33.7               | 130.0             |
| EfficientDet-d6 | 640                   | 52.6                 | 92.8                           | 89.29                               | 51.9               | 226.0             |
| EfficientDet-d7 | 640                   | 53.7                 | 122.0                          | 128.07                              | 51.9               | 325.0             |

[Explore Ultralytics Models](https://docs.ultralytics.com/models/){ .md-button }
