---
comments: true
description: Compare RTDETRv2 and YOLOv8 for object detection. Explore architecture, performance, and use cases to select the best model for your needs.
keywords: RTDETRv2, YOLOv8, object detection, computer vision, model comparison, deep learning, transformer architecture, real-time AI, Ultralytics
---

# Model Comparison: RTDETRv2 vs YOLOv8 for Object Detection

When choosing a computer vision model for object detection, it's essential to understand the strengths and weaknesses of different architectures. This page offers a detailed technical comparison between Ultralytics YOLOv8 and RTDETRv2, both state-of-the-art models. We'll explore their architectural approaches, performance benchmarks, and suitable applications to help you select the best model for your specific needs.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["RTDETRv2", "YOLOv8"]'></canvas>

## YOLOv8: Optimized for Speed and Flexibility

Ultralytics YOLOv8, launched on January 10, 2023 by Glenn Jocher, Ayush Chaurasia, and Jing Qiu from Ultralytics, is the latest iteration in the YOLO series, renowned for its speed and efficiency. It is designed to be versatile and user-friendly across a broad spectrum of object detection tasks.

### Architecture and Features

YOLOv8 maintains a single-stage detection approach, prioritizing rapid inference. Key architectural features include an anchor-free detection head, a flexible backbone, and a refined loss function. These enhancements contribute to its balance of speed and accuracy, making it suitable for real-time applications. For a deeper understanding of YOLO architectures, refer to our guide on [object detection architectures](https://www.ultralytics.com/glossary/object-detection-architectures).

### Performance Metrics

YOLOv8 offers a range of model sizes, from YOLOv8n to YOLOv8x, allowing users to scale performance according to computational resources. Detailed performance metrics can be found in our [YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/).

### Use Cases

YOLOv8's adaptability makes it ideal for diverse applications. Its speed is crucial for real-time systems like [AI in robotics](https://www.ultralytics.com/glossary/robotics) and [autonomous vehicles](https://www.ultralytics.com/solutions/ai-in-self-driving). It's also effective in industrial automation for tasks like [improving manufacturing with computer vision](https://www.ultralytics.com/blog/improving-manufacturing-with-computer-vision) and in [AI in agriculture](https://www.ultralytics.com/solutions/ai-in-agriculture). Explore more use cases on our [solutions page](https://www.ultralytics.com/solutions).

### Strengths and Weaknesses

**Strengths:**

- **Speed:** Excels in inference speed, crucial for real-time applications.
- **Efficiency:** Computationally efficient, suitable for various hardware including edge devices.
- **User-Friendly:** Well-documented with a user-friendly [Python package](https://docs.ultralytics.com/usage/python/).
- **Scalability:** Offers multiple model sizes to match different needs.

**Weaknesses:**

- **Accuracy in Complex Scenes:** As a single-stage detector, it may have slightly lower accuracy in highly complex scenes compared to two-stage detectors, though YOLOv8 has minimized this gap.
- **Hyperparameter Tuning:** Optimal performance may require careful [hyperparameter tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning/).

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## RTDETRv2: High Accuracy for Real-Time Needs

RTDETRv2, or Real-Time Detection Transformer version 2, was introduced by Baidu on April 17, 2023, with authors Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu. This model leverages Transformer architecture to achieve high accuracy while maintaining real-time performance.

### Architecture and Features

RTDETRv2 adopts a transformer-based architecture, enabling it to capture global context within images more effectively than CNN-based models. It uses a hybrid encoder combining CNNs and Transformers, and an anchor-free detection approach. This design allows for robust feature extraction and high accuracy, especially in complex scenarios. More details are available in the [RT-DETR Arxiv paper](https://arxiv.org/abs/2304.08069).

### Performance Metrics

RTDETRv2 is designed for high accuracy in real-time object detection. The model's performance metrics, including mAP and inference speed, are detailed in the comparison table below and further explored in the [official RT-DETR GitHub repository](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch).

### Use Cases

RTDETRv2 is particularly suited for applications where high detection accuracy is paramount and computational resources are sufficient. Ideal use cases include [AI in self-driving cars](https://www.ultralytics.com/solutions/ai-in-self-driving) requiring precise environmental perception, advanced robotics for accurate object interaction, and [AI in healthcare](https://www.ultralytics.com/solutions/ai-in-healthcare), such as in medical imaging analysis for precise anomaly detection.

### Strengths and Weaknesses

**Strengths:**

- **High Accuracy:** Transformer-based architecture provides superior object detection accuracy.
- **Real-time Performance:** Optimized for real-time applications, especially with GPU acceleration.
- **Robust Feature Extraction:** Vision Transformers excel in capturing global context.

**Weaknesses:**

- **Larger Model Size:** Transformer models can be larger, requiring more parameters and FLOPs.
- **Computational Resources:** May demand more computational power compared to lighter models like YOLOv8, especially for larger variants.

[Learn more about RTDETRv2](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

## Model Comparison Table

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | 54.3                 | -                              | 15.03                               | 76                 | 259               |
|            |                       |                      |                                |                                     |                    |                   |
| YOLOv8n    | 640                   | 37.3                 | 80.4                           | 1.47                                | 3.2                | 8.7               |
| YOLOv8s    | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m    | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l    | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x    | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |

## Conclusion

Both RTDETRv2 and YOLOv8 are powerful models, but they cater to different priorities. Choose **RTDETRv2** when accuracy is paramount and you have sufficient computational resources. Opt for **YOLOv8** when speed and efficiency are critical, especially for real-time applications and resource-limited environments.

For other object detection models, consider exploring [YOLOv7](https://docs.ultralytics.com/models/yolov7/), [YOLOv9](https://docs.ultralytics.com/models/yolov9/), and [YOLO11](https://docs.ultralytics.com/models/yolo11/) within the YOLO family, or [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/) for a model optimized through Neural Architecture Search. Refer to the [Ultralytics documentation](https://docs.ultralytics.com/models/) and [GitHub repository](https://github.com/ultralytics/ultralytics) for more detailed information.
