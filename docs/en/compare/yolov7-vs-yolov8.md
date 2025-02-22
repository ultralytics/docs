---
comments: true
description: Discover the key differences between YOLOv7 and YOLOv8 in terms of speed, accuracy, use cases, and performance for real-time object detection.
keywords: YOLOv7, YOLOv8, object detection, real-time, Ultralytics, model comparison, computer vision, deep learning, AI models, speed, accuracy, performance
---

# YOLOv7 vs YOLOv8: A Detailed Comparison for Object Detection

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv7", "YOLOv8"]'></canvas>

When it comes to real-time object detection, the YOLO (You Only Look Once) family of models has consistently pushed the boundaries of speed and accuracy. Ultralytics offers state-of-the-art YOLO models, and this page provides a technical comparison between two popular versions: YOLOv7 and YOLOv8, focusing on their object detection capabilities.

## YOLOv8: The Cutting Edge

[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) is the latest iteration in the YOLO series, building upon previous versions to offer enhanced performance and flexibility. It is designed to be a versatile and powerful model suitable for a wide range of object detection tasks.

**Architecture:** YOLOv8 adopts a flexible and modular architecture, allowing for easy customization and adaptation. It introduces improvements in the backbone network, detection head, and loss functions compared to its predecessors. The architecture is designed for optimal balance between speed and accuracy, making it suitable for real-time applications.

**Performance:** YOLOv8 achieves state-of-the-art performance across various model sizes. It offers a range of models (Nano, Small, Medium, Large, and Extra-Large), allowing users to select the best model based on their specific speed and accuracy requirements. Specifically, YOLOv8n achieves impressive speed on CPU ONNX at 80.4ms and 1.47ms on T4 TensorRT10, while maintaining a mAP<sup>val</sup><sub>50-95</sub> of 37.3 on the COCO dataset. Larger models like YOLOv8x reach a mAP<sup>val</sup><sub>50-95</sub> of 53.9.

**Use Cases:** Thanks to its speed and accuracy, YOLOv8 is ideal for real-time object detection applications, including [security alarm systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8), [smart parking management systems](https://www.ultralytics.com/blog/ultralytics-yolov8-for-smarter-parking-management-systems), and [enhancing recycling efficiency](https://www.ultralytics.com/blog/recycling-efficiency-the-power-of-vision-ai-in-automated-sorting). Its versatility also makes it suitable for various industries like [healthcare](https://www.ultralytics.com/solutions/ai-in-healthcare), [agriculture](https://www.ultralytics.com/solutions/ai-in-agriculture), and [manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing).

**Strengths:**

- **State-of-the-art performance:** Excellent balance of speed and accuracy.
- **Versatility:** Adaptable to various object detection tasks and model sizes.
- **Ease of Use:** Simple to use with pre-trained models and comprehensive documentation.
- **Real-time capabilities:** Optimized for fast inference.

**Weaknesses:**

- Larger models can be computationally intensive, requiring powerful hardware for optimal speed.

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## YOLOv7: High Accuracy and Efficiency

[Ultralytics YOLOv7](https://docs.ultralytics.com/models/yolov7/) is known for its high accuracy and efficiency, making it a strong contender for complex object detection tasks.

**Architecture:** YOLOv7 introduces techniques like "extend" and "compound scaling" in its architecture to improve efficiency without sacrificing accuracy. It focuses on optimizing the model's architecture for faster training and inference while maintaining high detection performance.

**Performance:** YOLOv7 demonstrates excellent performance, particularly in larger model configurations like YOLOv7x, which achieves a mAP<sup>val</sup><sub>50-95</sub> of 53.1 on the COCO dataset. While specific speed metrics on CPU ONNX are not listed in the provided table, it is known for its efficient architecture designed for fast inference, although typically not as fast as the smaller YOLOv8 models.

**Use Cases:** YOLOv7 is well-suited for applications requiring high accuracy, such as [wildlife conservation](https://www.ultralytics.com/blog/protecting-biodiversity-the-kashmir-world-foundations-success-story-with-yolov5-and-yolov8) and detailed [medical image analysis](https://www.ultralytics.com/blog/ultralytics-yolo11-in-hospitals-advancing-healthcare-with-computer-vision). Its efficiency also makes it applicable in scenarios where computational resources are somewhat constrained but high accuracy is paramount.

**Strengths:**

- **High Accuracy:** Particularly strong in larger model sizes.
- **Efficient Architecture:** Designed for faster training and inference.
- **Suitable for complex tasks:** Excels in scenarios demanding precise object detection.

**Weaknesses:**

- Inference speed might be slower compared to the smaller YOLOv8 models, especially on resource-constrained devices.

[Learn more about YOLOv7](https://docs.ultralytics.com/models/yolov7/){ .md-button }

## Model Comparison Table

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv7l | 640                   | 51.4                 | -                              | 6.84                                | 36.9               | 104.7             |
| YOLOv7x | 640                   | 53.1                 | -                              | 11.57                               | 71.3               | 189.9             |
|         |                       |                      |                                |                                     |                    |                   |
| YOLOv8n | 640                   | 37.3                 | 80.4                           | 1.47                                | 3.2                | 8.7               |
| YOLOv8s | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |

## Other YOLO Models

Besides YOLOv7 and YOLOv8, Ultralytics offers a range of other YOLO models to suit different needs. Users might also be interested in exploring:

- [YOLOv5](https://docs.ultralytics.com/models/yolov5/): A widely-used and versatile model known for its balance of performance and speed.
- [YOLOv6](https://docs.ultralytics.com/models/yolov6/): Developed by Meituan, focusing on industrial applications with a strong emphasis on speed and efficiency.
- [YOLOv9](https://docs.ultralytics.com/models/yolov9/): The successor to YOLOv8, aiming to further improve real-time object detection capabilities.
- [YOLOv10](https://docs.ultralytics.com/models/yolov10/): The latest model focusing on efficiency and speed, eliminating NMS for faster inference.
- [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/): Models from Deci AI, designed for high performance and efficiency, with quantization support.

Choosing between YOLOv7 and YOLOv8 depends on the specific application requirements. If real-time speed is the top priority, especially on less powerful hardware, smaller YOLOv8 models are excellent choices. For applications demanding the highest possible accuracy and where computational resources are less constrained, YOLOv7 and larger YOLOv8 models are highly effective options. Ultralytics continuously innovates, providing a rich ecosystem of YOLO models to address diverse computer vision needs.
