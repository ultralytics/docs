---
comments: true
description: Dive into the YOLOX vs YOLOv9 comparison. Explore benchmarks, architecture, and use cases to select the best object detection model for your needs.
keywords: YOLOX, YOLOv9, object detection comparison, AI models, Ultralytics, machine learning, computer vision, deep learning, model benchmarks
---

# YOLOX vs YOLOv9: A Detailed Comparison

Comparing state-of-the-art object detection models is crucial for selecting the right tool for your computer vision tasks. This page provides a technical comparison between YOLOX and YOLOv9, two popular models known for their efficiency and accuracy. We delve into their architectural nuances, performance benchmarks, and ideal applications to help you make an informed decision.

<script async src="https://cdn.jsdelivr.net/npm/chart.js@latest/dist/chart.min.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOX", "YOLOv9"]'></canvas>

## YOLOX: High-Performance Anchor-Free Detection

YOLOX is an anchor-free YOLO series model that stands out for its exceptional balance of speed and accuracy. It simplifies the YOLO pipeline by removing anchors, leading to faster training and inference speeds. YOLOX incorporates advanced techniques such as decoupled heads, SimOTA label assignment, and strong data augmentation. These features contribute to its high performance across various object detection tasks.

**Strengths:**

- **Anchor-Free Design:** Simplifies the model and reduces design parameters.
- **High Accuracy and Speed:** Achieves excellent mAP while maintaining fast inference times.
- **Scalability:** Offers various model sizes (Nano to XXL) to suit different computational resources.
- **Ease of Use:** Seamless integration with Ultralytics YOLO framework.

**Weaknesses:**

- Performance can be sensitive to hyperparameter tuning for specific datasets.
- May require more computational resources compared to extremely lightweight models for real-time applications on edge devices.

**Use Cases:**

- Real-time object detection in various applications such as robotics, surveillance, and autonomous driving.
- Applications requiring a balance between high accuracy and speed.
- Research and development in object detection due to its modular and adaptable design.

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## YOLOv9: Efficiency and Accuracy Through Learnable Gradient Path

YOLOv9 introduces the concept of Programmable Gradient Information (PGI) and Generalized Efficient Layer Aggregation Network (GELAN) to achieve state-of-the-art performance with fewer parameters and computations. PGI helps in learning more reliable gradient information, while GELAN optimizes network architecture for better parameter utilization. This results in a model that is both highly accurate and computationally efficient.

**Strengths:**

- **High Parameter Efficiency:** Achieves top-tier accuracy with a relatively small number of parameters and FLOPs.
- **Superior Accuracy:** Outperforms many other models in terms of mAP, especially in complex scenarios.
- **Fast Inference:** Offers impressive inference speed, making it suitable for real-time applications.
- **Advanced Architecture:** PGI and GELAN innovations contribute to enhanced learning and efficiency.

**Weaknesses:**

- Relatively newer architecture, community support and extensive real-world deployment experience may be still growing compared to more established models.
- Implementation and fine-tuning might require a deeper understanding of its novel components (PGI, GELAN).

**Use Cases:**

- Object detection tasks demanding the highest accuracy with limited computational resources.
- Edge deployment scenarios where model size and speed are critical.
- Applications like high-resolution image analysis and complex scene understanding.
- Research-oriented projects pushing the boundaries of object detection efficiency.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

## Model Comparison Table

| Model     | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| --------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOXnano | 416                   | 25.8                 | -                              | -                                   | 0.91               | 1.08              |
| YOLOXtiny | 416                   | 32.8                 | -                              | -                                   | 5.06               | 6.45              |
| YOLOXs    | 640                   | 40.5                 | -                              | 2.56                                | 9.0                | 26.8              |
| YOLOXm    | 640                   | 46.9                 | -                              | 5.43                                | 25.3               | 73.8              |
| YOLOXl    | 640                   | 49.7                 | -                              | 9.04                                | 54.2               | 155.6             |
| YOLOXx    | 640                   | 51.1                 | -                              | 16.1                                | 99.1               | 281.9             |
|           |                       |                      |                                |                                     |                    |                   |
| YOLOv9t   | 640                   | 38.3                 | -                              | 2.3                                 | 2.0                | 7.7               |
| YOLOv9s   | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m   | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c   | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e   | 640                   | 55.6                 | -                              | 16.77                               | 57.3               | 189.0             |

## Conclusion

Both YOLOX and YOLOv9 are excellent choices for object detection, each with its own strengths. YOLOX provides a robust and versatile anchor-free solution with a strong balance of speed and accuracy, suitable for a wide range of applications. YOLOv9 pushes the boundaries of efficiency and accuracy with its innovative architecture, making it ideal for scenarios demanding top performance with limited resources.

For users seeking other models within the Ultralytics ecosystem, consider exploring [YOLOv8](https://docs.ultralytics.com/models/yolov8/) for a well-rounded and versatile option, [YOLOv7](https://docs.ultralytics.com/models/yolov7/) for high speed and accuracy, and [RT-DETR](https://docs.ultralytics.com/models/rtdetr/) for a Transformer-based real-time detector. The choice ultimately depends on the specific requirements of your project, balancing factors like accuracy, speed, model size, and computational constraints. You can also explore other models like [YOLOv5](https://docs.ultralytics.com/models/yolov5/) and [YOLOv6](https://docs.ultralytics.com/models/yolov6/) for different performance profiles. For further exploration, refer to the [Ultralytics documentation](https://docs.ultralytics.com/models/) and [blog](https://www.ultralytics.com/blog) for in-depth guides and updates on the latest models.
