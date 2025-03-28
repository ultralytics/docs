---
comments: true
description: Compare YOLOv9 and PP-YOLOE+ models in architecture, performance, and use cases. Find the best object detection model for your needs.
keywords: YOLOv9,PP-YOLOE+,object detection,model comparison,computer vision,AI,deep learning,YOLO,PP-YOLOE,performance comparison
---

# YOLOv9 vs PP-YOLOE+: Detailed Technical Comparison

Selecting the right object detection model is crucial for computer vision tasks. This page provides a technical comparison between **YOLOv9** and **PP-YOLOE+**, examining their architectures, performance, and applications to guide your choice.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv9", "PP-YOLOE+"]'></canvas>

## YOLOv9: Programmable Gradient Information

**YOLOv9**, introduced in 2024, represents a significant advancement in the YOLO series, focusing on information preservation through novel architectural designs.

- **Architecture**: YOLOv9 is authored by Chien-Yao Wang and Hong-Yuan Mark Liao from the Institute of Information Science, Academia Sinica, Taiwan. It introduces Programmable Gradient Information (PGI) and Generalized Efficient Layer Aggregation Network (GELAN). PGI addresses information loss during deep network propagation, while GELAN optimizes network efficiency. This innovative combination aims to improve accuracy without significantly increasing computational cost. The original paper is available on [arXiv](https://arxiv.org/abs/2402.13616). The official [GitHub repository](https://github.com/WongKinYiu/yolov9) provides implementation details.
- **Performance**: YOLOv9 achieves state-of-the-art performance with a balance of speed and accuracy. As indicated in the comparison chart and table, YOLOv9 models demonstrate high mAP values while maintaining competitive inference speeds. For instance, YOLOv9c achieves 53.0% mAP<sup>val</sup>50-95.
- **Use Cases**: YOLOv9's enhanced efficiency and accuracy make it suitable for a wide range of applications, including [robotics](https://www.ultralytics.com/glossary/robotics), [autonomous driving](https://www.ultralytics.com/solutions/ai-in-automotive), and [security systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8) where high detection performance is critical with limited computational resources.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

## PP-YOLOE+: Enhanced Anchor-Free Detection

**PP-YOLOE+**, developed by PaddlePaddle and detailed in their [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/) framework, is an evolution of the PP-YOLOE series, known for its anchor-free approach and efficiency.

- **Architecture**: PP-YOLOE+ builds upon the anchor-free detection paradigm, simplifying the model and reducing the need for anchor-related hyperparameters. It typically includes improvements over the base PP-YOLOE in backbone, neck, and detection head design, often incorporating techniques like decoupled heads and VariFocal Loss to refine detection accuracy. The documentation and implementation are available on [PaddleDetection GitHub](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.8.1/configs/ppyoloe/README.md).
- **Performance**: PP-YOLOE+ models are designed to offer a strong balance between accuracy and inference speed. As shown in the comparison table, PP-YOLOE+ models like PP-YOLOE+m and PP-YOLOE+l provide competitive mAP scores and efficient inference times, making them versatile for various applications.
- **Use Cases**: PP-YOLOE+'s anchor-free design and balanced performance characteristics make it well-suited for applications like [industrial quality inspection](https://www.ultralytics.com/solutions/ai-in-manufacturing), [smart retail](https://www.ultralytics.com/blog/achieving-retail-efficiency-with-ai), and [environmental monitoring](https://www.ultralytics.com/blog/greener-future-through-vision-ai-and-ultralytics-yolo) where robust and efficient object detection is needed.

[PP-YOLOE+ Documentation (PaddleDetection)](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ppyoloe)

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv9t    | 640                   | 38.3                 | -                              | 2.3                                 | 2.0                | 7.7               |
| YOLOv9s    | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m    | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c    | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e    | 640                   | 55.6                 | -                              | 16.77                               | 57.3               | 189.0             |
|            |                       |                      |                                |                                     |                    |                   |
| PP-YOLOE+t | 640                   | 39.9                 | -                              | 2.84                                | 4.85               | 19.15             |
| PP-YOLOE+s | 640                   | 43.7                 | -                              | 2.62                                | 7.93               | 17.36             |
| PP-YOLOE+m | 640                   | 49.8                 | -                              | 5.56                                | 23.43              | 49.91             |
| PP-YOLOE+l | 640                   | 52.9                 | -                              | 8.36                                | 52.2               | 110.07            |
| PP-YOLOE+x | 640                   | 54.7                 | -                              | 14.3                                | 98.42              | 206.59            |

For users interested in other high-performance object detection models, Ultralytics also offers YOLOv5, YOLOv7, YOLOv8 and the cutting-edge YOLO11, each with unique strengths and optimizations. Explore our [model documentation](https://docs.ultralytics.com/models/) for further comparisons and details.
