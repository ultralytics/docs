---
comments: true
description: Compare YOLOv9 and RTDETRv2 object detection models. Explore their architectures, performance, and use cases in this technical analysis.
keywords: YOLOv9, RTDETRv2, object detection, model comparison, computer vision, Ultralytics
---

# YOLOv9 vs RTDETRv2: A Technical Comparison for Object Detection

<script async src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv9", "RTDETRv2"]'></canvas>

In the rapidly evolving field of computer vision, object detection models are crucial for a wide array of applications. Ultralytics consistently pushes the boundaries of real-time object detection, and this page provides a detailed technical comparison between two state-of-the-art models: YOLOv9 and RTDETRv2. We will delve into their architectural nuances, performance benchmarks, and suitability for different use cases.

## Architectural Overview

**YOLOv9** represents the cutting edge of the YOLO series, building upon previous versions to achieve enhanced accuracy and efficiency. It introduces the **Programmable Gradient Information (PGI)** and **Generalized Efficient Layer Aggregation Network (GELAN)**. PGI is designed to preserve complete information during the deep network propagation, mitigating information loss, while GELAN serves as an efficient network architecture that optimizes parameter utilization and computational efficiency. This combination allows YOLOv9 to achieve high accuracy with fewer parameters.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

**RTDETRv2**, on the other hand, takes a different architectural approach, leveraging a **Vision Transformer (ViT) backbone** integrated into a real-time detection transformer framework. This architecture is designed for high efficiency and speed, crucial for real-time applications. RTDETR models are known for their efficient inference and robust performance, particularly in scenarios requiring fast processing. RTDETRv2 enhances the original RT-DETR by incorporating advancements for better feature extraction and faster convergence during training.

[Learn more about RTDETRv2](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

## Performance Metrics

The following table summarizes the performance characteristics of YOLOv9 and RTDETRv2 models at a 640 image size, providing a comparative view based on key metrics:

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv9t    | 640                   | 38.3                 | -                              | 2.3                                 | 2.0                | 7.7               |
| YOLOv9s    | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m    | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c    | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e    | 640                   | 55.6                 | -                              | 16.77                               | 57.3               | 189.0             |
|            |                       |                      |                                |                                     |                    |                   |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | 54.3                 | -                              | 15.03                               | 76                 | 259               |

**Mean Average Precision (mAP):** YOLOv9 variants generally achieve higher mAP scores, indicating superior accuracy in object detection, particularly the larger models like YOLOv9e. RTDETRv2 models, while slightly lower in mAP, still offer competitive accuracy, especially considering their speed advantages. For a deeper understanding of mAP and other metrics, refer to our [YOLO Performance Metrics guide](https://docs.ultralytics.com/guides/yolo-performance-metrics/).

**Inference Speed:** RTDETRv2 models demonstrate faster inference speeds on TensorRT, especially noticeable in the larger models. This makes them particularly suitable for real-time applications where latency is critical. For optimizing YOLO inference, explore our guide on [OpenVINO Latency vs Throughput Modes](https://docs.ultralytics.com/guides/optimizing-openvino-latency-vs-throughput-modes/).

**Model Size and Complexity:** YOLOv9 models tend to have fewer parameters and FLOPs for similar or better mAP, showcasing their architectural efficiency. RTDETRv2 models, while having more parameters in some configurations, maintain a balance, offering a range of sizes to suit different computational budgets.

## Training and Use Cases

**YOLOv9** benefits from PGI and GELAN, which not only improve accuracy but also contribute to more stable and efficient training. It is ideally suited for applications where high detection accuracy is paramount, such as in [medical image analysis](https://www.ultralytics.com/glossary/medical-image-analysis), high-resolution [satellite imagery analysis](https://www.ultralytics.com/blog/using-computer-vision-to-analyse-satellite-imagery), and detailed quality control in [manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing).

**RTDETRv2** excels in scenarios demanding real-time object detection. Its speed and efficiency make it perfect for applications like [autonomous vehicles](https://www.ultralytics.com/solutions/ai-in-self-driving), real-time [security systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8), and robotics. For deploying YOLO models on edge devices, consider exploring our guides for [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/) and [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/).

## Strengths and Weaknesses

**YOLOv9 Strengths:**

- **High Accuracy:** Generally achieves higher mAP, making it suitable for tasks requiring precise object detection.
- **Efficient Architecture:** PGI and GELAN contribute to better parameter utilization and computational efficiency.
- **Robust Training:** Designed for stable and efficient training processes.

**YOLOv9 Weaknesses:**

- **Slower Inference Speed (Compared to RTDETRv2):** May not be the optimal choice for extremely latency-sensitive real-time applications.

**RTDETRv2 Strengths:**

- **Real-time Performance:** Excellent inference speed, particularly on hardware accelerators like TensorRT, ideal for real-time systems.
- **Efficient Inference:** Vision Transformer backbone optimized for speed and efficiency.
- **Versatile Model Sizes:** Offers a range of model sizes to balance accuracy and speed based on application needs.

**RTDETRv2 Weaknesses:**

- **Slightly Lower Accuracy (Compared to YOLOv9):** mAP scores are generally a bit lower than YOLOv9, particularly the larger models.
- **Higher Parameter Count in Some Configurations:** Some variants may have a larger number of parameters compared to YOLOv9 counterparts.

## Conclusion

Choosing between YOLOv9 and RTDETRv2 depends largely on the specific requirements of your application. If accuracy is the top priority and computational resources are less constrained, YOLOv9 is an excellent choice. If real-time performance and speed are critical, especially in resource-limited environments, RTDETRv2 offers a compelling solution.

Both models represent significant advancements in object detection and are part of the broader Ultralytics YOLO ecosystem, which includes other powerful models like [YOLOv8](https://docs.ultralytics.com/models/yolov8/) and [YOLOv10](https://docs.ultralytics.com/models/yolov10/). Explore [Ultralytics HUB](https://docs.ultralytics.com/hub/) to train and deploy these models easily. For further exploration, consider reviewing the [Ultralytics Docs](https://docs.ultralytics.com/guides/) for comprehensive guides and tutorials.
