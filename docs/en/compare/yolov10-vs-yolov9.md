---
comments: true
description: Compare YOLOv10 and YOLOv9 object detection models. Explore architectures, metrics, and use cases to choose the best model for your application.
keywords: YOLOv10,YOLOv9,Ultralytics,object detection,real-time AI,computer vision,model comparison,AI deployment,deep learning
---

# YOLOv10 vs. YOLOv9: Advancing Real-Time Object Detection

The year 2024 marked a period of rapid innovation in the [object detection](https://docs.ultralytics.com/tasks/detect/) landscape, with the release of two significant architectures: **YOLOv10** and **YOLOv9**. While both models aim to push the boundaries of speed and accuracy, they achieve this through fundamentally different architectural philosophies.

YOLOv10 focuses on eliminating the inference latency caused by post-processing through an NMS-free design, whereas YOLOv9 emphasizes information retention in deep networks using Programmable Gradient Information (PGI).

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv10", "YOLOv9"]'></canvas>

## Performance Comparison

The following table provides a detailed look at how these models compare across standard benchmarks. The data highlights the trade-offs between parameter efficiency, inference speed, and detection accuracy (mAP).

| Model    | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| -------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv10n | 640                   | 39.5                 | -                              | 1.56                                | 2.3                | 6.7               |
| YOLOv10s | 640                   | 46.7                 | -                              | 2.66                                | 7.2                | 21.6              |
| YOLOv10m | 640                   | 51.3                 | -                              | 5.48                                | 15.4               | 59.1              |
| YOLOv10b | 640                   | 52.7                 | -                              | 6.54                                | 24.4               | 92.0              |
| YOLOv10l | 640                   | 53.3                 | -                              | 8.33                                | 29.5               | 120.3             |
| YOLOv10x | 640                   | 54.4                 | -                              | 12.2                                | 56.9               | 160.4             |
|          |                       |                      |                                |                                     |                    |                   |
| YOLOv9t  | 640                   | 38.3                 | -                              | 2.3                                 | 2.0                | 7.7               |
| YOLOv9s  | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m  | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c  | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e  | 640                   | 55.6                 | -                              | 16.77                               | 57.3               | 189.0             |

## YOLOv10: The End-to-End Pioneer

**YOLOv10**, developed by researchers at [Tsinghua University](https://github.com/THU-MIG/yolov10), represents a shift toward end-to-end processing. Released on **May 23, 2024**, by Ao Wang, Hui Chen, and colleagues, it addresses the bottleneck of Non-Maximum Suppression (NMS).

[Learn more about YOLOv10](https://docs.ultralytics.com/models/yolov10/){ .md-button }

### Key Architectural Features

- **NMS-Free Training:** By employing consistent dual assignments, YOLOv10 eliminates the need for NMS during inference. This reduces latency and simplifies deployment pipelines, particularly for [edge computing](https://www.ultralytics.com/glossary/edge-computing) applications.
- **Holistic Efficiency Design:** The architecture optimizes various components to reduce computational overhead (FLOPs) while maintaining high capability.
- **Improved Latency:** As shown in the table, YOLOv10 models generally offer lower inference times compared to their YOLOv9 counterparts for similar accuracy levels.

For technical details, you can consult the [YOLOv10 arXiv paper](https://arxiv.org/abs/2405.14458).

## YOLOv9: Mastering Information Flow

**YOLOv9**, released on **February 21, 2024**, by Chien-Yao Wang and Hong-Yuan Mark Liao from [Academia Sinica](https://github.com/WongKinYiu/yolov9), focuses on the theoretical issue of information loss in deep neural networks.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

### Key Architectural Features

- **GELAN Architecture:** The **Generalized Efficient Layer Aggregation Network** combines the strengths of CSPNet and ELAN to maximize parameter utilization.
- **Programmable Gradient Information (PGI):** This auxiliary supervision mechanism ensures that deep layers retain critical information for accurate detection, making the model highly effective for tasks requiring high precision.
- **High Accuracy:** The YOLOv9e model achieves an impressive mAP<sup>val</sup> of 55.6%, outperforming many contemporaries in pure detection accuracy.

For a deeper dive, read the [YOLOv9 arXiv paper](https://arxiv.org/abs/2402.13616).

## Training and Ease of Use

Both models are fully integrated into the Ultralytics ecosystem, providing a unified and seamless experience for developers. Whether you are using YOLOv10 or YOLOv9, the [Ultralytics Python API](https://docs.ultralytics.com/usage/python/) abstracts away the complexity of training pipelines, [data augmentation](https://docs.ultralytics.com/guides/yolo-data-augmentation/), and logging.

### Code Example

Training a model on a custom dataset or a standard benchmark like COCO8 is straightforward. The framework automatically handles the differences in architecture.

```python
from ultralytics import YOLO

# Load a model (Choose YOLOv10 or YOLOv9)
model = YOLO("yolov10n.pt")  # or "yolov9c.pt"

# Train the model on the COCO8 dataset
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Validate the model
model.val()
```

!!! tip "Memory Efficiency"

    Ultralytics YOLO models are engineered for optimal [GPU memory](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit) usage. Compared to transformer-based architectures or older detection models, they allow for larger [batch sizes](https://www.ultralytics.com/glossary/batch-size) on consumer-grade hardware, making state-of-the-art AI accessible to a wider audience.

## Ideal Use Cases

Choosing between YOLOv10 and YOLOv9 often depends on the specific constraints of your deployment environment.

### When to Choose YOLOv10

- **Low Latency Constraints:** If your application runs on mobile devices or embedded systems where every millisecond counts, the NMS-free design of YOLOv10 offers a significant advantage.
- **Simple Deployment:** Removing post-processing steps simplifies the export to formats like [ONNX](https://docs.ultralytics.com/integrations/onnx/) or [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/), reducing the risk of operator incompatibility.
- **Real-Time Video:** Ideal for [traffic management](https://www.ultralytics.com/blog/ai-in-traffic-management-from-congestion-to-coordination) or high-speed manufacturing lines where throughput is critical.

### When to Choose YOLOv9

- **Maximum Accuracy:** For research applications or scenarios where precision is paramount (e.g., [medical image analysis](https://www.ultralytics.com/glossary/medical-image-analysis)), the PGI-enhanced architecture of YOLOv9e delivers superior results.
- **Small Object Detection:** The rich feature preservation of GELAN makes YOLOv9 particularly robust for detecting small or occluded objects in [aerial imagery](https://docs.ultralytics.com/datasets/detect/visdrone/).
- **Complex Scenes:** In environments with high visual clutter, the programmable gradient information helps the model distinguish relevant features more effectively.

## The Future is Here: YOLO26

While YOLOv9 and YOLOv10 are powerful tools, the field of computer vision evolves rapidly. Ultralytics recently released **YOLO26**, a model that synthesizes the best features of previous generations while introducing groundbreaking optimizations.

[Learn more about YOLO26](https://docs.ultralytics.com/models/yolo26/){ .md-button }

YOLO26 is the recommended choice for new projects, offering a superior balance of speed, accuracy, and versatility.

### Why Upgrade to YOLO26?

- **End-to-End NMS-Free:** Like YOLOv10, YOLO26 is natively end-to-end. It eliminates NMS post-processing, ensuring faster inference and simplified deployment pipelines.
- **MuSGD Optimizer:** Inspired by innovations in Large Language Model (LLM) training (specifically Moonshot AI's Kimi K2), YOLO26 utilizes a hybrid of SGD and the Muon optimizer. This results in significantly more stable training and faster convergence.
- **DFL Removal:** By removing Distribution Focal Loss, YOLO26 streamlines the model architecture, making it friendlier for export and compatible with a wider range of edge/low-power devices.
- **Performance Leap:** Optimizations specifically targeting CPU inference deliver speeds up to **43% faster** than previous generations, making it a powerhouse for [edge AI](https://www.ultralytics.com/glossary/edge-ai).
- **Task Versatility:** Unlike the detection-focused releases of v9 and v10, YOLO26 includes specialized improvements for all tasks:
  - **Segmentation:** New semantic segmentation loss and multi-scale proto.
  - **Pose:** Residual Log-Likelihood Estimation (RLE) for high-accuracy keypoints.
  - **OBB:** Specialized angle loss to handle boundary issues in [Oriented Bounding Box](https://docs.ultralytics.com/tasks/obb/) tasks.

### Streamlined Workflow with Ultralytics Platform

Developers can leverage the [Ultralytics Platform](https://platform.ultralytics.com/) (formerly HUB) to manage the entire lifecycle of their YOLO26 models. From annotating datasets to training on the cloud and deploying to edge devices, the Platform provides a unified interface that accelerates time-to-market.

## Conclusion

Both **YOLOv10** and **YOLOv9** represent significant milestones in the history of object detection. YOLOv10 proved that NMS-free architectures could achieve state-of-the-art performance, while YOLOv9 demonstrated the importance of gradient information flow in deep networks.

However, for developers seeking the most robust, versatile, and future-proof solution, **YOLO26** stands out as the premier choice. By combining an NMS-free design with the revolutionary MuSGD optimizer and extensive task support, YOLO26 offers the [best performance balance](https://docs.ultralytics.com/guides/yolo-performance-metrics/) for modern computer vision applications.

### Related Models

- [YOLO11](https://docs.ultralytics.com/models/yolo11/) - The robust predecessor to YOLO26, known for its stability.
- [YOLOv8](https://docs.ultralytics.com/models/yolov8/) - A versatile classic widely used in industry.
- [RT-DETR](https://docs.ultralytics.com/models/rtdetr/) - A transformer-based real-time detector.
