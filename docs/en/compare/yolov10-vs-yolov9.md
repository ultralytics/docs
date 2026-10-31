---
title: YOLOv10 vs YOLOv9 Comparison
comments: true
description: Compare YOLOv10 and YOLOv9 for object detection, including NMS-free training, PGI and GELAN, COCO mAP, TensorRT speed, parameter efficiency, and use cases.
keywords: YOLOv10, YOLOv9, Ultralytics, object detection, real-time AI, computer vision, model comparison, AI deployment, deep learning
---

# YOLOv10 vs YOLOv9

The evolution of real-time computer vision has brought steady gains in speed, accuracy, and architectural efficiency. When evaluating models for your next deployment, comparing **YOLOv10** and **YOLOv9** shows two distinct approaches to solving deep learning bottlenecks. While YOLOv9 focuses on maximizing gradient information flow during training, YOLOv10 introduced an end-to-end design that removes NMS post-processing.

This guide analyzes their architectural innovations, [performance metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics), and ideal use cases to help developers and researchers choose the right model for their computer vision tasks.

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv10", "YOLOv9"]'></canvas>

## YOLOv10: NMS-Free Real-Time Detection

Developed to address the latency bottlenecks of traditional object detectors, YOLOv10 introduces an end-to-end architecture that removes the need for [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms).

**Technical Details & Lineage:**

- **Authors:** Ao Wang, Hui Chen, Lihao Liu, et al.
- **Organization:** Tsinghua University
- **Date:** May 23, 2024
- **Links:** [arXiv Publication](https://arxiv.org/abs/2405.14458), [GitHub Repository](https://github.com/THU-MIG/yolov10), [Ultralytics Docs](https://docs.ultralytics.com/models/yolov10)

[Learn more about YOLOv10](https://docs.ultralytics.com/models/yolov10){ .md-button }

### Architecture and Strengths

YOLOv10's most significant contribution to the field is its consistent dual-assignment strategy for NMS-free training. By eliminating NMS, the model reduces inference latency, especially on edge devices where post-processing can bottleneck the entire pipeline. It optimizes various components from both efficiency and accuracy perspectives, resulting in a strong [trade-off between speed and parameters](https://en.wikipedia.org/wiki/Pareto_efficiency). For instance, the YOLOv10-S variant runs in 2.49 ms on a T4 GPU with TensorRT, making it highly suitable for high-speed [video analytics](https://docs.ultralytics.com/guides/analytics) and real-time robotic navigation.

### Weaknesses

While the NMS-free design was an important advance for bounding box detection, YOLOv10 is a pure object detector. It lacks the out-of-the-box versatility of newer models that natively support [Instance Segmentation](https://docs.ultralytics.com/tasks/segment) or [Pose Estimation](https://docs.ultralytics.com/tasks/pose).

!!! tip "Exporting YOLOv10"

    When preparing YOLOv10 for production, export the model to optimized formats like [TensorRT](https://docs.ultralytics.com/integrations/tensorrt) or ONNX, using `nms=False` to select its NMS-free one-to-one head. Running raw PyTorch weights in deployment can result in slower-than-expected inference due to unoptimized graph operations.

## YOLOv9: Programmable Gradient Information

Prior to YOLOv10, YOLOv9 introduced novel architectural concepts to solve the information bottleneck problem inherent in deep neural networks, allowing for highly efficient parameter utilization.

**Technical Details & Lineage:**

- **Authors:** Chien-Yao Wang and Hong-Yuan Mark Liao
- **Organization:** Institute of Information Science, Academia Sinica, Taiwan
- **Date:** February 21, 2024
- **Links:** [arXiv Publication](https://arxiv.org/abs/2402.13616), [GitHub Repository](https://github.com/WongKinYiu/yolov9), [Ultralytics Docs](https://docs.ultralytics.com/models/yolov9)

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9){ .md-button }

### Architecture and Strengths

YOLOv9 introduces Programmable Gradient Information (PGI) alongside the Generalized Efficient Layer Aggregation Network (GELAN). PGI ensures that crucial target information is not lost as data passes through the network's deep layers, generating reliable gradients for weight updates. GELAN maximizes the efficiency of the network's parameters. Together, these innovations allow YOLOv9 to achieve high mean Average Precision ([mAP](https://www.ultralytics.com/glossary/mean-average-precision-map)) on the [MS COCO dataset](https://cocodataset.org/), often outperforming heavier models while using fewer FLOPs. It is a strong model for researchers focused on maximizing accuracy metrics.

### Weaknesses

Despite its high accuracy, YOLOv9 still relies on standard NMS post-processing. This means that while the neural network operations are fast, the final bounding box filtering can introduce variable latency depending on the density of objects in the scene. Additionally, its training process can be memory-intensive compared to later models, requiring more powerful [GPU resources](https://developer.nvidia.com/cuda/gpus) for custom dataset fine-tuning.

## Performance Comparison

The table below illustrates the core metrics for both models. Notice how YOLOv10 typically achieves lower latency via TensorRT, while YOLOv9 pushes the upper limits of accuracy in its largest configuration.

| Model    | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| -------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| YOLOv10n | 640                         | 38.5                       | -                                    | **1.84**                                  | 2.3                      | **6.7**                 |
| YOLOv10s | 640                         | 46.3                       | -                                    | 2.49                                      | 7.2                      | 21.6                    |
| YOLOv10m | 640                         | 51.1                       | -                                    | 4.74                                      | 15.4                     | 59.1                    |
| YOLOv10b | 640                         | 52.5                       | -                                    | 5.74                                      | 19.1                     | 92.0                    |
| YOLOv10l | 640                         | 53.2                       | -                                    | 7.28                                      | 24.4                     | 120.3                   |
| YOLOv10x | 640                         | 54.4                       | -                                    | 10.70                                     | 29.5                     | 160.4                   |
|          |                             |                            |                                      |                                           |                          |                         |
| YOLOv9t  | 640                         | 38.3                       | -                                    | 2.3                                       | **2.0**                  | 7.7                     |
| YOLOv9s  | 640                         | 46.8                       | -                                    | 3.54                                      | 7.2                      | 26.7                    |
| YOLOv9m  | 640                         | 51.4                       | -                                    | 6.43                                      | 20.1                     | 76.8                    |
| YOLOv9c  | 640                         | 53.0                       | -                                    | 7.16                                      | 25.5                     | 102.8                   |
| YOLOv9e  | 640                         | **55.6**                   | -                                    | 16.77                                     | 58.1                     | 192.5                   |

## The Next Generation: Why YOLO26 Is Recommended

While YOLOv9 and YOLOv10 are important milestones, the machine learning field moves quickly. For production environments, many developers rely on the integrated, well-maintained ecosystem of [Ultralytics Platform](https://platform.ultralytics.com). As of 2026, the Ultralytics recommendation for both research and enterprise is **YOLO26**.

[YOLO26](https://platform.ultralytics.com/ultralytics/yolo26) builds on the concepts of its predecessors with a streamlined user experience, a simple API, and lower memory requirements during training than transformer-based architectures.

### Key Innovations in YOLO26

- **End-to-End NMS-Free Design:** Building on YOLOv10, YOLO26 supports optional end-to-end inference, skipping NMS post-processing with `nms=False` for simpler deployment and more predictable latency.
- **Up to 43% Faster CPU Inference:** YOLO26n runs up to 43% faster than YOLO11n on CPU, making YOLO26 well suited to [edge AI](https://www.ultralytics.com/glossary/edge-ai) and embedded systems without dedicated GPUs.
- **MuSGD Optimizer:** A hybrid of SGD and Muon (inspired by large language model optimizations) that provides stable training and fast convergence.
- **DFL Removal:** By removing Distribution Focal Loss, YOLO26 simplifies the model export process, improving compatibility with low-power devices and various edge deployment frameworks.
- **Task-Specific Enhancements:** Unlike single-task detectors, YOLO26 supports detection, instance segmentation, semantic segmentation, depth estimation, classification, pose estimation, and OBB. It uses a semantic segmentation loss to improve instance segmentation, Residual Log-Likelihood Estimation (RLE) for more accurate pose estimation, and a specialized angle loss to resolve OBB (Oriented Bounding Box) boundary issues.

!!! note "The Ultralytics Ecosystem Advantage"

    Choosing an Ultralytics model like [YOLO11](https://platform.ultralytics.com/ultralytics/yolo11) or YOLO26 provides ease of use, active development, a large community, and frequent updates that keep your models compatible with inference engines like [OpenVINO](https://docs.ultralytics.com/integrations/openvino) and CoreML.

## Practical Implementation

Training and deploying these models is straightforward with the Ultralytics Python package. The following example shows the Ultralytics training workflow, which automatically handles learning rate scheduling and data augmentation.

```python
from ultralytics import YOLO

# Load the recommended YOLO26 model
model = YOLO("yolo26n.pt")  # Also compatible with 'yolov10n.pt' or 'yolov9c.pt'

# Train the model on the COCO8 example dataset
train_results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=0, batch=16)

# Run inference
predictions = model.predict("https://ultralytics.com/images/bus.jpg")

# Export to ONNX for simplified edge deployment
model.export(format="onnx")
```

## Use Cases and Recommendations

Choosing between YOLOv10 and YOLOv9 depends on your specific project requirements, deployment constraints, and ecosystem preferences.

### When to Choose YOLOv10

YOLOv10 is a strong choice for:

- **NMS-Free Real-Time Detection:** Applications that benefit from end-to-end detection without Non-Maximum Suppression, reducing deployment complexity.
- **Balanced Speed-Accuracy Tradeoffs:** Projects requiring a strong balance between inference speed and detection accuracy across various model scales.
- **Consistent-Latency Applications:** Deployment scenarios where predictable inference times are critical, such as [robotics](https://www.ultralytics.com/glossary/robotics) or autonomous systems.

### When to Choose YOLOv9

YOLOv9 is recommended for:

- **Information Bottleneck Research:** Academic projects studying Programmable Gradient Information (PGI) and Generalized Efficient Layer Aggregation Network (GELAN) architectures.
- **Gradient Flow Optimization Studies:** Research focused on understanding and mitigating information loss in deep network layers during training.
- **High-Accuracy Detection Benchmarking:** Scenarios where YOLOv9's strong COCO benchmark performance is needed as a reference point for architectural comparisons.

### When to Choose Ultralytics (YOLO26)

For most new projects, Ultralytics YOLO26 offers the best combination of performance and developer experience:

- **NMS-Free Edge Deployment:** Applications requiring consistent, low-latency inference without the complexity of Non-Maximum Suppression post-processing.
- **CPU-Only Environments:** Devices without dedicated GPU acceleration, where YOLO26n runs up to 43% faster on CPU than YOLO11n.
- **Small Object Detection:** Challenging scenarios like [aerial drone imagery](https://docs.ultralytics.com/datasets/detect/visdrone) or IoT sensor analysis where ProgLoss and STAL improve accuracy on tiny objects.

## Conclusion

Both YOLOv9 and YOLOv10 offer unique advantages. YOLOv9 maximizes parameter efficiency and gradient flow, and YOLOv9e reaches the highest accuracy in this comparison. YOLOv10 introduced end-to-end bounding box detection in the YOLO family without the latency cost of NMS.

For developers seeking the best balance of performance, versatility, and ease of use, upgrading to newer models is worthwhile. With its MuSGD optimizer, ProgLoss + STAL for improved small-object detection, and multi-task support, YOLO26 is the recommended choice for real-world computer vision projects.
