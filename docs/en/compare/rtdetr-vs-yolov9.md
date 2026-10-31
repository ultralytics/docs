---
title: RTDETRv2 vs YOLOv9 Comparison
comments: true
description: Compare RTDETRv2 and YOLOv9 for real-time object detection, covering attention vs PGI/GELAN design, COCO mAP, TensorRT speed, and training memory.
keywords: RTDETRv2, YOLOv9, object detection, Ultralytics models, transformer vision, YOLO series, real-time object detection, model comparison, Vision Transformers, computer vision
---

# RTDETRv2 vs YOLOv9

Computer vision architectures have split into two main approaches: Convolutional Neural Networks (CNNs) and transformer-based models. Comparing RTDETRv2 and YOLOv9 means weighing global attention against programmable gradient information. Both are strong representatives of their approaches to real-time [object detection](https://www.ultralytics.com/glossary/object-detection).

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["RTDETRv2", "YOLOv9"]'></canvas>

## Introduction to the Models

### RTDETRv2: Real-Time Detection Transformer

Developed by researchers at Baidu, RTDETRv2 builds on the original [RT-DETR](https://docs.ultralytics.com/models/rtdetr) by introducing a "Bag-of-Freebies" to enhance the baseline Real-Time Detection Transformer. It tackles the traditional bottleneck of transformers—inference speed—making them viable for real-time applications.

- **Authors:** Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu
- **Organization:** [Baidu](https://www.baidu.com/)
- **Date:** 2024-07-24
- **Links:** [Arxiv](https://arxiv.org/abs/2407.17140), [GitHub](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)

A defining characteristic of RTDETRv2 is its natively **end-to-end NMS-free design**. By removing [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) from post-processing, the model stabilizes inference latency and simplifies the deployment pipeline. The global attention mechanism helps the model with complex scenes and dense crowds, as it evaluates the entire image context simultaneously.

[Learn more about RTDETRv2](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch){ .md-button }

### YOLOv9: Programmable Gradient Information

YOLOv9, an efficient CNN-based architecture, tackles the information bottleneck problem inherent in deep neural networks. It introduces Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN).

- **Authors:** Chien-Yao Wang and Hong-Yuan Mark Liao
- **Organization:** [Institute of Information Science, Academia Sinica](https://www.iis.sinica.edu.tw/en/index.html)
- **Date:** February 21, 2024
- **Links:** [Arxiv](https://arxiv.org/abs/2402.13616), [GitHub](https://github.com/WongKinYiu/yolov9)

YOLOv9 builds on proven [convolutional neural network](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn) foundations with a focus on parameter efficiency. By retaining key information during the feed-forward process, PGI keeps gradients reliable, resulting in a lightweight yet accurate model. However, unlike RTDETRv2, YOLOv9 still relies on standard NMS post-processing.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9){ .md-button }

## Performance and Resource Efficiency

The table below compares mean Average Precision (mAP) and computational cost on the [MS COCO dataset](https://cocodataset.org/). YOLOv9 uses far fewer parameters at similar accuracy (YOLOv9m reaches 51.4 mAP with 20.1M parameters vs 51.9 mAP and 36M for RTDETRv2-m) and is faster on T4 TensorRT up to the c/l size. YOLOv9e reaches the highest mAP (55.6) but is also the slowest model in the table (16.77 ms).

| Model      | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ---------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| RTDETRv2-s | 640                         | 48.1                       | -                                    | 5.03                                      | 20                       | 60                      |
| RTDETRv2-m | 640                         | 51.9                       | -                                    | 7.51                                      | 36                       | 100                     |
| RTDETRv2-l | 640                         | 53.4                       | -                                    | 9.76                                      | 42                       | 136                     |
| RTDETRv2-x | 640                         | 54.3                       | -                                    | 15.03                                     | 76                       | 259                     |
|            |                             |                            |                                      |                                           |                          |                         |
| YOLOv9t    | 640                         | 38.3                       | -                                    | **2.3**                                   | **2.0**                  | **7.7**                 |
| YOLOv9s    | 640                         | 46.8                       | -                                    | 3.54                                      | 7.2                      | 26.7                    |
| YOLOv9m    | 640                         | 51.4                       | -                                    | 6.43                                      | 20.1                     | 76.8                    |
| YOLOv9c    | 640                         | 53.0                       | -                                    | 7.16                                      | 25.5                     | 102.8                   |
| YOLOv9e    | 640                         | **55.6**                   | -                                    | 16.77                                     | 58.1                     | 192.5                   |

### Memory Requirements and Training Efficiency

Transformers like RTDETRv2 are memory-intensive during training and often require substantial CUDA memory. CNN architectures like YOLOv9 and other Ultralytics YOLO models typically use less memory, allowing developers to train with larger batch sizes on consumer-grade hardware.

!!! tip "Efficient Training"

    To make the most of your hardware, consider [Ultralytics Platform](https://platform.ultralytics.com) for cloud training. It handles environment setup and batch sizing automatically.

## The Ultralytics Advantage: Ecosystem and Ease of Use

Standalone repositories like the official RTDETRv2 or YOLOv9 GitHub pages are useful for research, but production environments need stability, ease of use, and a well-maintained ecosystem. Using these models through the [Ultralytics Python API](https://docs.ultralytics.com/usage/python) gives a consistent developer experience.

### Unified API and Versatility

The Ultralytics framework abstracts away the complexities of data loading, augmentations, and distributed training. Furthermore, while the original RTDETRv2 is strictly focused on detection, the Ultralytics ecosystem lets users move between [Object Detection](https://docs.ultralytics.com/tasks/detect), [Instance Segmentation](https://docs.ultralytics.com/tasks/segment), and [Pose Estimation](https://docs.ultralytics.com/tasks/pose).

```python
from ultralytics import RTDETR, YOLO

# Train a YOLOv9 model on custom data
model_yolo = YOLO("yolov9c.pt")
model_yolo.train(data="coco8.yaml", epochs=50, imgsz=640)

# Easily switch to RT-DETR for complex scene evaluation
model_rtdetr = RTDETR("rtdetr-l.pt")
results = model_rtdetr.predict("https://ultralytics.com/images/bus.jpg")

# Export to production-ready formats like TensorRT
model_yolo.export(format="engine")
```

With thorough documentation, automatic experiment tracking, and export to formats like [ONNX](https://docs.ultralytics.com/integrations/onnx), [TensorRT](https://docs.ultralytics.com/integrations/tensorrt), and OpenVINO, Ultralytics shortens the path from prototype to production.

## Ideal Use Cases

### Where RTDETRv2 Excels

Thanks to its global attention mechanism, RTDETRv2 is well suited to **server-side processing** and environments where global context is important. It performs well in:

- **Medical Imaging:** Identifying subtle anomalies where surrounding context is critical.
- **Aerial Surveillance:** Analyzing high-resolution drone footage where wide spatial context helps interpret the scene.
- **Dense Crowd Analysis:** Detecting individuals in crowded scenes, where NMS can suppress overlapping true detections.

### Where YOLOv9 Excels

YOLOv9 is a strong fit for **resource-constrained edge deployments**. Its computational efficiency suits:

- **Robotics:** Real-time navigation and obstacle avoidance where minimal latency is required.
- **Smart City IoT:** Deploying on edge devices like the [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson) for traffic monitoring.
- **Industrial Inspection:** High-speed assembly line quality control requiring high frames-per-second (FPS).

## The Future: Enter Ultralytics YOLO26

YOLOv9 and RTDETRv2 were both significant steps forward, and the field has continued to evolve. For new deployments, **[Ultralytics YOLO26](https://platform.ultralytics.com/ultralytics/yolo26)**, released in January 2026, combines ideas from both architectural approaches.

YOLO26 brings together strengths of transformers and CNNs:

- **End-to-End NMS-Free Design:** Like RTDETRv2, YOLO26 supports native end-to-end inference, skipping NMS post-processing with its optional one-to-one head (`nms=False`) for faster, simpler, and more predictable deployment pipelines.
- **MuSGD Optimizer:** Inspired by Large Language Model (LLM) training techniques (such as Moonshot AI's Kimi K2), YOLO26 uses a hybrid of SGD and Muon for more stable training and faster convergence.
- **Up to 43% Faster CPU Inference:** YOLO26n runs up to 43% faster on CPU than YOLO11n, which suits edge computing and devices without GPUs, where heavier transformers struggle.
- **DFL Removal:** Removing Distribution Focal Loss simplifies the model graph, which eases export to low-power edge devices and embedded Neural Processing Units (NPUs).
- **ProgLoss + STAL:** Progressive Loss and Small-Target-Aware Label Assignment (STAL) improve small-object recognition, which matters for IoT and aerial datasets.

For teams starting a new computer vision project, we recommend evaluating YOLO26. It offers optional NMS-free inference like a transformer, with the speed and training efficiency of a YOLO architecture.

[Learn more about YOLO26](https://platform.ultralytics.com/ultralytics/yolo26){ .md-button }

## Summary

Choosing between RTDETRv2 and YOLOv9 largely comes down to your deployment hardware and specific accuracy needs. RTDETRv2 provides high accuracy and context awareness for server-backed applications, while YOLOv9 offers strong efficiency for edge devices.

The Ultralytics ecosystem lets developers experiment with both YOLOv9 and the original RT-DETR using the same API. Newer models like [YOLO11](https://platform.ultralytics.com/ultralytics/yolo11) and the end-to-end-capable **YOLO26** add broader task support, fast inference, and low memory consumption.
