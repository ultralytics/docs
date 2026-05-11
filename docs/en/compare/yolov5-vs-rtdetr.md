---
comments: true
description: Compare YOLOv5 and RTDETRv2 for object detection. Explore their architectures, performance metrics, strengths, and best use cases in computer vision.
keywords: YOLOv5, RTDETRv2, object detection, model comparison, Ultralytics, computer vision, machine learning, real-time detection, Vision Transformers, AI models
---

# YOLOv5 vs RTDETRv2: Evaluating CNN vs. Transformer Architectures for Object Detection

The landscape of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) has expanded significantly over the past few years, offering developers a wide array of architectures to tackle complex visual tasks. Among the most popular paradigms are Convolutional Neural Networks (CNNs) and Detection Transformers (DETRs).

This guide provides an in-depth technical comparison between two pivotal models in these categories: [Ultralytics YOLOv5](https://platform.ultralytics.com/ultralytics/yolov5), a highly efficient and widely adopted CNN-based model, and RTDETRv2, a state-of-the-art transformer-based real-time object detector.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv5", "RTDETRv2"]'></canvas>

## Ultralytics YOLOv5: The Industry Standard for Efficiency

Since its release, Ultralytics YOLOv5 has become a cornerstone of the AI community, powering thousands of commercial applications and research projects globally. Built entirely on the [PyTorch](https://pytorch.org/) framework, it prioritized an intuitive developer experience without compromising on real-time performance.

**Key Characteristics:**

- **Author:** Glenn Jocher
- **Organization:** [Ultralytics](https://www.ultralytics.com/)
- **Date:** 2020-06-26
- **Links:** [GitHub Repository](https://github.com/ultralytics/yolov5)

### Architecture and Strengths

YOLOv5 utilizes a streamlined CNN architecture designed to maximize [feature extraction](https://www.ultralytics.com/glossary/feature-extraction) efficiency while maintaining an extremely low memory footprint. It employs a CSPDarknet backbone and a PANet neck, creating a powerful combination for multi-scale feature fusion.

One of the primary advantages of YOLOv5 is its **Performance Balance**. It strikes an exceptional trade-off between speed and accuracy, making it an ideal choice for [model deployment](https://docs.ultralytics.com/guides/model-deployment-options) on resource-constrained hardware like [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson) devices and smartphones.

Furthermore, YOLOv5 boasts unparalleled **Versatility**. Unlike models strictly confined to bounding box predictions, YOLOv5 natively supports [image classification](https://docs.ultralytics.com/tasks/classify) and [instance segmentation](https://docs.ultralytics.com/tasks/segment), providing a unified framework for varied visual tasks. Its [training efficiency](https://docs.ultralytics.com/guides/model-training-tips) is also remarkable, requiring significantly less CUDA memory during training compared to transformer-based architectures.

### Weaknesses

Because it relies on an older CNN framework, YOLOv5 inherently depends on [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) during post-processing to eliminate duplicate bounding boxes. While highly optimized within the Ultralytics framework, NMS can occasionally introduce latency bottlenecks on specialized edge NPUs.

[Learn more about YOLOv5](https://platform.ultralytics.com/ultralytics/yolov5){ .md-button }

## RTDETRv2: Real-Time Transformers by Baidu

RTDETRv2 (Real-Time Detection Transformer v2) represents a substantial leap in applying transformer architectures to real-time object detection, addressing the computational inefficiencies that historically plagued standard DETRs.

**Key Characteristics:**

- **Authors:** Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu
- **Organization:** [Baidu](https://www.baidu.com/)
- **Date:** 2024-07-24
- **Links:** [Arxiv Paper](https://arxiv.org/abs/2407.17140), [GitHub Repository](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)

### Architecture and Strengths

RTDETRv2 builds upon its predecessor by utilizing a hybrid encoder and a flexible decoder design to process images. The transformer's self-attention mechanism provides the model with a global understanding of the image context, allowing it to perform exceptionally well in complex scenes with severe object occlusion.

A defining feature of RTDETRv2 is its end-to-end, NMS-free design. By predicting object queries directly without requiring [anchor boxes](https://www.ultralytics.com/glossary/anchor-boxes) or NMS post-processing, it simplifies the inference pipeline. This architecture achieves an impressive [mAP (mean Average Precision)](https://www.ultralytics.com/glossary/mean-average-precision-map) on benchmark datasets like [COCO](https://docs.ultralytics.com/datasets/detect/coco).

### Weaknesses

Despite its real-time capabilities, RTDETRv2 has notably higher **memory requirements** compared to YOLO models. The attention mechanisms in transformers scale quadratically with sequence length, which can lead to out-of-memory errors during high-resolution training unless using massive GPU clusters. Additionally, it lacks the out-of-the-box versatility of the Ultralytics ecosystem, primarily focusing only on 2D [object detection](https://docs.ultralytics.com/tasks/detect) without native support for segmentation or pose estimation.

[Learn more about RTDETR](https://docs.ultralytics.com/models/rtdetr){ .md-button }

## Performance Comparison Table

To objectively evaluate these architectures, we have compiled their performance metrics. Values highlighted in **bold** represent the most efficient or highest performing metrics across the tested scales.

| Model      | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ---------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| YOLOv5n    | 640                         | 28.0                       | **73.6**                             | **1.12**                                  | **2.6**                  | **7.7**                 |
| YOLOv5s    | 640                         | 37.4                       | 120.7                                | 1.92                                      | 9.1                      | 24.0                    |
| YOLOv5m    | 640                         | 45.4                       | 233.9                                | 4.03                                      | 25.1                     | 64.2                    |
| YOLOv5l    | 640                         | 49.0                       | 408.4                                | 6.61                                      | 53.2                     | 135.0                   |
| YOLOv5x    | 640                         | 50.7                       | 763.2                                | 11.89                                     | 97.2                     | 246.4                   |
|            |                             |                            |                                      |                                           |                          |                         |
| RTDETRv2-s | 640                         | 48.1                       | -                                    | 5.03                                      | 20                       | 60                      |
| RTDETRv2-m | 640                         | 51.9                       | -                                    | 7.51                                      | 36                       | 100                     |
| RTDETRv2-l | 640                         | 53.4                       | -                                    | 9.76                                      | 42                       | 136                     |
| RTDETRv2-x | 640                         | **54.3**                   | -                                    | 15.03                                     | 76                       | 259                     |

!!! tip "Performance Context"

    While RTDETRv2-x achieves the highest absolute mAP, it requires nearly 30x the parameters of YOLOv5n. For high-speed applications running on limited hardware, Ultralytics models consistently offer the best computational efficiency.

## The Ultralytics Ecosystem Advantage

When moving a model from a research notebook to a production environment, the software surrounding the model is as important as the neural network architecture. The **Well-Maintained Ecosystem** provided by Ultralytics dramatically accelerates the development lifecycle.

### Unmatched Ease of Use

Ultralytics models prioritize an incredibly streamlined user experience. Whether you want to train a custom model, run validation, or export to hardware-specific formats like [TensorRT](https://docs.ultralytics.com/integrations/tensorrt) or [ONNX](https://docs.ultralytics.com/integrations/onnx), the [Ultralytics Python API](https://docs.ultralytics.com/usage/python) makes it achievable in just a few lines of code.

Here is a practical code example demonstrating how simple it is to train and run inference with an Ultralytics model:

```python
from ultralytics import YOLO

# Initialize the model (automatically downloads the weights)
model = YOLO("yolov5s.pt")

# Train the model on the COCO8 dataset
results = model.train(data="coco8.yaml", epochs=50, imgsz=640, device="cpu")

# Perform inference on an online image
inference_results = model.predict("https://ultralytics.com/images/bus.jpg")

# Display the resulting image with bounding boxes
inference_results[0].show()
```

This simple, unified API natively supports [experiment tracking](https://docs.ultralytics.com/guides/hyperparameter-tuning) integrations with tools like [Weights & Biases](https://docs.ultralytics.com/integrations/weights-biases) and [Comet](https://docs.ultralytics.com/integrations/comet), allowing developers to log metrics seamlessly without writing complex boilerplate code.

## Use Cases and Recommendations

Choosing between YOLOv5 and RT-DETR depends on your specific project requirements, deployment constraints, and ecosystem preferences.

### When to Choose YOLOv5

YOLOv5 is a strong choice for:

- **Proven Production Systems:** Existing deployments where YOLOv5's long track record of stability, extensive documentation, and massive community support are valued.
- **Resource-Constrained Training:** Environments with limited GPU resources where YOLOv5's efficient training pipeline and lower memory requirements are advantageous.
- **Extensive Export Format Support:** Projects requiring deployment across many formats including [ONNX](https://docs.ultralytics.com/integrations/onnx), [TensorRT](https://docs.ultralytics.com/integrations/tensorrt), [CoreML](https://docs.ultralytics.com/integrations/coreml), and [TFLite](https://docs.ultralytics.com/integrations/tflite).

### When to Choose RT-DETR

RT-DETR is recommended for:

- **Transformer-Based Detection Research:** Projects exploring attention mechanisms and transformer architectures for end-to-end object detection without NMS.
- **High-Accuracy Scenarios with Flexible Latency:** Applications where detection accuracy is the top priority and slightly higher inference latency is acceptable.
- **Large Object Detection:** Scenes with primarily medium-to-large objects where the global attention mechanism of transformers provides a natural advantage.

### When to Choose Ultralytics (YOLO26)

For most new projects, [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26) offers the best combination of performance and developer experience:

- **NMS-Free Edge Deployment:** Applications requiring consistent, low-latency inference without the complexity of Non-Maximum Suppression post-processing.
- **CPU-Only Environments:** Devices without dedicated GPU acceleration, where YOLO26's up to 43% faster CPU inference provides a decisive advantage.
- **Small Object Detection:** Challenging scenarios like [aerial drone imagery](https://docs.ultralytics.com/datasets/detect/visdrone) or IoT sensor analysis where ProgLoss and STAL significantly boost accuracy on tiny objects.

## Looking Forward: YOLO11 and YOLO26

If you are starting a new vision project today, it is highly recommended to explore the latest generations of Ultralytics models.

While YOLOv5 remains incredibly reliable, [YOLO11](https://platform.ultralytics.com/ultralytics/yolo11) offers improved accuracy and an expanded set of tasks including [Oriented Bounding Box (OBB)](https://docs.ultralytics.com/tasks/obb) detection.

Even more significantly, the cutting-edge [YOLO26](https://platform.ultralytics.com/ultralytics/yolo26) merges the best of both worlds. It implements an **End-to-End NMS-Free Design** (first pioneered in [YOLOv10](https://docs.ultralytics.com/models/yolov10)), eliminating the post-processing overhead while maintaining the efficiency of a CNN. YOLO26 also introduces the **MuSGD Optimizer**, inspired by LLM training innovations, for faster convergence. With **DFL Removal** (Distribution Focal Loss removed for simplified export and better edge/low-power device compatibility), YOLO26 delivers **Up to 43% Faster CPU Inference**, making it the absolute best choice for edge AI. Additionally, **ProgLoss + STAL** provides improved loss functions with notable improvements in small-object recognition, critical for IoT, robotics, and aerial imagery.

## Conclusion

Choosing between YOLOv5 and RTDETRv2 depends heavily on your deployment constraints. RTDETRv2 pushes the boundaries of mAP utilizing powerful transformer attention mechanisms but comes with a steep cost in memory and computational overhead.

Conversely, Ultralytics YOLOv5 offers a proven, highly optimized, and versatile solution that runs smoothly everywhere—from cloud servers to microcontrollers. For teams looking for the highest possible accuracy alongside seamless deployment tools, upgrading within the Ultralytics ecosystem to YOLO26 provides the definitive state-of-the-art solution for modern [vision AI](https://www.ultralytics.com/blog-category/vision-ai) applications.
