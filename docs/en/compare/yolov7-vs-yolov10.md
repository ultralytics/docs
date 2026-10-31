---
title: YOLOv7 vs YOLOv10 Comparison
comments: true
description: Compare YOLOv7 and YOLOv10 for real-time object detection, covering E-ELAN vs NMS-free dual assignments, COCO mAP, TensorRT speed, model size, and use cases.
keywords: YOLOv7, YOLOv10, object detection, model comparison, performance benchmarks, computer vision, Ultralytics YOLO, edge deployment, real-time AI
---

# YOLOv7 vs YOLOv10

The field of computer vision has advanced quickly over the past few years, with the YOLO (You Only Look Once) family of models leading progress in real-time object detection. Choosing the right architecture for your computer vision projects requires a clear understanding of the available options. This technical comparison covers the key differences between two landmark architectures: **YOLOv7** and **YOLOv10**.

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv7", "YOLOv10"]'></canvas>

## Introduction to the Models

Both of these models represent significant milestones in object detection, yet they take fundamentally different approaches to solving the challenges of object detection.

### YOLOv7: The Bag-of-Freebies Pioneer

Released on July 6, 2022, by researchers Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao from the [Institute of Information Science, Academia Sinica](https://www.iis.sinica.edu.tw/en/index.html), YOLOv7 introduced new ways to optimize real-time detectors. The original research, detailed in their [academic paper](https://arxiv.org/abs/2207.02696) and hosted on their [official GitHub repository](https://github.com/WongKinYiu/yolov7), focused heavily on architectural re-parameterization and a trainable "bag-of-freebies."

YOLOv7 leverages an extended efficient layer aggregation network (E-ELAN) to guide the network in learning diverse features without destroying the original gradient path. This makes it a robust choice for academic research benchmarks and systems heavily reliant on standard high-end GPUs.

[Learn more about YOLOv7](https://docs.ultralytics.com/models/yolov7){ .md-button }

### YOLOv10: Real-Time End-to-End Detection

Developed by Ao Wang and colleagues at [Tsinghua University](https://www.tsinghua.edu.cn/en/), YOLOv10 was released on May 23, 2024. As detailed in its [arXiv publication](https://arxiv.org/abs/2405.14458) and the [Tsinghua GitHub repository](https://github.com/THU-MIG/yolov10), this model eliminates a long-standing bottleneck in object detection: [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms).

YOLOv10 introduced consistent dual assignments for NMS-free training, changing the post-processing pipeline. Its holistic efficiency-accuracy driven model design reduces computational redundancy, resulting in an architecture well suited to edge devices that need low latency.

[Learn more about YOLOv10](https://docs.ultralytics.com/models/yolov10){ .md-button }

!!! tip "NMS-Free Architecture"

    The removal of Non-Maximum Suppression (NMS) in YOLOv10 allows the entire model to be exported as a single computational graph. This simplifies deployment using runtimes like [TensorRT](https://developer.nvidia.com/tensorrt) or [OpenVINO](https://docs.ultralytics.com/integrations/openvino).

## Performance and Metrics Comparison

When analyzing model performance, it is important to evaluate the trade-offs between accuracy, speed, and computational weight. The following table showcases how different sizes of these models stack up against each other.

| Model    | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| -------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| YOLOv7l  | 640                         | 51.4                       | -                                    | 6.84                                      | 36.9                     | 104.7                   |
| YOLOv7x  | 640                         | 53.1                       | -                                    | 11.57                                     | 71.3                     | 189.9                   |
|          |                             |                            |                                      |                                           |                          |                         |
| YOLOv10n | 640                         | 38.5                       | -                                    | **1.84**                                  | **2.3**                  | **6.7**                 |
| YOLOv10s | 640                         | 46.3                       | -                                    | 2.49                                      | 7.2                      | 21.6                    |
| YOLOv10m | 640                         | 51.1                       | -                                    | 4.74                                      | 15.4                     | 59.1                    |
| YOLOv10b | 640                         | 52.5                       | -                                    | 5.74                                      | 19.1                     | 92.0                    |
| YOLOv10l | 640                         | 53.2                       | -                                    | 7.28                                      | 24.4                     | 120.3                   |
| YOLOv10x | 640                         | **54.4**                   | -                                    | 10.70                                     | 29.5                     | 160.4                   |

### Analyzing the Trade-Offs

The metrics above reveal a clear generational gap. While YOLOv7x delivers a strong mAP<sup>val</sup> of 53.1%, it requires 71.3M parameters and 189.9B FLOPs. In contrast, YOLOv10l exceeds that accuracy (53.2% mAP) while requiring roughly one-third of the parameters (24.4M) and significantly fewer FLOPs (120.3B). Furthermore, YOLOv10n runs at 1.84 ms on a T4 GPU with TensorRT, making it well suited to real-time video analytics and mobile applications.

## Real-World Use Cases

The architectural differences between these models dictate their optimal use cases.

### When to Use YOLOv7

YOLOv7's rich feature representation suits complex scenes on GPU hardware. Use cases such as [monitoring traffic flow](https://www.ultralytics.com/blog/ai-in-traffic-management-from-congestion-to-coordination) in dense urban areas, analyzing satellite imagery, or identifying defects in heavy [manufacturing automation](https://www.ultralytics.com/blog/manufacturing-automation) can benefit from its re-parameterized architecture. It also remains common in legacy environments built around specific PyTorch 1.x pipelines.

### When to Use YOLOv10

The NMS-free, lightweight design of YOLOv10 suits constrained environments, including [edge computing devices](https://www.ultralytics.com/blog/edge-ai-and-edge-computing-powering-real-time-intelligence) such as NVIDIA Jetson modules or a Raspberry Pi. Its low-latency performance makes it a good fit for fast-moving applications like [sports analytics](https://www.ultralytics.com/blog/exploring-the-applications-of-computer-vision-in-sports), autonomous drone navigation, and high-speed robotic sorting on conveyor belts.

## The Ultralytics Ecosystem Advantage

While both models have strong academic roots, YOLOv10 is easiest to train and deploy within the Ultralytics ecosystem and the [Ultralytics Platform](https://platform.ultralytics.com). Developing computer vision models from scratch is difficult, and the Ultralytics ecosystem simplifies the workflow for machine learning engineers.

- **Ease of Use:** The Ultralytics Python API provides a unified interface. You can train, validate, and export models with just a few lines of code, avoiding the dependency conflicts common in academic repositories.
- **Well-Maintained Ecosystem:** The Ultralytics codebase is actively developed, and users benefit from integrations with popular ML tools like [Weights & Biases](https://docs.ultralytics.com/integrations/weights-biases) for logging, or [Gradio](https://docs.ultralytics.com/integrations/gradio) for fast web demos.
- **Memory Requirements:** Transformer-based object detectors often consume large amounts of CUDA memory during training. In contrast, Ultralytics YOLO models require far less memory, allowing for much larger [batch sizes](https://www.ultralytics.com/glossary/batch-size) on consumer-grade hardware.
- **Versatility:** The Ultralytics pipeline is not restricted to standard bounding boxes. It supports [instance segmentation](https://docs.ultralytics.com/tasks/segment), image classification, [pose estimation](https://docs.ultralytics.com/tasks/pose), and oriented bounding boxes across supported model families like [YOLO11](https://platform.ultralytics.com/ultralytics/yolo11) and [YOLOv8](https://platform.ultralytics.com/ultralytics/yolov8).

### Streamlined Training Example

Running a training pipeline with Ultralytics is straightforward. Whether you are leveraging the NMS-free speed of YOLOv10 or another supported model, the syntax remains consistent (note that the Ultralytics package does not support YOLOv7 training):

```python
from ultralytics import YOLO

# Load the preferred model (e.g., YOLOv10 Nano)
model = YOLO("yolov10n.pt")

# Train the model on the COCO8 dataset
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run an inference prediction on a sample image
predictions = model.predict("https://ultralytics.com/images/bus.jpg")

# Export to an edge-friendly format like ONNX
model.export(format="onnx")
```

## Use Cases and Recommendations

Choosing between YOLOv7 and YOLOv10 depends on your specific project requirements, deployment constraints, and ecosystem preferences.

### When to Choose YOLOv7

YOLOv7 is a strong choice for:

- **Academic Benchmarking:** Reproducing 2022-era state-of-the-art results or studying the effects of E-ELAN and trainable bag-of-freebies techniques.
- **Reparameterization Research:** Investigating planned reparameterized convolutions and compound model scaling strategies.
- **Existing Custom Pipelines:** Projects with heavily customized pipelines built around YOLOv7's specific architecture that cannot easily be refactored.

### When to Choose YOLOv10

YOLOv10 is recommended for:

- **NMS-Free Real-Time Detection:** Applications that benefit from end-to-end detection without Non-Maximum Suppression, reducing deployment complexity.
- **Balanced Speed-Accuracy Tradeoffs:** Projects requiring a strong balance between inference speed and detection accuracy across various model scales.
- **Consistent-Latency Applications:** Deployment scenarios where predictable inference times are critical, such as [robotics](https://www.ultralytics.com/glossary/robotics) or autonomous systems.

### When to Choose Ultralytics (YOLO26)

For most new projects, [Ultralytics YOLO26](https://platform.ultralytics.com/ultralytics/yolo26) offers the best combination of performance and developer experience:

- **NMS-Free Edge Deployment:** Applications requiring consistent, low-latency inference without the complexity of Non-Maximum Suppression post-processing.
- **CPU-Only Environments:** Devices without dedicated GPU acceleration, where YOLO26n runs up to 43% faster on CPU than YOLO11n.
- **Small Object Detection:** Challenging scenarios like [aerial drone imagery](https://docs.ultralytics.com/datasets/detect/visdrone) or IoT sensor analysis where ProgLoss and STAL improve accuracy on tiny objects.

## The Future: Introducing YOLO26

While YOLOv7 and YOLOv10 are impressive milestones, the frontier of AI is always advancing. Released in January 2026, **Ultralytics YOLO26** is the recommended model for efficiency and accuracy across edge and cloud deployments.

If you are starting a new computer vision project today, YOLO26 is the recommended architecture. It builds on its predecessors with several innovations:

- **End-to-End NMS-Free Design:** Taking inspiration from YOLOv10, YOLO26's optional one-to-one head (`nms=False`) skips NMS post-processing, giving lower and more consistent latency for real-time robotics.
- **Up to 43% Faster CPU Inference:** By removing the Distribution Focal Loss (DFL) module, YOLO26 speeds up execution on non-GPU edge hardware, with YOLO26n running up to 43% faster on CPU than YOLO11n, making it a good fit for [IoT devices](https://www.ultralytics.com/blog/industrial-iot-iiot-internet-of-things-explained).
- **MuSGD Optimizer:** Inspired by recent large language model training innovations, YOLO26 incorporates a hybrid of SGD and Muon for more stable training and faster convergence.
- **ProgLoss + STAL:** The progressive loss and small-target-aware label assignment improve small-object recognition, overcoming a historical weakness in older YOLO generations.
- **Multi-Task Versatility:** YOLO26 covers detection, instance segmentation, semantic segmentation, depth estimation, classification, pose estimation, and OBB, with task-specific optimizations such as Residual Log-Likelihood Estimation (RLE) for pose estimation and specialized angle losses for precise OBB detection in aerial imagery.

For engineers seeking the best balance of speed, accuracy, and deployment simplicity, YOLO26 is the natural upgrade from legacy models.
