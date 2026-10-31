---
title: RTDETRv2 vs YOLOv5 Comparison
comments: true
description: Compare RTDETRv2 and Ultralytics YOLOv5 for object detection, covering transformer vs CNN design, COCO mAP, TensorRT latency, model size, and training memory.
keywords: YOLOv5, RTDETRv2, object detection comparison, YOLOv5 vs RTDETRv2, Ultralytics models, model performance, computer vision, object detection, RT-DETR, YOLOv5 features, transformer architecture
---

# RTDETRv2 vs YOLOv5

The evolution of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) has largely been shaped by the need to balance accuracy with real-time inference speed. When comparing RTDETRv2 and Ultralytics YOLOv5, developers are weighing the global context modeling of transformer architectures against the optimized, production-proven efficiency of Convolutional Neural Networks (CNNs).

This guide compares the two architectures, covering their performance metrics, training methodologies, memory requirements, and ideal deployment scenarios to help you choose the best [object detection](https://docs.ultralytics.com/tasks/detect) model for your specific use case.

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["RTDETRv2", "YOLOv5"]'></canvas>

## RTDETRv2: The Transformer Approach to Real-Time Detection

Building upon the original Real-Time Detection Transformer (RT-DETR), RTDETRv2 introduces a series of "bag-of-freebies" to improve upon the baseline architecture without sacrificing its inference latency.

- **Authors:** Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu
- **Organization:** [Baidu](https://www.baidu.com/)
- **Date:** 2024-07-24
- **Links:** [Arxiv Paper](https://arxiv.org/abs/2407.17140), [GitHub Repository](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)

### Architecture and Capabilities

RTDETRv2 uses a hybrid CNN-Transformer architecture. A CNN backbone extracts fine-grained visual features, while the transformer encoder-decoder layers process the entire feature map to capture global context. A hallmark of RTDETRv2 is its end-to-end design, which removes the need for [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) post-processing.

RTDETRv2 achieves strong accuracy, particularly in complex, dense scenes where objects overlap, but it comes with trade-offs. The [attention mechanism](https://www.ultralytics.com/glossary/attention-mechanism) inherent to transformers requires more CUDA memory during training than standard CNNs. And while it performs well on GPUs like the NVIDIA A100 or T4, it is noticeably slower on standard CPUs and resource-constrained edge AI devices.

[Learn more about RTDETRv2](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch){ .md-button }

## Ultralytics YOLOv5: The Industry Standard for Efficiency

When it was released, Ultralytics YOLOv5 made high-performance computer vision accessible to developers worldwide through an intuitive, easy-to-use framework.

- **Author:** Glenn Jocher
- **Organization:** Ultralytics
- **Date:** June 26, 2020
- **Links:** [Official Documentation](https://docs.ultralytics.com/models/yolov5), [GitHub Repository](https://github.com/ultralytics/yolov5)

### Ecosystem and Performance Balance

YOLOv5 is built entirely on the [PyTorch](https://pytorch.org/) framework and uses an efficient CNN architecture. It was designed from the ground up for **ease of use**, with a simple API and extensive documentation.

YOLOv5's main advantages are its versatility and low memory requirements. Training a YOLOv5 model requires much less VRAM than transformer-based models, making it accessible to researchers and engineers with limited hardware budgets. And while RTDETRv2 focuses on bounding box detection, the YOLOv5 repository also supports [instance segmentation](https://docs.ultralytics.com/tasks/segment) (added in YOLOv5 v7.0) and [image classification](https://docs.ultralytics.com/tasks/classify).

!!! tip "Enterprise Model Management"

    You can train, validate, and deploy YOLOv5 directly on [Ultralytics Platform](https://platform.ultralytics.com), which provides cloud training and no-code deployment.

[Learn more about YOLOv5](https://platform.ultralytics.com/ultralytics/yolov5){ .md-button }

## Performance and Metrics Comparison

Performance on the standard [COCO dataset](https://docs.ultralytics.com/datasets/detect/coco) shows clear differences in how these models trade accuracy for resources.

| Model      | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ---------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| RTDETRv2-s | 640                         | 48.1                       | -                                    | 5.03                                      | 20                       | 60                      |
| RTDETRv2-m | 640                         | 51.9                       | -                                    | 7.51                                      | 36                       | 100                     |
| RTDETRv2-l | 640                         | 53.4                       | -                                    | 9.76                                      | 42                       | 136                     |
| RTDETRv2-x | 640                         | **54.3**                   | -                                    | 15.03                                     | 76                       | 259                     |
|            |                             |                            |                                      |                                           |                          |                         |
| YOLOv5n    | 640                         | 28.0                       | -                                    | **1.12**                                  | **1.9**                  | **4.5**                 |
| YOLOv5s    | 640                         | 37.4                       | -                                    | 1.92                                      | 7.2                      | 16.5                    |
| YOLOv5m    | 640                         | 45.4                       | -                                    | 4.03                                      | 21.2                     | 49.0                    |
| YOLOv5l    | 640                         | 49.0                       | -                                    | 6.61                                      | 46.5                     | 109.1                   |
| YOLOv5x    | 640                         | 50.7                       | -                                    | 11.89                                     | 86.7                     | 205.7                   |

### Analyzing the Trade-offs

RTDETRv2-x reaches a peak [mean Average Precision (mAP)](https://www.ultralytics.com/glossary/mean-average-precision-map) of 54.3%, 3.6 points above YOLOv5x (50.7%). YOLOv5x has lower TensorRT latency (11.89 ms vs 15.03 ms) and fewer FLOPs (205.7B vs 259B), but more parameters (86.7M vs 76M). At similar latency, RTDETRv2 is more accurate: RTDETRv2-s reaches 48.1% at 5.03 ms, while YOLOv5m reaches 45.4% at 4.03 ms. For low-power edge deployments, YOLOv5n runs in 1.12 ms with 1.9M parameters, a size tier that RTDETRv2 does not offer.

## Training Efficiency and Code Simplicity

One of the key strengths of the Ultralytics ecosystem is its unified API. If you choose the transformer architecture of RT-DETR for a compute-heavy task, you can still use it entirely within the Ultralytics Python package and switch models by changing a single line of code.

```python
from ultralytics import RTDETR, YOLO

# Load the Ultralytics YOLOv5 small model
model_yolo = YOLO("yolov5su.pt")  # YOLOv5u: anchor-free YOLOv5 weights for the ultralytics package

# Load the RT-DETR large model via Ultralytics
model_rtdetr = RTDETR("rtdetr-l.pt")

# Train YOLOv5 effortlessly on your custom data
model_yolo.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference with both models seamlessly
results_yolo = model_yolo("https://ultralytics.com/images/bus.jpg")
results_rtdetr = model_rtdetr("https://ultralytics.com/images/bus.jpg")

results_yolo[0].show()
```

By leveraging the Ultralytics library, developers automatically gain access to a well-maintained ecosystem featuring experiment tracking integrations (like [Weights & Biases](https://docs.ultralytics.com/integrations/weights-biases) and Comet) and one-line exports to deployment formats like [ONNX](https://onnx.ai/) and [OpenVINO](https://docs.ultralytics.com/integrations/openvino).

## Real-World Applications and Ideal Use Cases

### Where RTDETRv2 Shines

RTDETRv2 is best suited for environments where hardware is not a constraint and accuracy is the main objective.

- **Server-Side Medical Imaging:** Detecting subtle anomalies in high-resolution X-rays.
- **Satellite Imagery:** Tracking dense, overlapping objects in [aerial surveillance](https://www.ultralytics.com/blog/12-aerial-imagery-use-cases-powered-by-computer-vision) tasks on powerful cloud clusters.

### Where YOLOv5 Dominates

YOLOv5 is a practical choice for real-world deployment across diverse hardware.

- **Edge AI Devices:** Deploying [security alarm systems](https://docs.ultralytics.com/guides/security-alarm-system) on Raspberry Pi or NVIDIA Jetson devices where memory is strictly limited.
- **Mobile Applications:** Running fast, real-time bounding box and segmentation inference directly on smartphones via CoreML or LiteRT.
- **High-Speed Industrial Manufacturing:** Inspecting parts on rapid production lines where millisecond latency is critical to operational success.

!!! note "Exploring Other Ultralytics Models"

    YOLOv5 remains widely used, but newer Ultralytics models are available. If you are comparing models for a new project in 2026, consider [Ultralytics YOLO26](https://platform.ultralytics.com/ultralytics/yolo26), the recommended model. YOLO26 offers an **End-to-End NMS-Free Design** through its optional one-to-one head (`nms=False`), similar to transformers but with CNN speed, uses the **MuSGD Optimizer** for stable training, and YOLO26n delivers up to 43% faster CPU inference than YOLO11n. Alternatively, [YOLO11](https://platform.ultralytics.com/ultralytics/yolo11) remains a well-supported choice for versatile deployments requiring [Pose Estimation](https://docs.ultralytics.com/tasks/pose) and [OBB detection](https://docs.ultralytics.com/tasks/obb).

Ultimately, while RTDETRv2 pushes the accuracy ceiling using transformer layers, the Ultralytics YOLO framework offers a strong balance of speed, low memory requirements, and a developer experience that shortens the path from prototype to production.
