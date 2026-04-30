---
comments: true
description: Compare YOLOv10 and YOLOX for object detection. Explore architecture, benchmarks, and use cases to choose the best real-time detection model for your needs.
keywords: YOLOv10, YOLOX, object detection, Ultralytics, real-time, model comparison, benchmark, computer vision, deep learning, AI
---

# YOLOX vs YOLOv10: Comparing Anchor-Free and NMS-Free Real-Time Object Detection

The evolution of real-time computer vision models has been marked by significant architectural leaps. Two pivotal milestones in this journey are YOLOX and YOLOv10. Released in 2021, YOLOX successfully bridged the gap between academic research and industrial application by introducing a highly effective anchor-free design. Three years later, YOLOv10 revolutionized the field by eliminating the need for Non-Maximum Suppression (NMS) during post-processing, pushing the boundaries of efficiency and speed.

This comprehensive technical comparison explores the architectures, performance metrics, and ideal use cases for both models, providing insights to help you choose the right tool for your next object detection project.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='&#91;"YOLOX", "YOLOv10"&#93;'></canvas>

## Model Origins and Metadata

Understanding the origins of these models provides context for their architectural choices and intended deployment environments.

**YOLOX Details**  
Authors: Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun  
Organization: [Megvii](https://en.megvii.com/)  
Date: 2021-07-18  
Arxiv: [https://arxiv.org/abs/2107.08430](https://arxiv.org/abs/2107.08430)  
GitHub: [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)  
Docs: [https://yolox.readthedocs.io/en/latest/](https://yolox.readthedocs.io/en/latest/)

[Learn more about YOLOX](https://github.com/Megvii-BaseDetection/YOLOX){ .md-button }

**YOLOv10 Details**  
Authors: Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, and Guiguang Ding  
Organization: [Tsinghua University](https://www.tsinghua.edu.cn/en/)  
Date: 2024-05-23  
Arxiv: [https://arxiv.org/abs/2405.14458](https://arxiv.org/abs/2405.14458)  
GitHub: [https://github.com/THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)  
Docs: [https://docs.ultralytics.com/models/yolov10/](https://docs.ultralytics.com/models/yolov10/)

[Learn more about YOLOv10](https://docs.ultralytics.com/models/yolov10/){ .md-button }

## Architectural Innovations

The core differences between YOLOX and YOLOv10 lie in how they handle bounding box predictions and post-processing.

### YOLOX: Pioneering Anchor-Free Design

YOLOX made waves by transitioning the YOLO family to an anchor-free architecture. By predicting the center of an object rather than relying on predefined anchor boxes, YOLOX drastically reduced the number of design parameters and heuristic tuning required for custom datasets. Furthermore, it introduced a decoupled head, separating classification and regression tasks into distinct pathways. This approach resolved the conflict between identifying _what_ an object is and determining _where_ it is, leading to a noticeable bump in convergence speed and precision.

### YOLOv10: The NMS-Free Revolution

While YOLOX simplified the detection head, it still relied on NMS to filter out redundant bounding box predictions. YOLOv10 tackled this fundamental bottleneck. By utilizing consistent dual assignments during training, YOLOv10 achieves native end-to-end detection. It employs a one-to-many head during training to ensure rich supervisory signals, while utilizing a one-to-one head during inference to output final predictions directly. This holistic efficiency-accuracy driven design eliminates NMS entirely, significantly reducing inference latency on embedded chips.

!!! note "The Impact of Removing NMS"

    Non-Maximum Suppression is often a complex operation to accelerate on Neural Processing Units (NPUs). By removing it, YOLOv10 allows the entire model graph to execute seamlessly on specialized hardware, drastically improving compatibility with optimization frameworks like [OpenVINO](https://docs.ultralytics.com/integrations/openvino/) and TensorRT.

## Performance Metrics and Comparison

When evaluating models for production, balancing accuracy with computational overhead is critical. The table below illustrates the trade-offs between various scales of YOLOX and YOLOv10.

| Model     | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| --------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| YOLOXnano | 416                         | 25.8                       | -                                    | -                                         | **0.91**                 | **1.08**                |
| YOLOXtiny | 416                         | 32.8                       | -                                    | -                                         | 5.06                     | 6.45                    |
| YOLOXs    | 640                         | 40.5                       | -                                    | 2.56                                      | 9.0                      | 26.8                    |
| YOLOXm    | 640                         | 46.9                       | -                                    | 5.43                                      | 25.3                     | 73.8                    |
| YOLOXl    | 640                         | 49.7                       | -                                    | 9.04                                      | 54.2                     | 155.6                   |
| YOLOXx    | 640                         | 51.1                       | -                                    | 16.1                                      | 99.1                     | 281.9                   |
|           |                             |                            |                                      |                                           |                          |                         |
| YOLOv10n  | 640                         | 39.5                       | -                                    | **1.56**                                  | 2.3                      | 6.7                     |
| YOLOv10s  | 640                         | 46.7                       | -                                    | 2.66                                      | 7.2                      | 21.6                    |
| YOLOv10m  | 640                         | 51.3                       | -                                    | 5.48                                      | 15.4                     | 59.1                    |
| YOLOv10b  | 640                         | 52.7                       | -                                    | 6.54                                      | 24.4                     | 92.0                    |
| YOLOv10l  | 640                         | 53.3                       | -                                    | 8.33                                      | 29.5                     | 120.3                   |
| YOLOv10x  | 640                         | **54.4**                   | -                                    | 12.2                                      | 56.9                     | 160.4                   |

### Analyzing the Data

The metrics clearly demonstrate YOLOv10's generational leap. For instance, YOLOv10-S achieves a [mean Average Precision](https://docs.ultralytics.com/guides/yolo-performance-metrics/) of 46.7% compared to YOLOX-m's 46.9%, but does so using less than a third of the parameters (7.2M vs 25.3M) and significantly fewer FLOPs. Furthermore, the top-tier YOLOv10-X model pushes the mAP to 54.4%, making it highly competitive for demanding accuracy tasks while remaining faster than the older YOLOX-x architecture.

## The Ultralytics Ecosystem Advantage

While YOLOX remains a robust open-source research implementation, adopting YOLOv10 provides immediate access to the well-maintained ecosystem provided by Ultralytics. Choosing an Ultralytics-supported model ensures a streamlined user experience characterized by a simple API and extensive documentation.

Developers benefit heavily from the framework's memory requirements; training Ultralytics models typically consumes far less CUDA memory than heavy transformer-based alternatives like [RT-DETR](https://docs.ultralytics.com/models/rtdetr/). This efficient training footprint allows for larger batch sizes on consumer-grade hardware, accelerating the time from data collection to model deployment. Furthermore, the framework offers unmatched versatility, allowing users to switch seamlessly between [object detection](https://docs.ultralytics.com/tasks/detect/), [instance segmentation](https://docs.ultralytics.com/tasks/segment/), and [pose estimation](https://docs.ultralytics.com/tasks/pose/) with minimal code changes.

### Training and Inference Example

The unified API makes validating ideas incredibly fast. The following snippet demonstrates how easily you can train and deploy a YOLOv10 model using [PyTorch](https://pytorch.org/) backend:

```python
from ultralytics import YOLO

# Load a pre-trained YOLOv10 nano model
model = YOLO("yolov10n.pt")

# Train the model on the COCO8 dataset
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference on a sample image
predictions = model.predict("https://ultralytics.com/images/bus.jpg")

# Export the model for edge deployment
model.export(format="engine", half=True)
```

By leveraging built-in export routines, converting models to formats like [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/) or [ONNX](https://docs.ultralytics.com/integrations/onnx/) requires just a single line of code, entirely bypassing complex compilation hurdles.

## Ideal Use Cases and Deployment Scenarios

Choosing between these architectures depends largely on your hardware constraints and specific domain requirements.

### Real-Time Video Analytics

For applications requiring ultra-low latency, such as autonomous driving or real-time traffic monitoring, YOLOv10 is the superior choice. Its end-to-end NMS-free design ensures deterministic execution times, which is critical for safety systems where variable post-processing latency cannot be tolerated. The models easily achieve high frame rates on devices like the NVIDIA Jetson series.

### Academic Baselines and Edge Microcontrollers

YOLOX still holds value in academic settings where researchers want a clean, decoupled-head baseline for experimenting with label assignment strategies. Additionally, the exceptionally small YOLOX-Nano (under 1 million parameters) can be squeezed onto highly constrained edge microcontrollers where memory is measured in kilobytes, provided the hardware can support standard convolution operations.

## The Ultimate Standard: Ultralytics YOLO26

While YOLOv10 marked a massive leap by removing NMS, the field of computer vision advances rapidly. For developers aiming to implement the absolute best-in-class performance today, we highly recommend exploring [YOLO26](https://platform.ultralytics.com/ultralytics/yolo26).

Released as the latest standard in vision AI, YOLO26 takes the foundational ideas of its predecessors and supercharges them. It offers the ultimate performance balance, natively supporting detection, segmentation, pose, and oriented bounding boxes.

Here is why YOLO26 is the recommended choice for modern computer vision pipelines:

- **End-to-End NMS-Free Design:** Building on the breakthroughs of YOLOv10, YOLO26 is natively end-to-end, guaranteeing faster, deterministic inference times without post-processing bottlenecks.
- **Up to 43% Faster CPU Inference:** It is specifically optimized for edge computing, ensuring exceptional performance on mobile processors and devices lacking discrete GPUs.
- **MuSGD Optimizer:** Inspired by Large Language Model training (specifically Moonshot AI's Kimi K2), YOLO26 utilizes a hybrid of SGD and Muon for incredibly stable training and rapid convergence.
- **ProgLoss + STAL:** These advanced loss functions deliver notable improvements in small-object recognition, which is critical for demanding domains like aerial imagery and drone navigation.
- **DFL Removal:** By removing Distribution Focal Loss, YOLO26 simplifies the model graph for frictionless export to edge and low-power devices.
- **Task-Specific Improvements:** Whether you are using Residual Log-Likelihood Estimation (RLE) for pose estimation or specialized angle loss for OBB, YOLO26 is fine-tuned for every major vision task.

For developers ready to upgrade their pipelines with the most efficient training and deployment tools available, transitioning to the [Ultralytics Platform](https://platform.ultralytics.com) and leveraging YOLO26 guarantees you stay at the cutting edge of artificial intelligence. Users interested in older but stable architectures may also review [YOLO11](https://platform.ultralytics.com/ultralytics/yolo11) or [YOLOv8](https://platform.ultralytics.com/ultralytics/yolov8) for extensive community support and proven robustness.
