---
title: YOLOv5 vs YOLOv8 Comparison
comments: true
description: Compare YOLOv5 and YOLOv8 for speed, accuracy, and versatility. Learn which Ultralytics model is best for your object detection and vision tasks.
keywords: YOLOv5, YOLOv8, Ultralytics, object detection, computer vision, YOLO models, model comparison, AI, machine learning, deep learning
---

# YOLOv5 vs YOLOv8

When building scalable and efficient [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) applications, selecting the right architecture is critical. The Ultralytics ecosystem has steadily improved speed and accuracy, providing developers with robust tools for real-world deployments. This technical comparison examines the differences between **YOLOv5** and **YOLOv8**, exploring their architectures, performance trade-offs, and ideal use cases to help you make an informed decision for your next AI project.

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv5", "YOLOv8"]'></canvas>

Both of these models represent significant milestones in the history of real-time [object detection](https://docs.ultralytics.com/tasks/detect), and both benefit from the low memory requirements and [ease of use](https://docs.ultralytics.com/quickstart) that characterize the Ultralytics ecosystem.

## YOLOv5: The Reliable Industry Standard

Introduced in 2020, YOLOv5 rapidly became a widely adopted standard for fast, accessible, and reliable object detection. By leveraging a native [PyTorch](https://pytorch.org/) implementation, it streamlined the training and deployment lifecycle for engineers globally.

- **Authors:** Glenn Jocher
- **Organization:** [Ultralytics](https://www.ultralytics.com)
- **Date:** 2020-06-26
- **GitHub:** [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- **Docs:** [YOLOv5 Documentation](https://docs.ultralytics.com/models/yolov5)

### Architectural Strengths

YOLOv5 operates on an anchor-based detection paradigm, which relies on predefined [anchor boxes](https://www.ultralytics.com/glossary/anchor-boxes) to predict object boundaries. Its architecture incorporates a Cross-Stage Partial (CSP) network backbone, optimizing gradient flow and reducing computational redundancy. This results in a lightweight memory footprint, making it fast to train even on standard consumer [GPUs](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit).

### Ideal Use Cases

YOLOv5 suits projects where maximum throughput and minimal resource use are priorities. It excels in [edge AI](https://www.ultralytics.com/glossary/edge-ai) environments, such as deploying on [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi) or mobile devices. Its maturity means it has been proven in thousands of commercial deployments, offering stable behavior for traditional object detection workflows.

!!! tip "Legacy Deployment Advantage"

    Due to its widespread adoption, YOLOv5 has stable, well-tested export paths to deployment frameworks like [TensorRT](https://developer.nvidia.com/tensorrt) and [ONNX](https://onnx.ai/), which simplifies integration into older technology stacks.

[Learn more about YOLOv5](https://platform.ultralytics.com/ultralytics/yolov5){ .md-button }

## YOLOv8: The Unified Vision Framework

Released in January 2023, YOLOv8 was a major architectural shift, evolving from a dedicated object detector into a versatile, multi-task vision framework.

- **Authors:** Glenn Jocher, Ayush Chaurasia, and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com)
- **Date:** 2023-01-10
- **GitHub:** [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Docs:** [YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8)

### Architectural Innovations

Unlike its predecessor, YOLOv8 introduces an **anchor-free** detection head. This eliminates the need to manually tune anchor configurations based on dataset distributions, enhancing generalization across datasets, from the popular [COCO dataset](https://cocodataset.org/) to diverse custom data.

The architecture also upgrades the backbone with a **C2f module** (Cross-Stage Partial bottleneck with two convolutions), replacing the older C3 module. This enhancement improves feature representation without heavily taxing memory. Additionally, the implementation of a decoupled head—separating classification and regression tasks—improves convergence during [model training](https://docs.ultralytics.com/modes/train).

### Versatility and Python API

YOLOv8 introduced the modern `ultralytics` Python API, standardizing the workflow across various computer vision tasks. Whether you are performing [instance segmentation](https://docs.ultralytics.com/tasks/segment), [image classification](https://docs.ultralytics.com/tasks/classify), or [pose estimation](https://docs.ultralytics.com/tasks/pose), the unified API requires only minor configuration changes.

```python
from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Train on a custom dataset with built-in memory efficiency
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference and easily parse results
predictions = model("https://ultralytics.com/images/bus.jpg")
predictions[0].show()
```

[Learn more about YOLOv8](https://platform.ultralytics.com/ultralytics/yolov8){ .md-button }

## Detailed Performance Comparison

When comparing the two generations, there is a classic trade-off: YOLOv8 achieves higher mean Average Precision ([mAP](https://docs.ultralytics.com/guides/yolo-performance-metrics)) across the board, while YOLOv5 retains an edge in absolute raw inference speed and parameter count for its smallest variants.

Below is the detailed comparison of their performance metrics on the COCO dataset at an image size of 640 pixels.

| Model                                                                  | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ---------------------------------------------------------------------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| YOLOv5n                                                                | 640                         | 28.0                       | -                                    | **1.12**                                  | **1.9**                  | **4.5**                 |
| YOLOv5s                                                                | 640                         | 37.4                       | -                                    | **1.92**                                  | **7.2**                  | **16.5**                |
| YOLOv5m                                                                | 640                         | 45.4                       | -                                    | **4.03**                                  | **21.2**                 | **49.0**                |
| YOLOv5l                                                                | 640                         | 49.0                       | -                                    | **6.61**                                  | 46.5                     | **109.1**               |
| YOLOv5x                                                                | 640                         | 50.7                       | -                                    | **11.89**                                 | 86.7                     | **205.7**               |
|                                                                        |                             |                            |                                      |                                           |                          |                         |
| [YOLOv8n](https://platform.ultralytics.com/ultralytics/yolov8/yolov8n) | 640                         | **37.3**                   | 80.4                                 | 1.47                                      | 3.2                      | 8.7                     |
| [YOLOv8s](https://platform.ultralytics.com/ultralytics/yolov8/yolov8s) | 640                         | **44.9**                   | 128.4                                | 2.66                                      | 11.2                     | 28.6                    |
| [YOLOv8m](https://platform.ultralytics.com/ultralytics/yolov8/yolov8m) | 640                         | **50.2**                   | 234.7                                | 5.86                                      | 25.9                     | 78.9                    |
| [YOLOv8l](https://platform.ultralytics.com/ultralytics/yolov8/yolov8l) | 640                         | **52.9**                   | 375.2                                | 9.06                                      | **43.7**                 | 165.1                   |
| [YOLOv8x](https://platform.ultralytics.com/ultralytics/yolov8/yolov8x) | 640                         | **53.9**                   | 479.1                                | 14.37                                     | **68.2**                 | 257.8                   |

The data reveals that YOLOv8 provides a substantial boost in accuracy. For instance, `YOLOv8s` achieves a 44.9 mAP compared to `YOLOv5s` at 37.4 mAP, a 7.5-point gain that helps in dense scenes and with small objects. However, for highly constrained environments, `YOLOv5n` remains the most efficient option, with the lowest parameter count and FLOPs.

!!! note "Memory Requirements"

    Both models are highly optimized for lower CUDA memory usage during training compared to heavier architectures like [transformer models](https://www.ultralytics.com/glossary/transformer). This allows practitioners to utilize larger batch sizes on standard GPUs, accelerating the research lifecycle.

## The Ecosystem Advantage

Choosing either YOLOv5 or YOLOv8 grants developers access to the well-maintained [Ultralytics Platform](https://platform.ultralytics.com). This integrated environment offers simple tools for dataset annotation, cloud training, and model monitoring, while the `ultralytics` package adds built-in [hyperparameter tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning). The active development and strong community support ensure that developers can quickly resolve issues and integrate with external tools like [Weights & Biases](https://wandb.ai/site) and [ClearML](https://clear.ml/).

While other frameworks can have steeper learning curves, Ultralytics prioritizes a streamlined user experience, ensuring a favorable trade-off between speed and accuracy suitable for diverse real-world deployment scenarios.

## Beyond v8: Exploring YOLO11 and YOLO26

While YOLOv8 is a highly capable framework, the field of artificial intelligence evolves rapidly. Developers interested in state-of-the-art performance should also explore [YOLO11](https://platform.ultralytics.com/ultralytics/yolo11), which builds upon v8 with improved accuracy and speed.

For new projects, Ultralytics recommends **[Ultralytics YOLO26](https://platform.ultralytics.com/ultralytics/yolo26)**. Released in January 2026, YOLO26 introduces several improvements:

- **End-to-End NMS-Free Design:** Building on the NMS-free training that [YOLOv10](https://docs.ultralytics.com/models/yolov10) introduced to YOLO models, YOLO26's optional one-to-one head (`nms=False`) removes Non-Maximum Suppression post-processing, leading to simpler and faster deployment pipelines.
- **MuSGD Optimizer:** Inspired by the LLM training innovations seen in models like Kimi K2, YOLO26 utilizes a hybrid optimizer for more stable training and faster convergence.
- **Edge Computing:** YOLO26n runs up to **43% faster on CPU** than YOLO11n, making YOLO26 well suited to devices without dedicated GPUs.
- **Enhanced Accuracy:** Utilizing the new progressive loss and small-target-aware label assignment (ProgLoss + STAL), it improves small-object recognition, which is critical for [robotics](https://www.ultralytics.com/glossary/robotics) and aerial drone imagery.

Whether maintaining a legacy system with YOLOv5, scaling a versatile application with YOLOv8, or adopting YOLO26 for new projects, the Ultralytics ecosystem provides the tooling to train, deploy, and maintain modern vision AI.
