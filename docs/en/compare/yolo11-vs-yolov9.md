---
title: YOLO11 vs YOLOv9 Comparison
comments: true
description: Compare Ultralytics YOLO11 and YOLOv9 for real-time object detection, covering PGI and GELAN vs YOLO11 design, COCO mAP, TensorRT speed, and training workflow.
keywords: YOLO11, YOLOv9, model comparison, object detection, computer vision, Ultralytics, YOLO architecture, YOLO performance, real-time processing
---

# YOLO11 vs YOLOv9

Real-time [object detection](https://www.ultralytics.com/glossary/object-detection) architectures continue to improve. Two notable milestones are [Ultralytics YOLO11](https://platform.ultralytics.com/ultralytics/yolo11) and YOLOv9. Both perform well, but they take different approaches to the core challenges of deep learning inference and training.

This guide compares YOLO11 and YOLOv9, analyzing their architectures, performance metrics, and ideal deployment scenarios to help you choose the right model for your next computer vision project.

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO11", "YOLOv9"]'></canvas>

## Model Overview

### Ultralytics YOLO11

YOLO11 is an optimized, versatile model designed for production environments. It balances high accuracy with the practical requirements of [edge computing](https://www.ultralytics.com/glossary/edge-computing) and large-scale deployment.

- **Authors:** Glenn Jocher and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com)
- **Date:** 2024-09-27
- **GitHub:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Docs:** [https://docs.ultralytics.com/models/yolo11](https://docs.ultralytics.com/models/yolo11)

[Learn more about YOLO11](https://platform.ultralytics.com/ultralytics/yolo11){ .md-button }

### YOLOv9

YOLOv9 is an academic contribution that introduces new concepts to reduce information loss in deep neural networks, with a focus on feature extraction.

- **Authors:** Chien-Yao Wang and Hong-Yuan Mark Liao
- **Organization:** Institute of Information Science, Academia Sinica, Taiwan
- **Date:** 2024-02-21
- **arXiv:** [https://arxiv.org/abs/2402.13616](https://arxiv.org/abs/2402.13616)
- **GitHub:** [https://github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)
- **Docs:** [https://docs.ultralytics.com/models/yolov9](https://docs.ultralytics.com/models/yolov9)

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9){ .md-button }

## Architectural Innovations

### YOLOv9: Programmable Gradient Information

YOLOv9 tackles the "information bottleneck" problem—where data is lost as it passes through successive layers of a deep network. To solve this, the authors introduced Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN). PGI helps the gradients used to update weights during backpropagation retain more complete information, resulting in accurate feature representations. The GELAN architecture maximizes parameter efficiency, allowing YOLOv9 to achieve high accuracy with a relatively lightweight structure.

### YOLO11: Ecosystem and Efficiency

While YOLOv9 focuses on gradient flow, YOLO11 is engineered for real-world reliability and versatility. It refines the core YOLO architecture to keep CUDA memory requirements during training lower than those of transformer-heavy alternatives. YOLO11 is also more than an object detector: while YOLOv9 in Ultralytics covers detection and segmentation, YOLO11 supports [instance segmentation](https://docs.ultralytics.com/tasks/segment), [image classification](https://docs.ultralytics.com/tasks/classify), [pose estimation](https://docs.ultralytics.com/tasks/pose), and [oriented bounding boxes (OBB)](https://docs.ultralytics.com/tasks/obb).

!!! tip "Streamlined Development"

    One of YOLO11's greatest strengths is its integration into the [Ultralytics Python package](https://docs.ultralytics.com/usage/python), which abstracts away the complexities of data loading, augmentation, and distributed training into a unified API.

## Performance Comparison

When selecting a model for production, weigh the trade-off between mean Average Precision (mAP), inference speed, and parameter count.

| Model                                                                  | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ---------------------------------------------------------------------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| [YOLO11n](https://platform.ultralytics.com/ultralytics/yolo11/yolo11n) | 640                         | 39.5                       | **56.1**                             | **1.5**                                   | 2.6                      | **6.5**                 |
| [YOLO11s](https://platform.ultralytics.com/ultralytics/yolo11/yolo11s) | 640                         | 47.0                       | 90.0                                 | 2.5                                       | 9.4                      | 21.6                    |
| [YOLO11m](https://platform.ultralytics.com/ultralytics/yolo11/yolo11m) | 640                         | 51.5                       | 183.2                                | 4.7                                       | 20.1                     | 68.1                    |
| [YOLO11l](https://platform.ultralytics.com/ultralytics/yolo11/yolo11l) | 640                         | 53.4                       | 238.6                                | 6.2                                       | 25.3                     | 87.2                    |
| [YOLO11x](https://platform.ultralytics.com/ultralytics/yolo11/yolo11x) | 640                         | 54.7                       | 462.8                                | 11.3                                      | 56.9                     | 195.3                   |
|                                                                        |                             |                            |                                      |                                           |                          |                         |
| YOLOv9t                                                                | 640                         | 38.3                       | -                                    | 2.3                                       | **2.0**                  | 7.7                     |
| YOLOv9s                                                                | 640                         | 46.8                       | -                                    | 3.54                                      | 7.2                      | 26.7                    |
| YOLOv9m                                                                | 640                         | 51.4                       | -                                    | 6.43                                      | 20.1                     | 76.8                    |
| YOLOv9c                                                                | 640                         | 53.0                       | -                                    | 7.16                                      | 25.5                     | 102.8                   |
| YOLOv9e                                                                | 640                         | **55.6**                   | -                                    | 16.77                                     | 58.1                     | 192.5                   |

YOLOv9e achieves the highest overall accuracy (55.6 mAP), which makes it useful for academic benchmarking, though at 16.77 ms on T4 TensorRT versus 11.3 ms for YOLO11x. YOLO11 offers a better speed-to-accuracy ratio at every scale. For instance, YOLO11m achieves 51.5 mAP at 4.7 ms (TensorRT), versus 51.4 mAP at 6.43 ms for the similarly sized YOLOv9m.

## Training Methodologies and Ecosystem

The developer experience differs considerably between the two frameworks.

### Training YOLOv9

Training YOLOv9 from the original repository often requires working with customized research code, managing specific dependency versions, and using complex command-line arguments, which can slow down enterprise teams. YOLOv9 weights are also available in the `ultralytics` package, which removes most of this setup.

### Training YOLO11

YOLO11 uses the well-maintained Ultralytics Python API, which takes you from installation to a trained model in a few steps. Training is supported by readily available pretrained weights and an active community.

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 small model
model = YOLO("yolo11s.pt")

# Train on a custom dataset with built-in augmentations
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Export to ONNX format for deployment
model.export(format="onnx")
```

With just three lines of Python, developers can load a model, initiate training with optimized hyperparameter defaults, and export the trained architecture to frameworks like [ONNX](https://onnx.ai/) or [TensorRT](https://developer.nvidia.com/tensorrt) for edge deployment.

## Real-World Applications

### When to Choose YOLOv9

YOLOv9 is a good choice for researchers exploring deep learning architectures and gradient-information techniques. Its largest variant, YOLOv9e, suits accuracy-critical analytics on dense scenes where deployment complexity is secondary to algorithmic performance.

### When to Choose YOLO11

YOLO11 is the stronger choice for production. Its [object detection](https://docs.ultralytics.com/tasks/detect) capabilities suit [smart city traffic management](https://www.ultralytics.com/blog/optimizingtraffic-management-with-ultralytics-yolo11) and edge devices like the Raspberry Pi or NVIDIA Jetson. Furthermore, its versatility across various tasks means a single development pipeline can handle [segmentation in manufacturing](https://www.ultralytics.com/blog/improving-manufacturing-with-computer-vision) and [pose estimation in sports analytics](https://www.ultralytics.com/blog/using-pose-estimation-to-perfect-your-running-technique).

## Enter YOLO26

For developers starting new projects today, Ultralytics recommends [YOLO26](https://platform.ultralytics.com/ultralytics/yolo26) (released January 2026), the latest Ultralytics YOLO model.

YOLO26 combines recent advances into a production-ready model:

- **End-to-End NMS-Free Design:** YOLO26's optional one-to-one head (`nms=False`) skips Non-Maximum Suppression (NMS) post-processing, resulting in simpler and faster deployment pipelines.
- **DFL Removal:** The removal of Distribution Focal Loss improves compatibility with low-power devices and edge AI accelerators.
- **MuSGD Optimizer:** Inspired by large language model (LLM) training methods, the MuSGD optimizer (a hybrid of SGD and Muon) offers stable training and faster convergence.
- **Up to 43% Faster CPU Inference:** YOLO26n runs up to 43% faster than YOLO11n on CPU, which suits edge computing devices without dedicated GPUs.
- **ProgLoss + STAL:** Progressive Loss and Small-Target-Aware Label Assignment (STAL) improve small-object recognition, which is critical for [agricultural monitoring](https://www.ultralytics.com/blog/the-changing-landscape-of-ai-in-agriculture) and aerial imagery.

Users interested in exploring diverse architectures might also want to look into [RT-DETR](https://docs.ultralytics.com/models/rtdetr) for transformer-based detection or [YOLO-World](https://docs.ultralytics.com/models/yolo-world) for zero-shot open-vocabulary detection.

## Conclusion

Both YOLO11 and YOLOv9 are important models in the history of computer vision. YOLOv9 offers notable architectural ideas for feature retention. For most real-world deployments, from enterprise AI applications to [mobile edge devices](https://www.ultralytics.com/glossary/edge-ai), the ease of use, memory efficiency, and versatile task support of YOLO11 give it the advantage. For new systems, the newer YOLO26 adds faster CPU inference and optional NMS-free deployment.
