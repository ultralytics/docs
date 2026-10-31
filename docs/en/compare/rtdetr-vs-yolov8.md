---
title: RTDETRv2 vs YOLOv8 Comparison
comments: true
description: Compare RTDETRv2 and Ultralytics YOLOv8 for object detection, covering COCO mAP, CPU and TensorRT speed, training memory, task support, and deployment.
keywords: RTDETRv2, YOLOv8, object detection, computer vision, model comparison, deep learning, transformer architecture, real-time AI, Ultralytics
---

# RTDETRv2 vs YOLOv8

Computer vision models for object detection increasingly fall into two camps: [Convolutional Neural Networks (CNNs)](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn) and newer transformer-based architectures. This technical comparison examines how **RTDETRv2**, a real-time detection transformer, compares with **Ultralytics YOLOv8**, one of the most widely adopted CNN models. Both models are capable, but their architectures lead to distinct differences in training, deployment constraints, and performance.

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["RTDETRv2", "YOLOv8"]'></canvas>

## Model Overview: RTDETRv2

RTDETRv2 (Real-Time Detection Transformer version 2) builds on the original [RT-DETR](https://docs.ultralytics.com/models/rtdetr) by optimizing its transformer architecture for real-time inference.

**Key Technical Details:**

- **Authors:** Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu
- **Organization:** [Baidu](https://www.baidu.com/)
- **Date:** 2024-07-24
- **Links:** [arXiv Publication](https://arxiv.org/abs/2407.17140) | [GitHub Repository](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)

### Architecture and Strengths

RTDETRv2 uses a hybrid architecture that combines a CNN backbone with a transformer encoder-decoder. This lets the model consider the whole image at once, which helps it handle complex scenes with overlapping objects. Its defining feature is its native end-to-end design, which skips [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) post-processing. This reduces algorithmic complexity during the final stages of the detection pipeline. Its multi-scale features also help it detect both large structures and small background elements.

### Weaknesses

Despite their strong contextual understanding, transformer-based architectures like RTDETRv2 have high computational overhead during training. They need more CUDA memory, which makes them harder to train on consumer-grade hardware. Setting up a custom dataset and tuning training hyperparameters in the official repository also often requires domain expertise, as it lacks a beginner-friendly software wrapper. Deployment to low-power edge devices such as older [Raspberry Pi hardware](https://docs.ultralytics.com/guides/raspberry-pi) can also be challenging because attention layers are computationally expensive.

[Learn more about RTDETRv2](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch){ .md-button }

## Model Overview: YOLOv8

Since its release, [Ultralytics YOLOv8](https://platform.ultralytics.com/ultralytics/yolov8) has become a widely used standard for production computer vision, pairing a simple developer experience with strong accuracy.

**Key Technical Details:**

- **Authors:** Glenn Jocher, Ayush Chaurasia, and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com/about)
- **Date:** January 10, 2023
- **Links:** [Official Documentation](https://docs.ultralytics.com/models/yolov8) | [GitHub Repository](https://github.com/ultralytics/ultralytics)

### Architecture and Strengths

YOLOv8 uses an [anchor-free](https://www.ultralytics.com/glossary/anchor-free-detectors) CNN architecture with a decoupled head, improving object localization and classification accuracy over previous YOLO generations. Its main strengths are efficiency and versatility. The architecture requires less memory during training than vision transformers, allowing practitioners to run larger [batch sizes](https://www.ultralytics.com/glossary/batch-size) on standard GPUs. The Ultralytics ecosystem also provides a streamlined workflow: the unified Python API handles [hyperparameter tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning), training, validation, and export with just a few lines of code.

### Weaknesses

YOLOv8 does rely on traditional NMS during its post-processing phase. While the Ultralytics engine handles this under the hood efficiently, it technically introduces a slight post-processing latency when compared to natively NMS-free architectures.

[Learn more about YOLOv8](https://platform.ultralytics.com/ultralytics/yolov8){ .md-button }

## Performance and Metrics Comparison

The table below compares COCO validation accuracy ([mAP](https://www.ultralytics.com/glossary/mean-average-precision-map)), CPU and T4 TensorRT speed, parameters, and FLOPs. The two families prioritize different parts of the deployment pipeline.

| Model                                                                  | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ---------------------------------------------------------------------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| RTDETRv2-s                                                             | 640                         | 48.1                       | -                                    | 5.03                                      | 20                       | 60                      |
| RTDETRv2-m                                                             | 640                         | 51.9                       | -                                    | 7.51                                      | 36                       | 100                     |
| RTDETRv2-l                                                             | 640                         | 53.4                       | -                                    | 9.76                                      | 42                       | 136                     |
| RTDETRv2-x                                                             | 640                         | **54.3**                   | -                                    | 15.03                                     | 76                       | 259                     |
|                                                                        |                             |                            |                                      |                                           |                          |                         |
| [YOLOv8n](https://platform.ultralytics.com/ultralytics/yolov8/yolov8n) | 640                         | 37.3                       | **80.4**                             | **1.47**                                  | **3.2**                  | **8.7**                 |
| [YOLOv8s](https://platform.ultralytics.com/ultralytics/yolov8/yolov8s) | 640                         | 44.9                       | 128.4                                | 2.66                                      | 11.2                     | 28.6                    |
| [YOLOv8m](https://platform.ultralytics.com/ultralytics/yolov8/yolov8m) | 640                         | 50.2                       | 234.7                                | 5.86                                      | 25.9                     | 78.9                    |
| [YOLOv8l](https://platform.ultralytics.com/ultralytics/yolov8/yolov8l) | 640                         | 52.9                       | 375.2                                | 9.06                                      | 43.7                     | 165.1                   |
| [YOLOv8x](https://platform.ultralytics.com/ultralytics/yolov8/yolov8x) | 640                         | 53.9                       | 479.1                                | 14.37                                     | 68.2                     | 257.8                   |

!!! tip "Interpreting the Metrics"

    RTDETRv2-x reaches a slightly higher peak mAP (54.3) than YOLOv8x (53.9), and RTDETRv2 is more accurate at each comparable size. YOLOv8 is faster at the small end: YOLOv8s runs nearly twice as fast as RTDETRv2-s on TensorRT (2.66 ms vs 5.03 ms) with almost half the parameters, at 3.2 points lower mAP. At the large end, YOLOv8l and RTDETRv2-l have similar speed and size. Only YOLOv8 reports CPU ONNX speeds here.

### Memory Requirements and Training Efficiency

Training cost matters for independent developers and enterprise teams alike. Ultralytics YOLO models typically require less CUDA memory during the [training process](https://docs.ultralytics.com/modes/train) than transformer architectures. A standard RTDETRv2 model can bottleneck a consumer GPU, whereas YOLOv8 trains reliably on hardware like the NVIDIA RTX 4070.

## Ecosystem, API, and Ease of Use

The supporting software framework is another key differentiator. The Ultralytics ecosystem handles much of the engineering work, and active development, thorough documentation, and community support on [Discord](https://discord.com/invite/ultralytics) help keep projects moving.

YOLOv8 also goes beyond object detection, with native support for [instance segmentation](https://docs.ultralytics.com/tasks/segment), [image classification](https://docs.ultralytics.com/tasks/classify), [pose estimation](https://docs.ultralytics.com/tasks/pose), and [oriented bounding boxes (OBB)](https://docs.ultralytics.com/tasks/obb). RTDETRv2 focuses on detection only.

### Code Example: Unified Simplicity

The Ultralytics Python API runs both model families with the same interface.

```python
from ultralytics import RTDETR, YOLO

# Load an RT-DETR model and a YOLOv8 model
model_transformer = RTDETR("rtdetr-l.pt")
model_cnn = YOLO("yolov8l.pt")

# Predict on a sample image using the exact same API
results_transformer = model_transformer("https://ultralytics.com/images/bus.jpg")
results_cnn = model_cnn("https://ultralytics.com/images/bus.jpg")

# Export YOLOv8 to ONNX for rapid edge deployment
model_cnn.export(format="onnx")
```

Once trained, YOLOv8 exports in one line to [ONNX](https://docs.ultralytics.com/integrations/onnx), [TensorRT](https://docs.ultralytics.com/integrations/tensorrt), and [OpenVINO](https://docs.ultralytics.com/integrations/openvino) for efficient inference across hardware backends.

## Use Cases and Recommendations

Choosing between RT-DETR and YOLOv8 depends on your specific project requirements, deployment constraints, and ecosystem preferences.

### When to Choose RT-DETR

RT-DETR is a strong choice for:

- **Transformer-Based Detection Research:** Projects exploring attention mechanisms and transformer architectures for end-to-end object detection without NMS.
- **High-Accuracy Scenarios with Flexible Latency:** Applications where detection accuracy is the top priority and slightly higher inference latency is acceptable.
- **Large Object Detection:** Scenes with primarily medium-to-large objects where the global attention mechanism of transformers provides a natural advantage.

### When to Choose YOLOv8

YOLOv8 is recommended for:

- **Versatile Multi-Task Deployment:** Projects requiring a proven model for [detection](https://docs.ultralytics.com/tasks/detect), segmentation, classification, and pose estimation within the Ultralytics ecosystem.
- **Established Production Systems:** Existing production environments already built on the YOLOv8 architecture with stable, well-tested deployment pipelines.
- **Broad Community and Ecosystem Support:** Applications benefiting from YOLOv8's extensive tutorials, third-party integrations, and active community resources.

### When to Choose Ultralytics (YOLO26)

For most new projects, [Ultralytics YOLO26](https://platform.ultralytics.com/ultralytics/yolo26) offers the best combination of performance and developer experience:

- **NMS-Free Edge Deployment:** Applications requiring consistent, low-latency inference without the complexity of Non-Maximum Suppression post-processing.
- **CPU-Only Environments:** Devices without dedicated GPU acceleration, where YOLO26n's up to 43% faster CPU inference than YOLO11n provides a decisive advantage.
- **Small Object Detection:** Challenging scenarios like [aerial drone imagery](https://docs.ultralytics.com/datasets/detect/visdrone) or IoT sensor analysis where ProgLoss and STAL improve accuracy on small objects.

## Looking Forward: The YOLO26 Advantage

YOLOv8 remains a milestone model, but computer vision moves quickly. For teams starting new projects in 2026, Ultralytics YOLO26 is the recommended successor.

If you are drawn to the NMS-free design of RTDETRv2, YOLO26 offers an optional **End-to-End NMS-Free Design** (`nms=False`), combining the post-processing simplicity of transformers with the speed of CNNs. YOLO26 also uses the **MuSGD Optimizer**, which brings large language model (LLM) training techniques to vision models for more stable training and faster convergence. With **DFL Removal** (Distribution Focal Loss removed for simpler export and better edge/low-power device compatibility), YOLO26n achieves **up to 43% faster CPU inference** than YOLO11n. Combined with **ProgLoss + STAL** (Progressive Loss and Small-Target-Aware Label Assignment) for better small-object detection, YOLO26 is the recommended upgrade path from both YOLOv8 and RTDETRv2.

For alternative models, see [YOLO11](https://platform.ultralytics.com/ultralytics/yolo11) or read the detailed breakdown of [YOLOv10 vs YOLOv8](https://docs.ultralytics.com/compare/yolov10-vs-yolov8) to see how NMS-free architecture evolved in the YOLO family.
