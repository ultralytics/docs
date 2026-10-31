---
title: YOLO11 vs RTDETRv2 Comparison
comments: true
description: Compare Ultralytics YOLO11 and RTDETRv2 for real-time object detection, covering COCO mAP, T4 TensorRT speed, parameters, FLOPs, memory use, and deployment.
keywords: RTDETRv2, YOLO11, object detection, Ultralytics, Vision Transformer, YOLO, computer vision, real-time detection, model comparison
---

# YOLO11 vs RTDETRv2

Real-time detection increasingly involves a choice between [convolutional neural networks (CNNs)](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn) and [vision transformers (ViTs)](https://www.ultralytics.com/glossary/vision-transformer-vit). This technical comparison covers two leading [object detection](https://www.ultralytics.com/glossary/object-detection) architectures: **YOLO11**, a highly optimized CNN from Ultralytics, and **RTDETRv2**, an iteration of the Detection Transformer family from Baidu.

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO11", "RTDETRv2"]'></canvas>

This guide compares their architectures, performance metrics, and deployment scenarios to help machine learning engineers choose between them. Both models reach high accuracy, but [Ultralytics YOLO](https://www.ultralytics.com/yolo) models typically offer a better balance of speed, ecosystem support, and ease of use in production.

## YOLO11: The Benchmark for Real-World Versatility

Introduced by Ultralytics, YOLO11 builds on years of YOLO research to deliver a model that is fast, accurate, and versatile. It natively handles [object detection](https://docs.ultralytics.com/tasks/detect), [instance segmentation](https://docs.ultralytics.com/tasks/segment), [image classification](https://docs.ultralytics.com/tasks/classify), [pose estimation](https://docs.ultralytics.com/tasks/pose), and [oriented bounding box (OBB)](https://docs.ultralytics.com/tasks/obb) detection.

- **Authors:** Glenn Jocher and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com)
- **Date:** 2024-09-27
- **GitHub:** [Ultralytics Repository](https://github.com/ultralytics/ultralytics)
- **Docs:** [YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11)

[Learn more about YOLO11](https://platform.ultralytics.com/ultralytics/yolo11){ .md-button }

### Architecture and Strengths

YOLO11 uses a refined CNN backbone and multi-scale feature pyramids, making it resource-efficient. It suits environments with strict hardware constraints, with a small memory footprint during both training and inference. [Ultralytics Platform](https://platform.ultralytics.com) supports YOLO11 for data annotation, cloud training, and model monitoring without stitching together separate MLOps tools.

For [edge deployment](https://docs.ultralytics.com/guides/model-deployment-options), YOLO11 offers low latency. Its lightweight design runs efficiently on devices ranging from Raspberry Pi boards to consumer mobile phones, making it a common choice for smart retail, [manufacturing quality control](https://www.ultralytics.com/solutions/computer-vision-in-manufacturing), and automated traffic management.

## RTDETRv2: Real-Time Transformers by Baidu

RTDETRv2 (Real-Time Detection Transformer version 2) represents Baidu's effort to make transformer-based architectures viable for real-time tasks. It builds upon the original RT-DETR by incorporating a "bag-of-freebies" approach to improve baseline accuracy without inflating inference latency.

- **Authors:** Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu
- **Organization:** [Baidu](https://www.baidu.com/)
- **Date:** 2024-07-24
- **Arxiv:** [2407.17140](https://arxiv.org/abs/2407.17140)
- **GitHub:** [RTDETRv2 Repository](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)
- **Docs:** [RTDETRv2 README](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme)

[Learn more about RTDETRv2](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch){ .md-button }

### Architecture and Strengths

Unlike traditional CNNs, RTDETRv2 uses an encoder-decoder architecture with [self-attention](https://www.ultralytics.com/glossary/self-attention) mechanisms, allowing it to capture global context across an image. This helps in crowded scenes where occlusions are frequent. RTDETRv2 removes the need for [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) in post-processing, relying instead on Hungarian matching during training for one-to-one bipartite matching.

However, transformer models typically need more VRAM and CUDA memory. Training RTDETRv2 from scratch or fine-tuning it on custom datasets often requires high-end GPUs with large memory, which can be a barrier for smaller teams compared to the lighter training footprint of Ultralytics models.

## Performance and Metrics Analysis

Evaluated on the standard [COCO dataset](https://docs.ultralytics.com/datasets/detect/coco), the models show clear trade-offs between parameters, FLOPs, and accuracy ([mAP](https://www.ultralytics.com/glossary/mean-average-precision-map)).

| Model                                                                  | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ---------------------------------------------------------------------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| [YOLO11n](https://platform.ultralytics.com/ultralytics/yolo11/yolo11n) | 640                         | 39.5                       | **56.1**                             | **1.5**                                   | **2.6**                  | **6.5**                 |
| [YOLO11s](https://platform.ultralytics.com/ultralytics/yolo11/yolo11s) | 640                         | 47.0                       | 90.0                                 | 2.5                                       | 9.4                      | 21.6                    |
| [YOLO11m](https://platform.ultralytics.com/ultralytics/yolo11/yolo11m) | 640                         | 51.5                       | 183.2                                | 4.7                                       | 20.1                     | 68.1                    |
| [YOLO11l](https://platform.ultralytics.com/ultralytics/yolo11/yolo11l) | 640                         | 53.4                       | 238.6                                | 6.2                                       | 25.3                     | 87.2                    |
| [YOLO11x](https://platform.ultralytics.com/ultralytics/yolo11/yolo11x) | 640                         | **54.7**                   | 462.8                                | 11.3                                      | 56.9                     | 195.3                   |
|                                                                        |                             |                            |                                      |                                           |                          |                         |
| RTDETRv2-s                                                             | 640                         | 48.1                       | -                                    | 5.03                                      | 20                       | 60                      |
| RTDETRv2-m                                                             | 640                         | 51.9                       | -                                    | 7.51                                      | 36                       | 100                     |
| RTDETRv2-l                                                             | 640                         | 53.4                       | -                                    | 9.76                                      | 42                       | 136                     |
| RTDETRv2-x                                                             | 640                         | 54.3                       | -                                    | 15.03                                     | 76                       | 259                     |

### Unpacking the Results

YOLO11 delivers more accuracy per parameter. YOLO11x reaches a higher mAP<sup>val</sup> (54.7) than RTDETRv2-x (54.3) while using fewer parameters (56.9M vs 76M) and fewer FLOPs (195.3B vs 259B).

YOLO11 is also faster on T4 [TensorRT](https://docs.ultralytics.com/integrations/tensorrt). YOLO11s runs in 2.5 ms versus 5.03 ms for RTDETRv2-s (47.0 vs 48.1 mAP), and YOLO11m is both more accurate (51.5 mAP) and faster (4.7 ms) than RTDETRv2-s. This makes YOLO11 a strong fit for real-time video analytics where frame processing time is the main bottleneck.

!!! note "The Cost of Transformers"

    While RTDETRv2 achieves excellent accuracy through its attention layers, these mechanisms scale quadratically with image resolution, leading to higher VRAM consumption during both training and inference. YOLO11 avoids this cost with efficient convolutional blocks.

## Training Ecosystem and Usability

The core advantage of adopting an Ultralytics model lies in the surrounding ecosystem. Training RTDETRv2 often involves navigating complex research-grade repositories, adjusting intricate bipartite matching loss weights, and managing significant memory overhead.

Ultralytics focuses on developer experience. The unified Python API removes boilerplate code, integrates with tools like [Weights & Biases](https://docs.ultralytics.com/integrations/weights-biases) for [experiment tracking](https://www.ultralytics.com/glossary/experiment-tracking), and handles data augmentation automatically.

Here is how simple it is to train and export a model using the `ultralytics` package:

```python
from ultralytics import YOLO

# Initialize YOLO11 model with pre-trained weights
model = YOLO("yolo11n.pt")

# Train the model efficiently on a local GPU or cloud instance
train_results = model.train(
    data="coco8.yaml",
    epochs=100,
    imgsz=640,
    device=0,  # Utilize CUDA GPU
)

# Export the trained model to ONNX for widespread deployment
export_path = model.export(format="onnx")
```

Once trained, exporting a YOLO11 model to formats like [ONNX](https://docs.ultralytics.com/integrations/onnx), [OpenVINO](https://docs.ultralytics.com/integrations/openvino), or [CoreML](https://docs.ultralytics.com/integrations/coreml) requires a single command, so the same model can run across diverse hardware backends.

!!! tip "Multi-Task Capabilities"

    While RTDETRv2 focuses exclusively on bounding box detection, YOLO11 also supports instance segmentation, image classification, pose estimation, and OBB detection, so you can consolidate multiple vision tasks into a single model family.

## Use Cases and Recommendations

Choosing between YOLO11 and RT-DETR depends on your specific project requirements, deployment constraints, and ecosystem preferences.

### When to Choose YOLO11

YOLO11 is a strong choice for:

- **Production Edge Deployment:** Commercial applications on devices like [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi) or [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson) where reliability and active maintenance are paramount.
- **Multi-Task Vision Applications:** Projects requiring detection, segmentation, classification, pose estimation, and OBB within a single unified framework.
- **Rapid Prototyping and Deployment:** Teams that need to move quickly from data collection to production using the streamlined [Ultralytics Python API](https://docs.ultralytics.com/usage/python).

### When to Choose RT-DETR

RT-DETR is recommended for:

- **Transformer-Based Detection Research:** Projects exploring attention mechanisms and transformer architectures for end-to-end object detection without NMS.
- **High-Accuracy Scenarios with Flexible Latency:** Applications where detection accuracy is the top priority and slightly higher inference latency is acceptable.
- **Large Object Detection:** Scenes with primarily medium-to-large objects where the global attention mechanism of transformers provides a natural advantage.

### When to Choose Ultralytics (YOLO26)

For most new projects, [Ultralytics YOLO26](https://platform.ultralytics.com/ultralytics/yolo26) offers the best combination of performance and developer experience:

- **NMS-Free Edge Deployment:** Applications requiring consistent, low-latency inference without the complexity of Non-Maximum Suppression post-processing.
- **CPU-Only Environments:** Devices without dedicated GPU acceleration, where YOLO26's up to 43% faster CPU inference (YOLO26n vs YOLO11n) provides a decisive advantage.
- **Small Object Detection:** Challenging scenarios like [aerial drone imagery](https://docs.ultralytics.com/datasets/detect/visdrone) or IoT sensor analysis where ProgLoss and STAL improve accuracy on small objects.

## Looking Ahead: The Power of YOLO26

YOLO11 remains a solid production choice, but new projects should start with YOLO26. Released in January 2026, YOLO26 narrows the architectural gap with transformers through an optional end-to-end NMS-free head (`nms=False`, first pioneered in [YOLOv10](https://docs.ultralytics.com/models/yolov10)), which removes NMS post-processing latency and simplifies deployment logic.

YOLO26 also introduces several other changes:

- **MuSGD Optimizer:** Inspired by the [large language model (LLM)](https://www.ultralytics.com/glossary/large-language-model-llm) training techniques of Moonshot AI's Kimi K2, this hybrid of SGD and Muon provides more stable training and faster convergence.
- **DFL Removal:** Removing Distribution Focal Loss simplifies export and improves compatibility with low-power edge devices.
- **ProgLoss + STAL:** Progressive Loss and Small-Target-Aware Label Assignment (STAL) improve small-object recognition, a key requirement for drone surveillance, [agricultural monitoring](https://www.ultralytics.com/solutions/computer-vision-in-agriculture), and IoT edge sensors.
- **Up to 43% Faster CPU Inference:** For deployments without dedicated GPUs, YOLO26n runs up to 43% faster than YOLO11n on CPU.

[Learn more about YOLO26](https://platform.ultralytics.com/ultralytics/yolo26){ .md-button }

For a wider range of architectures, see [YOLOv8](https://platform.ultralytics.com/ultralytics/yolov8), the widely adopted [YOLOv5](https://platform.ultralytics.com/ultralytics/yolov5), and specialized models like [YOLO-World](https://docs.ultralytics.com/models/yolo-world) for open-vocabulary detection. Whether you choose the proven stability of YOLO11 or the newer YOLO26, the Ultralytics ecosystem provides the tools to take your computer vision projects to production.
