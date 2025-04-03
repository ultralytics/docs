---
comments: true
description: Discover the key differences between YOLOX and RTDETRv2. Compare performance, architecture, and use cases for optimal object detection model selection.
keywords: YOLOX, RTDETRv2, object detection, YOLOX vs RTDETRv2, performance comparison, Ultralytics, machine learning, computer vision, object detection models
---

# YOLOX vs RTDETRv2: A Technical Comparison for Object Detection

Choosing the right object detection model is crucial for computer vision tasks. This page provides a detailed technical comparison between two popular choices: **YOLOX** and **RTDETRv2**. We will analyze their architectures, performance metrics, and ideal use cases to help you make an informed decision for your project.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOX", "RTDETRv2"]'></canvas>

## YOLOX: High-Performance Anchor-Free Object Detection

**YOLOX** (You Only Look Once X) is an anchor-free object detection model known for its simplicity and high performance.

- **Authors:** Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun
- **Organization:** Megvii
- **Date:** 2021-07-18
- **Arxiv Link:** [https://arxiv.org/abs/2107.08430](https://arxiv.org/abs/2107.08430)
- **GitHub Link:** [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- **Documentation Link:** [https://yolox.readthedocs.io/en/latest/](https://yolox.readthedocs.io/en/latest/)

### Architecture and Key Features

YOLOX distinguishes itself with an **anchor-free approach**, eliminating the need for predefined anchor boxes. This simplifies the model design and reduces hyperparameters, potentially improving generalization. It employs a **decoupled head** for classification and localization, enhancing accuracy. Advanced augmentation techniques like **MixUp** and **Mosaic** are used during training. YOLOX offers multiple model sizes (Nano to XLarge) catering to diverse computational resources, similar to the scalability seen in [Ultralytics YOLO](https://docs.ultralytics.com/models/yolov8/) models.

### Performance Metrics

YOLOX achieves a good balance between speed and accuracy. For instance, **YOLOX-s** achieves **40.5% mAP<sup>val</sup> 50-95** with **9.0M parameters** and an inference speed of **2.56ms** on an NVIDIA T4 GPU with TensorRT. Larger models like **YOLOX-x** reach **51.1% mAP<sup>val</sup> 50-95**, demonstrating scalability for higher accuracy demands at the cost of speed and resources.

### Strengths and Weaknesses

**Strengths:**

- **High Speed and Efficiency:** Optimized for fast inference, suitable for real-time applications.
- **Anchor-Free Design:** Simplifies architecture and training.
- **Scalability:** Offers various model sizes.
- **Strong Performance:** Achieves competitive results among single-stage detectors.

**Weaknesses:**

- **Accuracy Gap:** While performant, it may lag slightly behind more complex transformer-based models like RTDETRv2 in absolute accuracy on certain datasets.

### Ideal Use Cases

YOLOX is well-suited for applications requiring a balance of speed and accuracy:

- **Robotics**: Real-time perception for [robot navigation](https://www.ultralytics.com/glossary/robotics).
- **Surveillance**: Efficient object detection in video streams for [security systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8).
- **Industrial Inspection**: Automated visual inspection on production lines, contributing to improvements in [manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing).
- **Edge Devices**: Deployment on resource-constrained devices due to its efficient model sizes, similar to the versatility of [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) on platforms like [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/).

[Learn more about YOLOX](https://yolox.readthedocs.io/en/latest/){ .md-button }

## RTDETRv2: High Accuracy Real-Time Detection Transformer v2

**RTDETRv2** (Real-Time Detection Transformer version 2) leverages Vision Transformers (ViT) for object detection, aiming for high accuracy while maintaining real-time performance.

- **Authors:** Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu
- **Organization:** Baidu
- **Date:** 2023-04-17 (Original RT-DETR), 2024-07-24 (RTDETRv2 improvements)
- **Arxiv Link:** [https://arxiv.org/abs/2304.08069](https://arxiv.org/abs/2304.08069) (Original RT-DETR), [https://arxiv.org/abs/2407.17140](https://arxiv.org/abs/2407.17140) (RTDETRv2)
- **GitHub Link:** [https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)
- **Documentation Link:** [https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme)

### Architecture and Key Features

RTDETRv2 employs a **transformer-based architecture**, enabling it to capture global context within images through self-attention mechanisms. This differs from traditional CNN-based models like YOLOX. It combines CNNs for feature extraction with transformer layers, aiming for state-of-the-art accuracy. Like YOLOX, it is also **anchor-free**.

### Performance Metrics

RTDETRv2 models prioritize accuracy. **RTDETRv2-s** achieves **48.1% mAP<sup>val</sup> 50-95** with **20M parameters** and an inference speed of **5.03ms** on a T4 TensorRT10. The larger **RTDETRv2-x** reaches **54.3% mAP<sup>val</sup> 50-95**, showcasing high accuracy at increased computational cost.

### Strengths and Weaknesses

**Strengths:**

- **High Accuracy:** Transformer architecture enables superior accuracy, especially in complex scenes.
- **Real-Time Performance:** Achieves competitive speeds, particularly with hardware acceleration like [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/).
- **Robust Feature Extraction:** Effectively captures global context.

**Weaknesses:**

- **Larger Model Size & Memory:** Transformer models generally have higher parameter counts and FLOPs, demanding more computational resources and significantly more CUDA memory during training compared to efficient CNN models like [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/).
- **Inference Speed:** May be slower than highly optimized models like YOLOX or [Ultralytics YOLOv10](https://docs.ultralytics.com/models/yolov10/) on resource-constrained devices.
- **Complexity:** Transformer architectures can be more complex to train and optimize.

### Ideal Use Cases

RTDETRv2 is suited for applications where high accuracy is paramount and sufficient resources are available:

- **Autonomous Vehicles**: Reliable perception for [self-driving cars](https://www.ultralytics.com/solutions/ai-in-automotive).
- **Medical Imaging**: Precise detection in [medical images](https://www.ultralytics.com/solutions/ai-in-healthcare).
- **High-Resolution Analysis**: Detailed analysis of large images like [satellite imagery](https://www.ultralytics.com/blog/using-computer-vision-to-analyse-satellite-imagery).
- **Complex Robotics**: Accurate object interaction in challenging environments.

[Learn more about RTDETRv2](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme){ .md-button }

## Performance Comparison

The table below summarizes the performance of various YOLOX and RTDETRv2 model variants on the COCO val dataset.

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOXnano  | 416                   | 25.8                 | -                              | -                                   | **0.91**           | **1.08**          |
| YOLOXtiny  | 416                   | 32.8                 | -                              | -                                   | 5.06               | 6.45              |
| YOLOXs     | 640                   | 40.5                 | -                              | **2.56**                            | 9.0                | 26.8              |
| YOLOXm     | 640                   | 46.9                 | -                              | 5.43                                | 25.3               | 73.8              |
| YOLOXl     | 640                   | 49.7                 | -                              | 9.04                                | 54.2               | 155.6             |
| YOLOXx     | 640                   | 51.1                 | -                              | 16.1                                | 99.1               | 281.9             |
|            |                       |                      |                                |                                     |                    |                   |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | **54.3**             | -                              | 15.03                               | 76                 | 259               |

### Why Choose Ultralytics YOLO Models?

While YOLOX and RTDETRv2 offer strong capabilities, [Ultralytics YOLO models](https://docs.ultralytics.com/models/) like [YOLOv8](https://docs.ultralytics.com/models/yolov8/) and [YOLO11](https://docs.ultralytics.com/models/yolo11/) provide several advantages:

- **Ease of Use:** Streamlined Python API, extensive [documentation](https://docs.ultralytics.com/), and numerous [guides](https://docs.ultralytics.com/guides/) simplify development.
- **Well-Maintained Ecosystem:** Active development, strong community support, frequent updates, readily available pre-trained weights, and integration with [Ultralytics HUB](https://docs.ultralytics.com/hub/) for seamless [MLOps](https://www.ultralytics.com/glossary/machine-learning-operations-mlops).
- **Performance Balance:** Excellent trade-off between speed and accuracy, suitable for diverse real-world scenarios.
- **Memory Efficiency:** Lower memory requirements during training and inference compared to transformer models like RTDETRv2, which often demand significant CUDA memory.
- **Versatility:** Support for multiple tasks beyond object detection, including [segmentation](https://docs.ultralytics.com/tasks/segment/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), [classification](https://docs.ultralytics.com/tasks/classify/), and [tracking](https://docs.ultralytics.com/modes/track/).
- **Training Efficiency:** Faster training times and efficient resource utilization.

For users exploring alternatives, consider comparing these models with others like [YOLOv5](https://docs.ultralytics.com/models/yolov5/), [YOLOv9](https://docs.ultralytics.com/models/yolov9/), or checking comparisons such as [YOLOv8 vs YOLOX](https://docs.ultralytics.com/compare/yolov8-vs-yolox/) and [RT-DETR vs YOLOv8](https://docs.ultralytics.com/compare/rtdetr-vs-yolov8/).
