---
description: Discover the key differences between YOLOX and RTDETRv2. Compare performance, architecture, and use cases for optimal object detection model selection.
keywords: YOLOX, RTDETRv2, object detection, YOLOX vs RTDETRv2, performance comparison, Ultralytics, machine learning, computer vision, object detection models
---

# YOLOX vs RTDETRv2: A Technical Comparison for Object Detection

Choosing the right object detection model is crucial for computer vision tasks. Ultralytics offers a range of models to suit different needs, and this page provides a detailed technical comparison between two popular choices: **YOLOX** and **RTDETRv2**. We will analyze their architectures, performance, and ideal use cases to help you make an informed decision.

<script async src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOX", "RTDETRv2"]'></canvas>

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOXnano  | 416                   | 25.8                 | -                              | -                                   | 0.91               | 1.08              |
| YOLOXtiny  | 416                   | 32.8                 | -                              | -                                   | 5.06               | 6.45              |
| YOLOXs     | 640                   | 40.5                 | -                              | 2.56                                | 9.0                | 26.8              |
| YOLOXm     | 640                   | 46.9                 | -                              | 5.43                                | 25.3               | 73.8              |
| YOLOXl     | 640                   | 49.7                 | -                              | 9.04                                | 54.2               | 155.6             |
| YOLOXx     | 640                   | 51.1                 | -                              | 16.1                                | 99.1               | 281.9             |
|            |                       |                      |                                |                                     |                    |                   |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | 54.3                 | -                              | 15.03                               | 76                 | 259               |

## YOLOX: High-Performance Anchor-Free Object Detection

**YOLOX** (You Only Look Once X) is an anchor-free object detection model known for its simplicity and high performance. Developed by **Megvii** and introduced on **2021-07-18** ([Arxiv Link: YOLOX Paper](https://arxiv.org/abs/2107.08430)), YOLOX builds upon the YOLO series by streamlining the architecture and training process. It aims to bridge the gap between research and industrial applications with its efficient design and strong performance. The official **GitHub repository** for YOLOX is available at [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), and comprehensive **documentation** can be found at [YOLOX Read the Docs](https://yolox.readthedocs.io/en/latest/).

### Architecture and Key Features

YOLOX distinguishes itself with an **anchor-free approach**, eliminating the need for predefined anchor boxes. This simplifies the model design and reduces the number of hyperparameters, leading to easier training and better generalization, particularly for datasets with varying object sizes. It employs a **decoupled head** for classification and localization, enhancing training efficiency and accuracy by allowing specialized optimization for each task. Advanced augmentation techniques like **MixUp** and **Mosaic** are utilized to improve robustness and generalization. YOLOX offers multiple model sizes, from Nano to XLarge, catering to diverse computational resources.

### Performance Metrics

YOLOX achieves a good balance between speed and accuracy. For example, **YOLOX-s** achieves **40.5% mAP**<sup>val</sup><sub>50-95</sub> with **9.0M parameters** and an inference speed of **2.56ms** on a T4 TensorRT10. Larger models like **YOLOX-x** reach **51.1% mAP**<sup>val</sup><sub>50-95</sub>, demonstrating scalability for higher accuracy demands at the cost of speed and resources.

### Strengths and Weaknesses

**Strengths:**

- **High Speed and Efficiency:** Optimized for fast inference, making it suitable for real-time applications.
- **Anchor-Free Design:** Simplifies architecture and training, improving generalization.
- **Scalability:** Offers various model sizes to accommodate different hardware constraints.
- **Strong Performance:** Achieves state-of-the-art results among single-stage detectors.

**Weaknesses:**

- **Accuracy Gap:** While highly performant, it may still lag slightly behind more complex two-stage detectors or transformer-based models in terms of absolute accuracy on certain complex datasets, although YOLOX significantly bridges this gap compared to prior YOLO versions.

### Ideal Use Cases

YOLOX is well-suited for applications requiring a balance of speed and accuracy, including:

- **Robotics**: Real-time perception for robot navigation and interaction.
- **Surveillance**: Efficient object detection in video streams for security applications, such as in security alarm systems.
- **Industrial Inspection**: Automated visual inspection on production lines, contributing to improvements in manufacturing.
- **Edge Devices**: Deployment on resource-constrained devices due to its efficient model sizes, similar to YOLOv5's versatility on edge devices like Raspberry Pi and NVIDIA Jetson.

[Learn more about YOLOX](https://yolox.readthedocs.io/en/latest/){ .md-button }

## RTDETRv2: High Accuracy Real-Time Detection Transformer v2

**RTDETRv2** (Real-Time Detection Transformer version 2) represents a different approach by leveraging Vision Transformers (ViT) for object detection, aiming for a blend of high accuracy and real-time performance. Developed by **Baidu** and introduced on **2023-04-17** ([Arxiv Link: RT-DETR Paper](https://arxiv.org/abs/2304.08069), [Arxiv Link: RTDETRv2 Paper](https://arxiv.org/abs/2407.17140)), RTDETRv2 utilizes transformer architectures to capture global context, potentially leading to higher accuracy, especially in complex scenes. The **GitHub repository** is available at [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch), with documentation in the repository's README.

### Architecture and Key Features

RTDETRv2 employs a **transformer-based architecture**, enabling it to capture global context within images through self-attention mechanisms. This differs from traditional CNN-based architectures like YOLO, which primarily rely on local convolutional operations. RTDETRv2 combines CNNs for initial feature extraction with transformer layers for global context modeling, aiming to achieve state-of-the-art accuracy while maintaining competitive inference speeds. It is also designed as an **anchor-free detector**, simplifying the detection process, similar to YOLOX.

### Performance Metrics

RTDETRv2 models prioritize accuracy. For instance, **RTDETRv2-s** achieves **48.1% mAP**<sup>val</sup><sub>50-95</sub> with **20M parameters** and an inference speed of **5.03ms** on a T4 TensorRT10. The larger variants, like **RTDETRv2-x**, reach **54.3% mAP**<sup>val</sup><sub>50-95</sub>, showcasing its capability for high-accuracy tasks, though with increased computational cost. RTDETRv2's performance places it competitively against models like YOLOv8 and YOLOv9 in terms of accuracy, while aiming to maintain real-time speeds.

### Strengths and Weaknesses

**Strengths:**

- **High Accuracy:** Transformer architecture enables superior object detection accuracy, particularly in complex scenarios.
- **Real-Time Performance:** Achieves competitive inference speeds, making it suitable for real-time applications, especially with hardware acceleration.
- **Robust Feature Extraction:** Vision Transformers effectively capture global context and intricate details, enhancing detection quality.

**Weaknesses:**

- **Larger Model Size:** Models like RTDETRv2-x have a larger parameter count and FLOPs compared to smaller YOLO models, requiring more computational resources.
- **Inference Speed:** While real-time capable, inference speed might be slower than the fastest YOLO models, such as YOLOv5 or YOLOX, on resource-constrained devices.

### Ideal Use Cases

RTDETRv2 is ideally suited for applications where high accuracy is paramount and sufficient computational resources are available, such as:

- **Autonomous Vehicles**: For reliable and precise perception of the environment, crucial for safety and navigation in AI in self-driving cars.
- **Medical Imaging**: For precise detection of anomalies in medical images, aiding in diagnostics and potentially in areas like tumor detection in medical imaging.
- **High-Resolution Image Analysis**: Applications requiring detailed analysis of large images, such as satellite imagery analysis and urban planning by analyzing satellite imagery to uncover signs of urban decline.
- **Robotics**: Enabling robots to accurately interact with and manipulate objects in complex settings, similar to how AI plays a role in robotics.

[Learn more about RTDETRv2](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme){ .md-button }

Users interested in other high-performance object detection models might also consider exploring Ultralytics YOLOv8 and YOLOv11 for efficient and versatile solutions, or RT-DETR for another transformer-based option. For comparisons with other models, refer to pages like YOLOv5 vs RT-DETR v2 and YOLOv7 vs RT-DETR for further insights.