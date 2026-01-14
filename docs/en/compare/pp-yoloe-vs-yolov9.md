---
comments: true
description: Explore the differences between PP-YOLOE+ and YOLOv9 with detailed architecture, performance benchmarks, and use case analysis for object detection.
keywords: PP-YOLOE+, YOLOv9, object detection, model comparison, computer vision, anchor-free detector, programmable gradient information, AI models, benchmarking
---

# PP-YOLOE+ vs YOLOv9: Evolution of Real-Time Object Detection Architectures

In the rapidly advancing field of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv), the quest for the optimal balance between inference speed and detection accuracy drives constant innovation. This comparison explores two significant milestones in this journey: **PP-YOLOE+**, an anchor-free detector from Baidu's PaddlePaddle team, and **YOLOv9**, a groundbreaking architecture introducing Programmable Gradient Information (PGI) from the researchers at Academia Sinica.

Both models have pushed the boundaries of [object detection](https://docs.ultralytics.com/tasks/detect/), offering robust solutions for applications ranging from autonomous driving to industrial inspection. This guide analyzes their unique architectural contributions, benchmarks, and deployment considerations.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["PP-YOLOE+", "YOLOv9"]'></canvas>

## Model Origins and Core Philosophies

### PP-YOLOE+: Refined Anchor-Free Detection

**Authors:** PaddlePaddle Authors  
**Organization:** Baidu  
**Date:** April 2, 2022  
**Framework:** PaddlePaddle

PP-YOLOE+ is an evolution of the PP-YOLO series, specifically improving upon PP-YOLOE. It was designed to be a high-performance industrial detector, emphasizing an anchor-free paradigm to simplify the hyperparameter search often associated with anchor-based methods. Using the [PaddlePaddle framework](https://github.com/PaddlePaddle/PaddleDetection/), it introduces a scalable backbone and efficient task alignment learning.

For developers deeply integrated into the Baidu ecosystem, PP-YOLOE+ offers a strong "cloud-to-edge" solution. You can explore the foundational research in their [arXiv paper](https://arxiv.org/abs/2203.16250) or view the implementation on [GitHub](https://github.com/PaddlePaddle/PaddleDetection/).

[Learn more about PP-YOLOE+](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.8.1/configs/ppyoloe/README.md){ .md-button }

### YOLOv9: Addressing the Information Bottleneck

**Authors:** Chien-Yao Wang, Hong-Yuan Mark Liao  
**Organization:** Academia Sinica  
**Date:** February 21, 2024  
**Framework:** PyTorch

YOLOv9 confronts a fundamental issue in deep learning: information loss as data passes through successive layers. By introducing **Programmable Gradient Information (PGI)** and the **Generalized Efficient Layer Aggregation Network (GELAN)**, YOLOv9 ensures that critical feature data is preserved for the [detection head](https://www.ultralytics.com/glossary/detection-head). This results in superior parameter efficiency, allowing smaller models to achieve accuracy comparable to much larger predecessors.

YOLOv9 is fully supported within the Ultralytics ecosystem, allowing for seamless training and deployment. Read the detailed [arXiv pre-print](https://arxiv.org/abs/2402.13616) for a deep dive into the math behind PGI.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

## Architectural Comparison

The architectural divergence between these two models highlights the shift in design philosophy over the two years separating their releases.

### PP-YOLOE+ Architecture

PP-YOLOE+ is characterized by its **CSPRepResStage**, a backbone that combines the gradient flow benefits of CSPNet with the re-parameterization techniques of RepVGG. This allows the model to have complex structures during training but simplify into a streamlined set of convolutions during inference, boosting speed without losing accuracy.

Key features include:

- **Anchor-Free Head:** Eliminates the need for clustering anchor boxes, making the model more robust to varied object shapes.
- **Task Alignment Learning (TAL):** Explicitly aligns the classification and localization tasks, ensuring that the highest confidence scores correspond to the most accurate bounding boxes.
- **ET-Head:** An Efficient Task-aligned head that reduces computational overhead while maintaining precision.

### YOLOv9 Architecture

YOLOv9 takes a different approach, focusing on the flow of gradient information. Deep networks often suffer from the "information bottleneck," where essential data for prediction is lost in deep layers.

Key innovations include:

- **PGI (Programmable Gradient Information):** An auxiliary supervision framework that generates reliable gradients for the main branch, ensuring deep features retain semantic information.
- **GELAN (Generalized Efficient Layer Aggregation Network):** A novel architecture that prioritizes weight efficiency. It allows YOLOv9 to achieve higher [mAP scores](https://www.ultralytics.com/glossary/mean-average-precision-map) with fewer parameters than comparative models like YOLOv8 or PP-YOLOE+.
- **Reversible Functions:** Inspired by [dynamic architecture design](https://docs.ultralytics.com/models/yolo-world/), this ensures that no information is lost during the feed-forward process.

!!! note "The Ultralytics Advantage"

    While PP-YOLOE+ requires the PaddlePaddle framework, YOLOv9 runs natively in PyTorch and is integrated into the `ultralytics` package. This grants YOLOv9 users access to an extensive suite of tools for **[model export](https://docs.ultralytics.com/modes/export/)** (ONNX, TensorRT, CoreML), **[tracking](https://docs.ultralytics.com/modes/track/)**, and **[deployment](https://docs.ultralytics.com/guides/model-deployment-options/)** without complex codebase conversions.

## Performance Benchmarks

The following table compares the performance of PP-YOLOE+ and YOLOv9 on the [COCO dataset](https://docs.ultralytics.com/datasets/detect/coco/). Both model families offer varying sizes (Tiny/Nano to Extra Large) to suit different resource constraints.

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| PP-YOLOE+t | 640                   | 39.9                 | -                              | 2.84                                | 4.85               | 19.15             |
| PP-YOLOE+s | 640                   | 43.7                 | -                              | 2.62                                | 7.93               | 17.36             |
| PP-YOLOE+m | 640                   | 49.8                 | -                              | 5.56                                | 23.43              | 49.91             |
| PP-YOLOE+l | 640                   | 52.9                 | -                              | 8.36                                | 52.2               | 110.07            |
| PP-YOLOE+x | 640                   | 54.7                 | -                              | 14.3                                | 98.42              | 206.59            |
|            |                       |                      |                                |                                     |                    |                   |
| YOLOv9t    | 640                   | 38.3                 | -                              | **2.3**                             | **2.0**            | **7.7**           |
| YOLOv9s    | 640                   | **46.8**             | -                              | 3.54                                | **7.1**            | 26.4              |
| YOLOv9m    | 640                   | **51.4**             | -                              | 6.43                                | **20.0**           | 76.3              |
| YOLOv9c    | 640                   | **53.0**             | -                              | 7.16                                | **25.3**           | **102.1**         |
| YOLOv9e    | 640                   | **55.6**             | -                              | 16.77                               | **57.3**           | **189.0**         |

### Analysis

- **Parameter Efficiency:** YOLOv9 demonstrates superior efficiency. For example, **YOLOv9e** achieves a massive **55.6% mAP** with only 57.3M parameters, whereas **PP-YOLOE+x** requires nearly double the parameters (98.42M) to achieve a lower mAP of 54.7%. This makes YOLOv9 significantly friendlier for memory-constrained environments.
- **Inference Speed:** While PP-YOLOE+ shows competitive speeds on T4 GPUs (TensorRT), YOLOv9 maintains a strong balance, often providing higher accuracy per unit of computation (FLOPs).
- **Model Size:** The compact nature of YOLOv9 (e.g., YOLOv9c vs PP-YOLOE+l) translates to faster download times and reduced storage costs for edge deployments.

## Training and Usability

### Training with Ultralytics

One of the primary advantages of YOLOv9 is its integration into the Ultralytics ecosystem. Training a model is straightforward and requires minimal code. The system automatically handles data augmentation, logging, and distributed training.

```python
from ultralytics import YOLO

# Load a pretrained YOLOv9c model
model = YOLO("yolov9c.pt")

# Train the model on the COCO8 dataset
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference with the best weights
results = model("path/to/image.jpg")
```

In contrast, setting up PP-YOLOE+ typically requires installing PaddlePaddle and cloning the PaddleDetection repository, which may have a steeper learning curve for users already accustomed to PyTorch workflows.

### Ecosystem Benefits

Using Ultralytics models provides access to a well-maintained ecosystem:

- **Broad Dataset Support:** Native support for datasets like [Objects365](https://docs.ultralytics.com/datasets/detect/objects365/), [Global Wheat](https://docs.ultralytics.com/datasets/detect/globalwheat2020/), and [VisDrone](https://docs.ultralytics.com/datasets/detect/visdrone/).
- **Ultralytics Platform:** A unified solution for managing datasets, training, and deployment, which simplifies the MLOps lifecycle.
- **Community Support:** A massive, active community of developers contributing to [GitHub issues](https://github.com/ultralytics/ultralytics/issues) and discussions, ensuring rapid bug fixes and feature requests.

## Use Cases and Recommendations

### When to choose PP-YOLOE+

PP-YOLOE+ remains a strong contender if your infrastructure is already built around the Baidu/PaddlePaddle ecosystem. Its anchor-free design is forgiving for datasets with extreme aspect ratios. It is capable of high-performance serving in environments specifically optimized for Paddle inference engines.

### When to choose Ultralytics YOLO Models

For most researchers and developers, **YOLOv9** (and the newer **YOLO26**) is the recommended choice due to:

- **Versatility:** Support for [Instance Segmentation](https://docs.ultralytics.com/tasks/segment/) and [Panoptic Segmentation](https://www.ultralytics.com/glossary/panoptic-segmentation) alongside detection.
- **Portability:** Effortless export to [TFLite](https://docs.ultralytics.com/integrations/tflite/) for mobile, [OpenVINO](https://docs.ultralytics.com/integrations/openvino/) for Intel CPUs, and [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/) for NVIDIA GPUs.
- **Accuracy-Efficiency Balance:** As shown in the benchmarks, YOLOv9 delivers higher accuracy with significantly fewer parameters, reducing memory usage and computational cost.

!!! tip "Looking Ahead: The Power of YOLO26"

    While YOLOv9 offers exceptional performance, the recently released **[YOLO26](https://docs.ultralytics.com/models/yolo26/)** represents the bleeding edge of computer vision.

    YOLO26 features a native **end-to-end NMS-free design**, eliminating post-processing latency. With the removal of Distribution Focal Loss (DFL) and the introduction of the **MuSGD optimizer** (combining SGD and Muon), YOLO26 offers up to **43% faster CPU inference** compared to previous generations. For new projects requiring maximum efficiency and state-of-the-art accuracy, we recommend evaluating YOLO26.

## Conclusion

Both PP-YOLOE+ and YOLOv9 represent significant achievements in the field of real-time object detection. PP-YOLOE+ successfully refined the anchor-free approach within the PaddlePaddle framework. However, YOLOv9's introduction of PGI and GELAN has set a new standard for parameter efficiency and information preservation in deep networks.

Combined with the ease of use provided by the Ultralytics Python API and the comprehensive support of the Ultralytics Platform, YOLOv9—and its successor YOLO26—offers a more accessible, versatile, and future-proof solution for modern computer vision challenges.
