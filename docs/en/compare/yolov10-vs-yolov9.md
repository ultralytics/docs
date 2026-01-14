---
comments: true
description: Compare YOLOv10 and YOLOv9 object detection models. Explore architectures, metrics, and use cases to choose the best model for your application.
keywords: YOLOv10,YOLOv9,Ultralytics,object detection,real-time AI,computer vision,model comparison,AI deployment,deep learning
---

# YOLOv10 vs. YOLOv9: Advancements in Real-Time Object Detection

The evolution of the YOLO (You Only Look Once) architecture has consistently pushed the boundaries of computer vision, balancing speed and accuracy for real-time applications. This comparison explores **YOLOv10**, known for its groundbreaking NMS-free end-to-end approach, and **YOLOv9**, which introduced architectural innovations like Programmable Gradient Information (PGI). Both models represent significant milestones in the history of object detection, offering distinct advantages for developers and researchers.

For those seeking the absolute latest in performance and efficiency, we also recommend exploring [YOLO26](https://docs.ultralytics.com/models/yolo26/), the newest state-of-the-art model from Ultralytics.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv10", "YOLOv9"]'></canvas>

## Model Overview

### YOLOv10: End-to-End Efficiency

Released in **May 2024** by researchers from **Tsinghua University**, YOLOv10 introduced a paradigm shift by eliminating the need for Non-Maximum Suppression (NMS) during inference. This was achieved through a consistent dual assignment strategy during training, allowing the model to be natively end-to-end. This design significantly reduces latency and simplifies deployment pipelines.

**Key Authors:** Ao Wang, Hui Chen, Lihao Liu, et al.  
**Organization:** Tsinghua University  
**Date:** 2024-05-23  
**Links:** [arXiv](https://arxiv.org/abs/2405.14458) | [GitHub](https://github.com/THU-MIG/yolov10)

[Learn more about YOLOv10](https://docs.ultralytics.com/models/yolov10/){ .md-button }

### YOLOv9: Architectural Innovation

Released in **February 2024** by **Academia Sinica**, YOLOv9 focused on overcoming the information bottleneck problem in deep neural networks. It introduced Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN). These innovations allow the model to retain more semantic information throughout the deep layers, resulting in high accuracy and parameter efficiency.

**Key Authors:** Chien-Yao Wang and Hong-Yuan Mark Liao  
**Organization:** Institute of Information Science, Academia Sinica, Taiwan  
**Date:** 2024-02-21  
**Links:** [arXiv](https://arxiv.org/abs/2402.13616) | [GitHub](https://github.com/WongKinYiu/yolov9)

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

## Performance Comparison

When comparing these two models, it is crucial to look at metrics on standard datasets like [Microsoft COCO](https://docs.ultralytics.com/datasets/detect/coco/). YOLOv10 emphasizes low latency and end-to-end speed, while YOLOv9 excels in parameter efficiency and maintaining high accuracy through its novel gradient path planning.

!!! tip "Performance Balance"

    Ultralytics models are designed to offer the best trade-off between speed and accuracy. While YOLOv9 and YOLOv10 are excellent, users looking for the most optimized deployment experience across edge and cloud environments should consider the [Ultralytics ecosystem](https://www.ultralytics.com) and the newest [YOLO26 models](https://docs.ultralytics.com/models/yolo26/).

### Metrics Analysis

The table below highlights the performance differences. YOLOv10 generally achieves lower latency due to the removal of NMS, making it highly suitable for applications requiring strict real-time constraints. YOLOv9, particularly in its larger variants, demonstrates impressive accuracy-to-parameter ratios.

| Model    | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| -------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv10n | 640                   | 39.5                 | -                              | 1.56                                | 2.3                | **6.7**           |
| YOLOv10s | 640                   | 46.7                 | -                              | **2.66**                            | 7.2                | **21.6**          |
| YOLOv10m | 640                   | 51.3                 | -                              | **5.48**                            | **15.4**           | **59.1**          |
| YOLOv10b | 640                   | 52.7                 | -                              | 6.54                                | 24.4               | 92.0              |
| YOLOv10l | 640                   | 53.3                 | -                              | 8.33                                | 29.5               | 120.3             |
| YOLOv10x | 640                   | 54.4                 | -                              | **12.2**                            | **56.9**           | **160.4**         |
|          |                       |                      |                                |                                     |                    |                   |
| YOLOv9t  | 640                   | 38.3                 | -                              | 2.3                                 | **2.0**            | 7.7               |
| YOLOv9s  | 640                   | **46.8**             | -                              | 3.54                                | **7.1**            | 26.4              |
| YOLOv9m  | 640                   | **51.4**             | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c  | 640                   | **53.0**             | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e  | 640                   | **55.6**             | -                              | 16.77                               | 57.3               | 189.0             |

## Architecture Deep Dive

### YOLOv10 Architecture

The core innovation of YOLOv10 lies in its **Consistent Dual Assignments**. During training, the model uses a "one-to-many" head to provide rich supervisory signals and a "one-to-one" head to ensure unique predictions. This allows the model to be deployed using only the one-to-one head, eliminating the need for NMS post-processing. Additionally, it employs a **Rank-Guided Block Design** to reduce redundancy in different stages of the model, optimizing computational cost without sacrificing [accuracy](https://www.ultralytics.com/glossary/accuracy).

### YOLOv9 Architecture

YOLOv9 introduces **GELAN (Generalized Efficient Layer Aggregation Network)**, which combines the strengths of CSPNet and ELAN to improve parameter utilization. Its other major contribution, **PGI (Programmable Gradient Information)**, addresses the loss of information as data propagates through deep networks. PGI provides an auxiliary supervision branch that guides the learning process, ensuring that the main branch learns more robust features. This makes YOLOv9 particularly effective for tasks requiring high precision, such as [medical image analysis](https://docs.ultralytics.com/datasets/detect/brain-tumor/).

## Use Cases and Applications

The choice between these two models often depends on the specific requirements of your project.

### When to Choose YOLOv10

- **Edge Deployment:** The NMS-free design significantly lowers CPU overhead, making it ideal for mobile devices and embedded systems like the Raspberry Pi.
- **Low Latency Requirements:** Applications like [autonomous driving](https://www.ultralytics.com/glossary/autonomous-vehicles) or high-speed manufacturing lines benefit from the predictable and low latency of YOLOv10.
- **Simple Pipelines:** The removal of post-processing steps simplifies the export process to formats like [ONNX](https://docs.ultralytics.com/integrations/onnx/) or TensorRT.

### When to Choose YOLOv9

- **High Accuracy Research:** If your primary goal is maximizing mAP on complex datasets, YOLOv9e offers excellent performance.
- **Feature Richness:** The GELAN architecture is robust for feature extraction, which can be beneficial in scenarios with occluded or small objects.
- **General Purpose Detection:** For standard server-side deployments where extreme latency optimization is less critical than detection quality.

## Training and Ease of Use with Ultralytics

Both YOLOv10 and YOLOv9 are integrated into the Ultralytics Python package, ensuring a seamless user experience. This integration provides access to a well-maintained ecosystem, including simple CLI commands, extensive documentation, and active community support.

### Training Example

Training either model is straightforward using the Ultralytics API. The framework handles data augmentation, logging, and evaluation automatically.

```python
from ultralytics import YOLO

# Load a model (YOLOv10n or YOLOv9c)
model = YOLO("yolov10n.pt")  # or "yolov9c.pt"

# Train the model on the COCO8 dataset
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Validate the model
model.val()
```

!!! note "Memory Efficiency"

    Ultralytics YOLO models are known for their lower memory requirements during training compared to many transformer-based architectures. This allows researchers to train effective models on consumer-grade GPUs with limited CUDA memory, democratizing access to state-of-the-art [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv).

## Future-Proofing with Ultralytics

While YOLOv9 and YOLOv10 are excellent models, the field of AI moves rapidly. Ultralytics is committed to continuous improvement. Our latest release, **YOLO26**, builds upon the NMS-free design of YOLOv10 but introduces further optimizations like the **MuSGD Optimizer** and **DFL Removal** for even faster inference and better training stability.

YOLO26 also enhances task versatility, offering specialized improvements for [pose estimation](https://docs.ultralytics.com/tasks/pose/), [segmentation](https://docs.ultralytics.com/tasks/segment/), and [oriented bounding box (OBB)](https://docs.ultralytics.com/tasks/obb/) detection, ensuring you have the best tools for any vision task.

## Conclusion

Both YOLOv10 and YOLOv9 represent significant steps forward. YOLOv10's contribution to end-to-end efficiency makes it a favorite for real-time edge applications, while YOLOv9's architectural depth serves high-accuracy needs well. By utilizing these models within the Ultralytics ecosystem, developers gain the advantages of a unified API, robust export options, and a supportive community, ensuring success in their computer vision projects.

For the most advanced capabilities, we encourage users to also evaluate [YOLO26](https://docs.ultralytics.com/models/yolo26/), which combines the best features of previous generations into a unified, high-performance solution.
