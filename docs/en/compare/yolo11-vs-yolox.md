---
comments: true
description: Explore YOLO11 and YOLOX, two leading object detection models. Compare architecture, performance, and use cases to select the best model for your needs.
keywords: YOLO11, YOLOX, object detection, machine learning, computer vision, model comparison, deep learning, Ultralytics, real-time detection, anchor-free models
---

# Ultralytics YOLO11 vs YOLOX: A Technical Comparison

In the rapidly evolving landscape of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv), the YOLO (You Only Look Once) family of models has consistently set the standard for real-time [object detection](https://docs.ultralytics.com/tasks/detect/). This comparison explores two significant iterations in this lineage: **Ultralytics YOLO11**, released in 2024, and **YOLOX**, a 2021 release from Megvii. While both models share the YOLO heritage, they represent different eras of architectural innovation and optimization.

This guide provides an in-depth analysis of their architectures, performance metrics on the [COCO dataset](https://docs.ultralytics.com/datasets/detect/coco/), and suitability for modern deployment scenarios.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO11", "YOLOX"]'></canvas>

## Performance Metrics Comparison

The following table benchmarks [YOLO11](https://docs.ultralytics.com/models/yolo11/) against YOLOX. The data highlights the advancements in efficiency and accuracy achieved by the Ultralytics team over the years between these releases.

| Model       | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ----------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| **YOLO11n** | 640                   | 39.5                 | **56.1**                       | **1.5**                             | **2.6**            | **6.5**           |
| YOLO11s     | 640                   | **47.0**             | **90.0**                       | **2.5**                             | 9.4                | **21.5**          |
| **YOLO11m** | 640                   | **51.5**             | **183.2**                      | **4.7**                             | **20.1**           | **68.0**          |
| YOLO11l     | 640                   | **53.4**             | **238.6**                      | **6.2**                             | **25.3**           | **86.9**          |
| **YOLO11x** | 640                   | **54.7**             | **462.8**                      | **11.3**                            | **56.9**           | **194.9**         |
|             |                       |                      |                                |                                     |                    |                   |
| YOLOXnano   | 416                   | 25.8                 | -                              | -                                   | 0.91               | 1.08              |
| YOLOXtiny   | 416                   | 32.8                 | -                              | -                                   | 5.06               | 6.45              |
| YOLOXs      | 640                   | 40.5                 | -                              | 2.56                                | 9.0                | 26.8              |
| YOLOXm      | 640                   | 46.9                 | -                              | 5.43                                | 25.3               | 73.8              |
| YOLOXl      | 640                   | 49.7                 | -                              | 9.04                                | 54.2               | 155.6             |
| YOLOXx      | 640                   | 51.1                 | -                              | 16.1                                | 99.1               | 281.9             |

## Ultralytics YOLO11

Released by **Ultralytics** on September 27, 2024, [YOLO11](https://docs.ultralytics.com/models/yolo11/) builds upon the robust foundation of its predecessors to deliver state-of-the-art (SOTA) performance. Authored by Glenn Jocher and Jing Qiu, YOLO11 introduces a refined architecture designed to maximize [feature extraction](https://www.ultralytics.com/glossary/feature-extraction) while minimizing computational cost.

### Key Features and Strengths

- **Superior Efficiency:** YOLO11m achieves a higher mean Average Precision (mAP) than [YOLOv8m](https://docs.ultralytics.com/models/yolov8/) while using **22% fewer parameters**. This reduction in model size translates directly to lower memory usage and faster inference on edge devices.
- **Versatility:** Unlike many competitors restricted to bounding boxes, YOLO11 natively supports a wide array of [computer vision tasks](https://docs.ultralytics.com/tasks/), including [instance segmentation](https://docs.ultralytics.com/tasks/segment/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), [oriented object detection (OBB)](https://docs.ultralytics.com/tasks/obb/), and [image classification](https://docs.ultralytics.com/tasks/classify/).
- **Ecosystem Integration:** YOLO11 is fully integrated into the Ultralytics ecosystem, providing seamless access to tools like [Ultralytics HUB](https://hub.ultralytics.com/) for no-code training and extensive deployment options via the `ultralytics` Python package.

### Architecture

YOLO11 employs a modernized backbone and neck architecture, incorporating C3k2 blocks and an optimized SPPF module. These changes enhance the model's ability to capture intricate patterns in data, making it particularly effective for challenging scenarios like small object detection in [aerial imagery](https://www.ultralytics.com/solutions/ai-in-agriculture).

!!! info "Training Efficiency"

    YOLO11 models utilize advanced augmentation pipelines and optimized hyperparameters out of the box. This allows users to achieve high accuracy with fewer training epochs compared to older architectures, saving significant GPU hours and electricity.

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

## YOLOX

Introduced in July 2021 by researchers at **Megvii** (Zheng Ge et al.), YOLOX was a pivotal release that shifted the YOLO paradigm towards an anchor-free design. It aimed to bridge the gap between the research community and industrial applications by simplifying the detection head and incorporating advanced label assignment strategies.

### Key Features

- **Anchor-Free Design:** YOLOX removed the anchor box mechanism used in previous versions (like YOLOv4 and YOLOv5), simplifying the design and reducing the number of hyperparameters developers needed to tune.
- **Decoupled Head:** The model separates the classification and regression tasks into different heads, which helped improve convergence speed and accuracy at the time of its release.
- **SimOTA:** YOLOX introduced SimOTA, an advanced dynamic label assignment strategy that treats the training process as an optimal transport problem.

### Limitations in 2026

While revolutionary in 2021, YOLOX shows its age when compared to modern counterparts like YOLO11.

- **Lower Accuracy-to-Compute Ratio:** As seen in the table above, YOLOX requires significantly more FLOPs and parameters to achieve accuracy levels that YOLO11 surpasses with lighter models.
- **Limited Task Support:** YOLOX is primarily focused on object detection, lacking native, integrated support for complex tasks like segmentation or pose estimation within a unified API.
- **Complex Deployment:** Without the streamlined [export modes](https://docs.ultralytics.com/modes/export/) found in the Ultralytics framework, deploying YOLOX to formats like TensorRT or CoreML can require more manual engineering effort.

[Learn more about YOLOX](https://github.com/Megvii-BaseDetection/YOLOX){ .md-button }

## Comparison: Architecture and Usability

The choice between these two models often comes down to the trade-off between using a legacy research tool (YOLOX) versus a modern, production-grade platform (YOLO11).

### Ease of Use and Ecosystem

Ultralytics models are renowned for their developer-friendly experience. Installing and running YOLO11 takes seconds, whereas older research repositories often require complex environment setups.

**YOLO11 Python Example:**
The Ultralytics API unifies training, validation, and inference into a few lines of code.

```python
from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Run inference on an image
results = model("https://ultralytics.com/images/bus.jpg")

# View results
results[0].show()
```

In contrast, utilizing YOLOX typically involves cloning a GitHub repository and navigating a more complex directory structure for configuration and execution.

### Performance Balance and Memory

For real-world applications where hardware resources are constrained, YOLO11 offers a distinct advantage.

- **Edge Deployment:** The **YOLO11n** model, with only 2.6M parameters, is engineered for devices like the Raspberry Pi or NVIDIA Jetson Nano. It offers a 39.5 mAP at speeds that allow for real-time video processing.
- **Memory Efficiency:** Training YOLO11 generally requires less [CUDA memory](https://docs.ultralytics.com/guides/yolo-common-issues/#cuda-out-of-memory) than YOLOX for comparable batch sizes, thanks to optimized memory management in the Ultralytics engine.

### Versatility and Real-World Applications

YOLO11's versatility makes it the superior choice for complex [industrial applications](https://www.ultralytics.com/solutions/ai-in-manufacturing).

- **Autonomous Vehicles:** YOLO11's high speed and accuracy are critical for detecting pedestrians and vehicles in real-time. The ability to perform [object tracking](https://docs.ultralytics.com/modes/track/) natively further enhances its utility here.
- **Healthcare:** In [medical imaging](https://www.ultralytics.com/solutions/ai-in-healthcare), where precision is paramount, YOLO11's segmentation capabilities can delineate tumor boundaries or analyze cell structures, tasks that standard YOLOX cannot perform without significant modification.
- **Retail Analytics:** For [smart retail](https://www.ultralytics.com/solutions/ai-in-retail), YOLO11 can track customer movement (Pose estimation) and monitor inventory levels (Object detection) simultaneously using the same unified framework.

!!! tip "Did You Know?"

    Ultralytics actively maintains its models with frequent updates. While the YOLOX repository has seen reduced activity since 2022, YOLO11 benefits from continuous improvements, bug fixes, and compatibility updates with the latest versions of [PyTorch](https://pytorch.org/) and [ONNX](https://onnx.ai/).

## Summary

While **YOLOX** was an important stepping stone in the history of anchor-free detection, **Ultralytics YOLO11** represents the current pinnacle of vision AI efficiency.

For developers and researchers starting new projects in 2026, **YOLO11** is the recommended choice due to:

1.  **Higher Performance:** Better mAP with lower computational costs.
2.  **Broader Capabilities:** Support for detection, segmentation, classification, pose, and OBB.
3.  **Streamlined Workflow:** A robust, well-documented API that simplifies [training](https://docs.ultralytics.com/modes/train/) and [deployment](https://docs.ultralytics.com/guides/model-deployment-options/).

For those interested in the absolute latest innovations, Ultralytics has also released **[YOLO26](https://docs.ultralytics.com/models/yolo26/)**, which pushes the boundaries of speed and accuracy even further. However, YOLO11 remains a highly capable and supported option for all standard computer vision needs.

## Additional Resources

- **Ultralytics YOLO11 Docs:** [Official Documentation](https://docs.ultralytics.com/models/yolo11/)
- **YOLOX Paper:** [Arxiv:2107.08430](https://arxiv.org/abs/2107.08430)
- **Ultralytics GitHub:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Community:** Join the [Ultralytics Discord](https://discord.com/invite/ultralytics) to discuss model selection and implementation strategies.
