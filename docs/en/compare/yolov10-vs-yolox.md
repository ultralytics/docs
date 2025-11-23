---
comments: true
description: Compare YOLOv10 and YOLOX for object detection. Explore performance metrics, architecture, strengths, and ideal use cases for these top AI models.
keywords: YOLOv10, YOLOX, object detection, YOLO comparison, real-time AI models, Ultralytics, computer vision, model performance, anchor-free detection, AI benchmark
---

# YOLOv10 vs. YOLOX: A Technical Comparison

In the rapidly evolving landscape of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv), selecting the right object detection model is crucial for balancing performance, efficiency, and deployment ease. This technical comparison explores the differences between **YOLOv10**, the latest real-time end-to-end detector from Tsinghua University, and **YOLOX**, a highly regarded anchor-free model from Megvii.

While YOLOX introduced significant innovations in 2021 regarding anchor-free detection mechanisms, YOLOv10 represents the cutting edge of 2024, offering NMS-free inference and tighter integration with the [Ultralytics ecosystem](https://docs.ultralytics.com).

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv10", "YOLOX"]'></canvas>

## YOLOv10: Real-Time End-to-End Detection

YOLOv10 aims to bridge the gap between post-processing efficiency and model architecture. By introducing a consistent dual assignment strategy for NMS-free training, it eliminates the need for [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) during inference, significantly reducing latency.

**Technical Details:**

- **Authors:** Ao Wang, Hui Chen, Lihao Liu, et al.
- **Organization:** [Tsinghua University](https://www.tsinghua.edu.cn/en/)
- **Date:** 2024-05-23
- **Arxiv:** [arXiv:2405.14458](https://arxiv.org/abs/2405.14458)
- **GitHub:** [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)

### Architecture and Strengths

YOLOv10 builds upon the strengths of previous YOLO generations but optimizes the architecture for both efficiency and accuracy. It employs a holistic model design that includes lightweight classification heads and spatial-channel decoupled downsampling.

- **NMS-Free Inference:** The removal of NMS is a game-changer for [real-time inference](https://www.ultralytics.com/glossary/real-time-inference) applications, ensuring predictable latency and lower CPU overhead on edge devices.
- **Efficiency-Accuracy Balance:** YOLOv10 achieves state-of-the-art performance with lower parameter counts and FLOPs compared to its predecessors and competitors.
- **Ultralytics Integration:** Being fully supported by the `ultralytics` package means users benefit from a unified [Python API](https://docs.ultralytics.com/usage/python/), seamless export to formats like [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/) and [OpenVINO](https://docs.ultralytics.com/integrations/openvino/), and extensive documentation.

!!! tip "Ecosystem Advantage"

    YOLOv10's integration into the Ultralytics ecosystem provides immediate access to advanced features like [auto-annotation](https://www.ultralytics.com/glossary/data-annotation), [cloud training](https://docs.ultralytics.com/hub/cloud-training/), and a robust community for support.

### Weaknesses

- **Newer Architecture:** As a 2024 release, the ecosystem of third-party tutorials is growing rapidly but may not yet match the volume of older legacy models.

[Learn more about YOLOv10](https://docs.ultralytics.com/models/yolov10/){ .md-button }

## YOLOX: The Anchor-Free Pioneer

Released in 2021, YOLOX switched to an anchor-free mechanism and decoupled heads, diverging from the anchor-based approaches of YOLOv4 and YOLOv5. It utilizes SimOTA (Simplified Optimal Transport Assignment) for label assignment, which was a significant step forward in dynamic label assignment strategies.

**Technical Details:**

- **Authors:** Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun
- **Organization:** [Megvii](https://www.megvii.com/)
- **Date:** 2021-07-18
- **Arxiv:** [arXiv:2107.08430](https://arxiv.org/abs/2107.08430)
- **GitHub:** [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

### Architecture and Strengths

YOLOX remains a strong baseline in the research community due to its clean anchor-free design.

- **Anchor-Free Mechanism:** By removing predefined [anchor boxes](https://www.ultralytics.com/glossary/anchor-boxes), YOLOX reduces design complexity and the number of hyperparameters requiring tuning.
- **Decoupled Head:** Separating classification and localization tasks improved convergence speed and accuracy relative to older coupled-head designs.
- **Strong Baseline:** It serves as a reliable benchmark for academic research into detection heads and assignment strategies.

### Weaknesses

- **Inference Speed:** While efficient for its time, YOLOX generally trails behind newer models like YOLOv10 and [YOLO11](https://docs.ultralytics.com/models/yolo11/) in terms of raw inference speed, especially when NMS time is factored in.
- **Fragmented Workflow:** Unlike Ultralytics models, YOLOX often requires its own specific codebase and environment setup, lacking the unified interface for [training](https://docs.ultralytics.com/modes/train/), validation, and deployment found in modern frameworks.
- **Resource Intensity:** Higher FLOPs and parameter counts for similar accuracy levels compared to modern efficient architectures.

[Learn more about YOLOX](https://github.com/Megvii-BaseDetection/YOLOX){ .md-button }

## Performance Analysis

The comparison below highlights the significant advancements made in efficiency and accuracy over the three years separating these models. The metrics focus on model size (parameters), computational cost (FLOPs), and accuracy (mAP) on the COCO dataset.

| Model     | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| --------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv10n  | 640                   | 39.5                 | -                              | 1.56                                | 2.3                | 6.7               |
| YOLOv10s  | 640                   | 46.7                 | -                              | 2.66                                | 7.2                | 21.6              |
| YOLOv10m  | 640                   | 51.3                 | -                              | 5.48                                | 15.4               | 59.1              |
| YOLOv10b  | 640                   | 52.7                 | -                              | 6.54                                | 24.4               | 92.0              |
| YOLOv10l  | 640                   | 53.3                 | -                              | 8.33                                | 29.5               | 120.3             |
| YOLOv10x  | 640                   | 54.4                 | -                              | 12.2                                | 56.9               | 160.4             |
|           |                       |                      |                                |                                     |                    |                   |
| YOLOXnano | 416                   | 25.8                 | -                              | -                                   | 0.91               | 1.08              |
| YOLOXtiny | 416                   | 32.8                 | -                              | -                                   | 5.06               | 6.45              |
| YOLOXs    | 640                   | 40.5                 | -                              | 2.56                                | 9.0                | 26.8              |
| YOLOXm    | 640                   | 46.9                 | -                              | 5.43                                | 25.3               | 73.8              |
| YOLOXl    | 640                   | 49.7                 | -                              | 9.04                                | 54.2               | 155.6             |
| YOLOXx    | 640                   | 51.1                 | -                              | 16.1                                | 99.1               | 281.9             |

### Critical Observations

1. **Accuracy vs. Size:** YOLOv10 consistently delivers higher mAP with fewer parameters. For instance, **YOLOv10s** achieves **46.7 mAP** with only **7.2M** parameters, whereas **YOLOXs** achieves **40.5 mAP** with **9.0M** parameters. This demonstrates YOLOv10's superior architectural efficiency.
2. **Compute Efficiency:** The FLOPs count for YOLOv10 models is significantly lower. **YOLOv10x** operates at **160.4B FLOPs** compared to the massive **281.9B FLOPs** of **YOLOXx**, while still outperforming it in accuracy (54.4 vs 51.1 mAP).
3. **Inference Speed:** The removal of NMS and optimized architecture allows YOLOv10 to achieve lower latency. The T4 TensorRT benchmarks show YOLOv10x running at **12.2ms**, significantly faster than YOLOXx at **16.1ms**.

## Ideal Use Cases

### YOLOv10: The Modern Standard

YOLOv10 is the preferred choice for most new development projects, particularly those requiring:

- **Edge AI Deployment:** Its low memory footprint and high efficiency make it perfect for devices like the [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/) or NVIDIA Jetson.
- **Real-Time Applications:** Systems requiring immediate feedback, such as autonomous driving, robotics, and [video analytics](https://docs.ultralytics.com/guides/analytics/), benefit from the NMS-free low latency.
- **Rapid Development:** The Ultralytics ecosystem allows for quick [dataset management](https://docs.ultralytics.com/datasets/), training, and deployment via the `ultralytics` package.

### YOLOX: Legacy and Research

YOLOX remains relevant for:

- **Academic Research:** Researchers studying the evolution of anchor-free detectors or specific label assignment strategies like SimOTA often use YOLOX as a baseline.
- **Legacy Systems:** Existing production pipelines already optimized for YOLOX may continue to use it where upgrade costs outweigh performance gains.

## Using YOLOv10 with Ultralytics

One of the most significant advantages of YOLOv10 is its ease of use. The Ultralytics Python API simplifies the entire workflow, from loading pre-trained weights to training on custom data.

Below is an example of how to run predictions and train a YOLOv10 model:

```python
from ultralytics import YOLO

# Load a pre-trained YOLOv10n model
model = YOLO("yolov10n.pt")

# Run inference on an image
results = model.predict("path/to/image.jpg")

# Train the model on a custom dataset (COCO format)
model.train(data="coco8.yaml", epochs=100, imgsz=640)
```

!!! example "Training Efficiency"

    Ultralytics YOLO models are known for their [training efficiency](https://docs.ultralytics.com/modes/train/), often requiring less CUDA memory than older architectures or transformer-based models. This allows for training larger batches on standard consumer GPUs.

## Conclusion

While **YOLOX** played a pivotal role in popularizing anchor-free detection, **YOLOv10** represents the next leap forward in computer vision technology. With its NMS-free architecture, superior accuracy-to-computation ratio, and seamless integration into the robust Ultralytics ecosystem, YOLOv10 offers a compelling package for developers and researchers alike.

For those looking to deploy state-of-the-art [object detection](https://docs.ultralytics.com/tasks/detect/), YOLOv10 provides the necessary speed and precision. Developers interested in even broader capabilities, such as pose estimation or oriented bounding boxes, might also consider exploring the versatile [YOLO11](https://docs.ultralytics.com/models/yolo11/) or the widely adopted [YOLOv8](https://docs.ultralytics.com/models/yolov8/).
