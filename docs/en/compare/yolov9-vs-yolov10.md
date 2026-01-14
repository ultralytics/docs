---
comments: true
description: Explore a detailed technical comparison of YOLOv9 and YOLOv10, covering architecture, performance, and use cases. Find the best model for your needs.
keywords: YOLOv9, YOLOv10, object detection, Ultralytics, computer vision, model comparison, AI models, deep learning, efficiency, accuracy, real-time
---

# YOLOv9 vs YOLOv10: Architectural Evolution in Object Detection

The landscape of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) evolves rapidly, with significant breakthroughs occurring within months of each other. Two of the most impactful models released in 2024 were YOLOv9 and YOLOv10. While both aim to push the boundaries of real-time [object detection](https://docs.ultralytics.com/tasks/detect/), they approach the problem with distinct architectural philosophies.

YOLOv9 focuses on resolving information bottlenecks in deep networks using Programmable Gradient Information (PGI), whereas YOLOv10 introduces a paradigm shift by eliminating non-maximum suppression (NMS) for truly end-to-end inference. This guide provides a detailed technical comparison to help researchers and developers select the optimal architecture for their specific deployment needs, while also highlighting how the latest [YOLO26](https://docs.ultralytics.com/models/yolo26/) integrates the best of these worlds.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv9", "YOLOv10"]'></canvas>

| Model    | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| -------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv9t  | 640                   | 38.3                 | -                              | 2.3                                 | **2.0**            | 7.7               |
| YOLOv9s  | 640                   | **46.8**             | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m  | 640                   | **51.4**             | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c  | 640                   | **53.0**             | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e  | 640                   | **55.6**             | -                              | 16.77                               | 57.3               | 189.0             |
|          |                       |                      |                                |                                     |                    |                   |
| YOLOv10n | 640                   | **39.5**             | -                              | **1.56**                            | 2.3                | **6.7**           |
| YOLOv10s | 640                   | 46.7                 | -                              | **2.66**                            | 7.2                | **21.6**          |
| YOLOv10m | 640                   | 51.3                 | -                              | **5.48**                            | **15.4**           | **59.1**          |
| YOLOv10b | 640                   | 52.7                 | -                              | **6.54**                            | **24.4**           | **92.0**          |
| YOLOv10l | 640                   | 53.3                 | -                              | 8.33                                | 29.5               | 120.3             |
| YOLOv10x | 640                   | 54.4                 | -                              | **12.2**                            | 56.9               | **160.4**         |

## YOLOv9: Programmable Gradient Information

Released on February 21, 2024, YOLOv9 was developed by Chien-Yao Wang and Hong-Yuan Mark Liao at the [Institute of Information Science, Academia Sinica](https://www.iis.sinica.edu.tw/en/index.html). It builds upon the legacy of YOLOv7, aiming to solve the issue of information loss as data passes through deep neural networks.

### Architecture and Innovations

The core innovation of YOLOv9 is **Programmable Gradient Information (PGI)**. In deep networks, critical feature information often gets lost or diluted during the feed-forward process, a phenomenon known as the information bottleneck. PGI provides an auxiliary supervision framework that ensures gradients are reliably propagated back to update weights effectively, even in the deepest layers.

This is complemented by the **Generalized Efficient Layer Aggregation Network (GELAN)**. GELAN optimizes parameter utilization, allowing the model to achieve higher accuracy with fewer parameters compared to previous iterations. The architecture focuses heavily on computational efficiency during training, making it a robust choice for research environments where training resources are constrained.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

### Strengths and Use Cases

YOLOv9 excels in scenarios requiring high [precision](https://www.ultralytics.com/glossary/precision) and [recall](https://www.ultralytics.com/glossary/recall), particularly for small object detection where information retention is critical. Its architecture makes it highly effective for tasks such as [medical image analysis](https://www.ultralytics.com/glossary/medical-image-analysis) or detailed aerial surveillance.

!!! tip "Training Stability"

    The PGI auxiliary branch is primarily used during training to guide gradient flow. It can be removed during inference, meaning users get the benefit of better training supervision without paying a latency penalty at runtime.

## YOLOv10: The End-to-End Revolution

Introduced on May 23, 2024, by researchers from [Tsinghua University](https://www.tsinghua.edu.cn/en/), YOLOv10 represents a significant structural departure from traditional YOLO designs. It addresses one of the longest-standing bottlenecks in object detection: the reliance on Non-Maximum Suppression (NMS).

### Architecture and Innovations

The defining feature of YOLOv10 is its **Consistent Dual Assignments** strategy for NMS-free training. Traditional detectors predict multiple bounding boxes for a single object and use NMS post-processing to filter out duplicates. This step introduces latency and sensitivity to hyperparameters.

YOLOv10 eliminates this by employing a dual-head architecture during training:

1.  **One-to-Many Head:** Provides rich supervision signals (like standard YOLOs) to improve convergence.
2.  **One-to-One Head:** Matches exactly one prediction to one ground truth, mirroring the inference behavior.

By aligning these two heads, the model learns to output unique, high-quality predictions directly. During inference, only the one-to-one head is used, removing the need for NMS entirely. Furthermore, YOLOv10 incorporates **Holistic Efficiency-Accuracy Driven Model Design**, utilizing large-kernel convolutions and partial [self-attention](https://www.ultralytics.com/glossary/self-attention) (PSA) to boost performance with minimal computational cost.

[Learn more about YOLOv10](https://docs.ultralytics.com/models/yolov10/){ .md-button }

### Strengths and Use Cases

YOLOv10 is ideal for [edge computing](https://www.ultralytics.com/glossary/edge-computing) and real-time applications where every millisecond counts. The removal of NMS significantly reduces inference latency and engineering complexity, making it perfect for autonomous vehicles and high-speed industrial inspection.

## Why Choose Ultralytics Models?

While YOLOv9 and YOLOv10 offer impressive capabilities, the Ultralytics ecosystem provides a unified interface that makes deploying these advanced architectures simple and efficient. Ultralytics models, including [YOLO11](https://docs.ultralytics.com/models/yolo11/) and the cutting-edge YOLO26, are designed with the user experience in mind.

### The YOLO26 Advantage

For developers seeking the absolute state-of-the-art, **YOLO26** represents the pinnacle of this evolutionary path. Released in January 2026, it synthesizes the NMS-free breakthrough pioneered by YOLOv10 with advanced optimization techniques.

- **Natively End-to-End:** Like YOLOv10, YOLO26 is NMS-free, ensuring simplified deployment pipelines and faster inference.
- **MuSGD Optimizer:** Inspired by LLM training, YOLO26 utilizes a hybrid of SGD and Muon, resulting in more stable training runs.
- **Enhanced Efficiency:** With the removal of Distribution Focal Loss (DFL) and improved loss functions like ProgLoss, YOLO26 achieves up to **43% faster CPU inference** compared to predecessors, making it superior for edge devices.
- **Versatility:** Unlike v9 and v10 which are primarily detection-focused, YOLO26 natively supports [segmentation](https://docs.ultralytics.com/tasks/segment/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), [classification](https://docs.ultralytics.com/tasks/classify/), and [OBB](https://docs.ultralytics.com/tasks/obb/) tasks.

### Ease of Use and Ecosystem

Using the Ultralytics Python package, switching between these models is as simple as changing a filename. The framework handles complex tasks like data augmentation, [export](https://docs.ultralytics.com/modes/export/) to formats like [ONNX](https://docs.ultralytics.com/integrations/onnx/) or [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/), and visualization automatically.

```python
from ultralytics import YOLO

# Load a YOLOv9 model for high-precision tasks
model_v9 = YOLO("yolov9c.pt")
model_v9.train(data="coco8.yaml", epochs=100)

# Load a YOLOv10 model for ultra-low latency requirements
model_v10 = YOLO("yolov10s.pt")
results = model_v10("path/to/image.jpg")
```

The Ultralytics Platform further simplifies this workflow, offering tools for dataset management and cloud training that seamlessly integrate with these models. This robust support system ensures that whether you choose YOLOv9 for its architectural depth, YOLOv10 for its speed, or YOLO26 for its comprehensive performance, you have the tools to succeed.

!!! example "Memory Efficiency"

    Ultralytics implementations are renowned for their memory efficiency. While transformer-based models often require massive GPU memory, Ultralytics YOLO models are optimized to run on consumer-grade hardware and edge devices like the NVIDIA Jetson or Raspberry Pi, significantly lowering the barrier to entry for advanced AI projects.

## Conclusion

Both models represent significant achievements in the field. YOLOv9 pushes the limits of what convolutional networks can learn through PGI, offering excellent [accuracy](https://www.ultralytics.com/glossary/accuracy) for difficult detection tasks. YOLOv10 successfully challenges the NMS paradigm, offering a glimpse into the future of end-to-end real-time vision.

However, for most new projects in 2026, **YOLO26** is the recommended choice. It adopts the NMS-free design of v10 but refines it with the versatile, multi-task capabilities and ecosystem support that Ultralytics is known for. By choosing Ultralytics, you ensure your project is built on a foundation of continuous innovation, rigorous maintenance, and broad community support.
