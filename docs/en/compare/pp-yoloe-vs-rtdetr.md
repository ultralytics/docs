---
comments: true
description: Explore a detailed comparison of PP-YOLOE+ and RTDETRv2 object detection models, analyzing performance, accuracy, and use cases to guide your decision.
keywords: PP-YOLOE+, RTDETRv2, object detection, model comparison, real-time detection, anchor-free detection, transformers, ultralytics, computer vision
---

# PP-YOLOE+ vs RTDETRv2: A Deep Dive into Modern Object Detection

The landscape of object detection is constantly evolving, with new architectures pushing the boundaries of speed and accuracy. In this technical comparison, we examine two significant models: **PP-YOLOE+**, a refined version of the YOLO-style detector by Baidu, and **RTDETRv2**, the latest iteration of the Real-Time Detection Transformer, also from Baidu. Understanding the nuances of these architectures helps developers choose the right tool for tasks ranging from [real-time surveillance](https://www.ultralytics.com/blog/real-time-security-monitoring-with-ai-and-ultralytics-yolo11) to autonomous driving.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["PP-YOLOE+", "RTDETRv2"]'></canvas>

## Model Overview

### PP-YOLOE+

**Authors:** PaddlePaddle Authors  
**Organization:** [Baidu](https://www.baidu.com/)  
**Date:** 2022-04-02  
**Arxiv:** [https://arxiv.org/abs/2203.16250](https://arxiv.org/abs/2203.16250)  
**GitHub:** [PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/)

PP-YOLOE+ represents a polished evolution of the YOLO family, specifically building upon PP-YOLOE. It is an anchor-free model that emphasizes a strong balance between inference speed and detection precision. By utilizing a powerful backbone and a refined training strategy, it targets edge and cloud deployments where traditional CNN-based efficiency is required.

[Learn more about PP-YOLOE+](https://docs.ultralytics.com/models/yoloe/){ .md-button }

### RTDETRv2

**Authors:** Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu  
**Organization:** [Baidu](https://www.baidu.com/)  
**Date:** 2023-04-17  
**Arxiv:** [https://arxiv.org/abs/2304.08069](https://arxiv.org/abs/2304.08069)  
**GitHub:** [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)

RTDETRv2 is a transformative step forward, literally. It brings the power of Vision Transformers (ViT) to real-time object detection. Unlike traditional CNN-based detectors that often require Non-Maximum Suppression (NMS) post-processing, RTDETRv2 (like its predecessor RT-DETR) is designed to be end-to-end. It leverages an efficient hybrid encoder to process multi-scale features, aiming to beat YOLO models in both speed and accuracy while eliminating NMS overhead.

[Learn more about RT-DETR](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

## Performance Comparison

Comparing performance metrics is crucial for selecting a model for production. The table below highlights the differences in Mean Average Precision (mAP), inference latency on T4 GPUs, and model complexity (Parameters and FLOPs).

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| PP-YOLOE+t | 640                   | 39.9                 | -                              | **2.84**                            | **4.85**           | 19.15             |
| PP-YOLOE+s | 640                   | 43.7                 | -                              | 2.62                                | 7.93               | **17.36**         |
| PP-YOLOE+m | 640                   | 49.8                 | -                              | 5.56                                | 23.43              | 49.91             |
| PP-YOLOE+l | 640                   | 52.9                 | -                              | 8.36                                | 52.2               | 110.07            |
| PP-YOLOE+x | 640                   | **54.7**             | -                              | 14.3                                | 98.42              | 206.59            |
|            |                       |                      |                                |                                     |                    |                   |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | 54.3                 | -                              | 15.03                               | 76                 | 259               |

### Analysis of Metrics

**Accuracy (mAP):**
RTDETRv2 generally exhibits superior accuracy for a given model scale compared to the older PP-YOLOE+. For example, **RTDETRv2-s** achieves an impressive **48.1% mAP**, significantly outperforming the PP-YOLOE+s at 43.7% mAP. This demonstrates the capability of the transformer architecture to capture global context better than pure CNNs, which is particularly helpful in complex scenes with [occlusions](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/sam-2.md).

**Speed and Latency:**
While transformers are traditionally slower, RTDETRv2 is optimized for real-time performance. However, PP-YOLOE+ still holds an edge in raw inference speed on T4 GPUs for smaller models (e.g., PP-YOLOE+s at roughly 2.6ms vs RTDETRv2-s at 5.03ms). This makes PP-YOLOE+ a strong contender for ultra-low latency applications, whereas RTDETRv2 offers a better trade-off if accuracy is the priority.

!!! tip "Latency vs. Throughput"

    When evaluating speed, consider whether your application is latency-sensitive (single image processing, e.g., autonomous braking) or throughput-sensitive (batch processing, e.g., analyzing hours of video footage). Transformers often require more memory but can be very efficient in batched scenarios.

## Architectural Differences

### PP-YOLOE+: The CNN Refinement

PP-YOLOE+ is built on a CSPResNet backbone and uses a Feature Pyramid Network (FPN) with a path aggregation network (PANet). It employs a Task Aligned Learning (TAL) strategy, which dynamically selects positive samples based on the alignment of classification and localization quality.

- **Backbone:** CSPResNet with varying depth and width.
- **Head:** Efficient Task-aligned Head (ET-Head).
- **Anchor-Free:** Reduces the number of hyper-parameters related to anchor boxes.
- **Post-processing:** Requires NMS to filter duplicate boxes.

### RTDETRv2: The Transformer Revolution

RTDETRv2 (and the broader RT-DETR family) fundamentally shifts the paradigm by introducing a hybrid encoder. It uses a CNN backbone (like ResNet or HGNetv2) to extract high-level features but processes them with a Transformer encoder to capture long-range dependencies.

- **Hybrid Encoder:** Decouples intra-scale interaction and cross-scale fusion to reduce computational cost.
- **Query Selection:** Uses IoU-aware query selection to initialize object queries, focusing on the most relevant parts of the image.
- **NMS-Free:** The transformer decoder outputs a fixed set of predictions, eliminating the need for NMS. This simplifies deployment pipelines and avoids the latency variability associated with NMS in dense scenes.

## Usability and Ecosystem

When choosing a model, the ease of integration and ecosystem support are as important as raw metrics.

### Ease of Use

The **Ultralytics** ecosystem provides first-class support for both styles of architectures, including RT-DETR and YOLO models. Developers can train, validate, and deploy these models using a unified API. This significantly lowers the barrier to entry compared to managing disparate repositories with different dependencies.

```python
from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("rtdetr-l.pt")

# Run inference on an image
results = model("https://ultralytics.com/images/bus.jpg")
```

### Versatility and Tasks

While PP-YOLOE+ is primarily focused on detection, modern Ultralytics models like [YOLO11](https://docs.ultralytics.com/models/yolo11/) and the upcoming **YOLO26** offer native support for a wider array of tasks, including [instance segmentation](https://docs.ultralytics.com/tasks/segment/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), and [Oriented Bounding Box (OBB)](https://docs.ultralytics.com/tasks/obb/) detection. RTDETRv2 is currently focused on detection, though its transformer nature makes it adaptable for segmentation in the future.

### Training Efficiency

One critical consideration is resource consumption. Transformer-based models like RTDETRv2 typically require more GPU memory during training compared to CNN-based models like PP-YOLOE+ or Ultralytics YOLO models. This higher VRAM requirement can be a bottleneck for researchers with limited hardware. Ultralytics YOLO models are renowned for their **training efficiency**, offering rapid convergence with lower memory footprints, making them accessible to a wider audience.

## Real-World Applications

### When to use PP-YOLOE+ (or YOLO11/YOLO26)

PP-YOLOE+ excels in scenarios where hardware resources are constrained, and every millisecond counts.

- **Embedded Devices:** Deploying on older edge hardware where transformer operations are not well-supported.
- **High-FPS Video:** Processing high-frame-rate video streams where minimal latency is required, such as [sports analytics](https://www.ultralytics.com/blog/application-and-impact-of-ai-in-basketball-and-nba).
- **Memory Constrained Environments:** Systems with limited RAM where the overhead of a transformer is prohibitive.

### When to use RTDETRv2

RTDETRv2 is ideal for applications demanding higher accuracy and robustness to complex visual patterns.

- **Crowded Scenes:** The global attention mechanism helps distinguish objects in dense crowds better than CNNs.
- **Simplified Pipelines:** Applications where removing the NMS step simplifies the post-processing logic, ensuring consistent inference times.
- **High-End GPU Deployment:** Utilizing powerful cloud GPUs where the slightly higher computational cost is offset by superior detection quality.

## Conclusion

Both PP-YOLOE+ and RTDETRv2 are formidable tools in the computer vision arsenal. PP-YOLOE+ represents the peak of efficiency for traditional CNN architectures, while RTDETRv2 paves the way for transformer-based real-time detection.

For most users, however, the seamless integration, active community support, and versatile task support of **Ultralytics** models (such as [YOLO11](https://docs.ultralytics.com/models/yolo11/) and the newly released **YOLO26**) offer the most balanced experience. With features like native **End-to-End NMS-Free Design** and optimized inference speeds, modern Ultralytics models bridge the gap, offering the simplicity of CNNs with the advanced capabilities often associated with transformers.

For further exploration, consider checking out other high-performance models in our documentation:

- [YOLO11](https://docs.ultralytics.com/models/yolo11/) - The versatile state-of-the-art predecessor.
- [YOLOv10](https://docs.ultralytics.com/models/yolov10/) - Pioneering NMS-free training.
- [FastSAM](https://docs.ultralytics.com/models/fast-sam/) - For real-time segment anything tasks.
