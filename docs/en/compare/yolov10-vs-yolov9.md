---
comments: true
description: Compare YOLOv10 and YOLOv9 object detection models. Explore architectures, metrics, and use cases to choose the best model for your application.
keywords: YOLOv10,YOLOv9,Ultralytics,object detection,real-time AI,computer vision,model comparison,AI deployment,deep learning
---

# YOLOv10 vs. YOLOv9: A Comprehensive Technical Comparison

The landscape of object detection has evolved rapidly, with successive iterations of the YOLO (You Only Look Once) architecture pushing the boundaries of speed and accuracy. Two of the most significant recent contributions to this field are **YOLOv10** and **YOLOv9**. While both models achieve state-of-the-art performance on the [COCO dataset](https://docs.ultralytics.com/datasets/detect/coco/), they diverge significantly in their design philosophies and architectural objectives.

YOLOv10 prioritizes low latency and end-to-end efficiency by eliminating the need for non-maximum suppression (NMS), whereas YOLOv9 focuses on maximizing information retention and accuracy through Programmable Gradient Information (PGI). This guide provides a detailed technical comparison to help developers and researchers select the optimal model for their [computer vision applications](https://www.ultralytics.com/blog/everything-you-need-to-know-about-computer-vision-in-2025).

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv10", "YOLOv9"]'></canvas>

## YOLOv10: The End-to-End Real-Time Detector

Released in May 2024 by researchers at Tsinghua University, [YOLOv10](https://docs.ultralytics.com/models/yolov10/) represents a paradigm shift in the YOLO lineage. Its primary innovation is the removal of the [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) post-processing step, which has traditionally been a bottleneck for inference latency.

**Technical Details:**

- **Authors:** Ao Wang, Hui Chen, Lihao Liu, et al.
- **Organization:** [Tsinghua University](https://www.tsinghua.edu.cn/en/)
- **Date:** 2024-05-23
- **Arxiv:** [Real-Time End-to-End Object Detection](https://arxiv.org/abs/2405.14458)
- **GitHub:** [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)

### Architecture and Key Innovations

YOLOv10 achieves its efficiency through a combination of **Consistent Dual Assignments** and a **Holistic Efficiency-Accuracy Driven Model Design**.

1.  **NMS-Free Training:** Traditional YOLO models rely on NMS to filter out duplicate bounding boxes. YOLOv10 utilizes a dual assignment strategy during [model training](https://docs.ultralytics.com/modes/train/). A one-to-many branch provides rich supervisory signals for learning, while a one-to-one branch ensures that the model generates a single best prediction per object during inference. This allows the model to be deployed without NMS, significantly reducing [inference latency](https://www.ultralytics.com/glossary/inference-latency).
2.  **Model Optimization:** The architecture includes lightweight classification heads, spatial-channel decoupled downsampling, and rank-guided block design. These features reduce computational redundancy and memory usage, making the model highly efficient on hardware with limited resources.

!!! tip "Efficiency Advantage"
    The removal of NMS in YOLOv10 is particularly beneficial for edge deployment. On devices where CPU resources are scarce, avoiding the computational cost of sorting and filtering thousands of candidate boxes can result in substantial speedups.

[Learn more about YOLOv10](https://docs.ultralytics.com/models/yolov10/){ .md-button }

## YOLOv9: Mastering Information Retention

Introduced in February 2024 by Chien-Yao Wang and Hong-Yuan Mark Liao, [YOLOv9](https://docs.ultralytics.com/models/yolov9/) targets the "information bottleneck" problem inherent in deep neural networks. As data passes through successive layers (feature extraction), crucial information can be lost, leading to degraded accuracy, especially for small or difficult-to-detect objects.

**Technical Details:**

- **Authors:** Chien-Yao Wang, Hong-Yuan Mark Liao
- **Organization:** [Institute of Information Science, Academia Sinica](https://www.iis.sinica.edu.tw/en/page/AboutUs/Introduction.html)
- **Date:** 2024-02-21
- **Arxiv:** [Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)
- **GitHub:** [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)

### Architecture and Key Innovations

YOLOv9 introduces novel concepts to ensure that the network retains and utilizes as much input information as possible.

1.  **Programmable Gradient Information (PGI):** PGI provides an auxiliary supervision framework that generates reliable gradients for updating network weights. This ensures that deep layers receive complete input information, mitigating the vanishing gradient problem and improving convergence.
2.  **Generalized Efficient Layer Aggregation Network (GELAN):** This new architecture replaces the conventional ELAN used in previous versions. GELAN optimizes parameter utilization and computational efficiency (FLOPs), allowing YOLOv9 to achieve higher accuracy with a model size comparable to its predecessors.

!!! note "Deep Learning Insight"
    YOLOv9's focus on information retention makes it exceptionally strong at detecting objects in complex scenes where feature details might otherwise be lost during downsampling operations in the [backbone](https://www.ultralytics.com/glossary/backbone).

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

## Performance Metrics: Speed vs. Accuracy

The choice between these two models often comes down to a trade-off between raw inference speed and detection precision. The table below highlights the performance differences across various model scales.

**Analysis:**

- **Latency:** YOLOv10 consistently outperforms YOLOv9 in latency, particularly in the smaller model sizes (N and S). For instance, **YOLOv10n** achieves an inference speed of **1.56 ms** on TensorRT, significantly faster than comparable models.
- **Accuracy:** YOLOv9 excels in accuracy at the higher end of the spectrum. The **YOLOv9e** model achieves a remarkable **55.6% mAP**, making it the superior choice for applications where precision is paramount.
- **Efficiency:** YOLOv10 offers excellent accuracy-per-parameter. **YOLOv10b** achieves 52.7% mAP with lower latency than **YOLOv9c**, demonstrating the effectiveness of its holistic design.

| Model    | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| -------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv10n | 640                   | 39.5                 | -                              | **1.56**                            | 2.3                | **6.7**           |
| YOLOv10s | 640                   | 46.7                 | -                              | 2.66                                | 7.2                | 21.6              |
| YOLOv10m | 640                   | 51.3                 | -                              | 5.48                                | 15.4               | 59.1              |
| YOLOv10b | 640                   | 52.7                 | -                              | 6.54                                | 24.4               | 92.0              |
| YOLOv10l | 640                   | 53.3                 | -                              | 8.33                                | 29.5               | 120.3             |
| YOLOv10x | 640                   | 54.4                 | -                              | 12.2                                | 56.9               | 160.4             |
|          |                       |                      |                                |                                     |                    |                   |
| YOLOv9t  | 640                   | 38.3                 | -                              | 2.3                                 | **2.0**            | 7.7               |
| YOLOv9s  | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m  | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c  | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e  | 640                   | **55.6**             | -                              | 16.77                               | 57.3               | 189.0             |

## Ideal Use Cases

Understanding the strengths of each model helps in selecting the right tool for your specific [project goals](https://docs.ultralytics.com/guides/defining-project-goals/).

### When to Choose YOLOv10

- **Edge AI Deployment:** Applications running on devices like [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/) or Raspberry Pi benefit from the NMS-free design, which reduces CPU overhead.
- **High-Frequency Video Analysis:** Scenarios requiring processing of high-FPS video streams, such as [traffic monitoring](https://www.ultralytics.com/blog/ai-in-traffic-management-from-congestion-to-coordination) or sports analytics.
- **Real-Time Robotics:** Autonomous systems that rely on low-latency feedback loops for navigation and [obstacle avoidance](https://www.ultralytics.com/blog/from-algorithms-to-automation-ais-role-in-robotics).

### When to Choose YOLOv9

- **High-Precision Inspection:** Industrial quality control where missing a defect (false negative) is costly.
- **Small Object Detection:** Applications involving [satellite imagery analysis](https://www.ultralytics.com/blog/using-computer-vision-to-analyze-satellite-imagery) or medical imaging where objects are small and feature-poor.
- **Complex Scenes:** Environments with high occlusion or clutter where maximum information retention is necessary to distinguish objects.

## Usage with Ultralytics

One of the significant advantages of using these models is their integration into the Ultralytics ecosystem. Both YOLOv10 and YOLOv9 can be utilized via the same unified Python API and Command Line Interface (CLI), simplifying the workflow from training to [deployment](https://docs.ultralytics.com/guides/model-deployment-options/).

### Python Example

The following code demonstrates how to load and run inference with both models using the `ultralytics` package.

```python
from ultralytics import YOLO

# Load a YOLOv10 model (NMS-free, high speed)
model_v10 = YOLO("yolov10n.pt")

# Load a YOLOv9 model (High accuracy)
model_v9 = YOLO("yolov9c.pt")

# Run inference on an image
# The API remains consistent regardless of the underlying architecture
results_v10 = model_v10("https://ultralytics.com/images/bus.jpg")
results_v9 = model_v9("https://ultralytics.com/images/bus.jpg")

# Print results
for r in results_v10:
    print(f"YOLOv10 Detections: {r.boxes.shape[0]}")

for r in results_v9:
    print(f"YOLOv9 Detections: {r.boxes.shape[0]}")
```

### The Ultralytics Advantage

Choosing Ultralytics for your computer vision projects offers several benefits beyond just model architecture:

- **Ease of Use:** The user-friendly API allows you to switch between YOLOv9, YOLOv10, and other models like [YOLO11](https://docs.ultralytics.com/models/yolo11/) by simply changing the weights file name.
- **Performance Balance:** Ultralytics implementations are optimized for real-world performance, balancing speed and accuracy.
- **Training Efficiency:** The framework supports features like [automatic mixed precision (AMP)](https://docs.ultralytics.com/modes/train/) and multi-GPU training, making it easier to train custom models on your own [datasets](https://docs.ultralytics.com/datasets/).
- **Memory Requirements:** Ultralytics models typically exhibit lower memory usage compared to transformer-based alternatives, facilitating training on consumer-grade GPUs.

## Conclusion

Both **YOLOv10** and **YOLOv9** represent significant milestones in object detection. **YOLOv10** is the clear winner for applications prioritizing speed and efficiency, thanks to its innovative NMS-free architecture. Conversely, **YOLOv9** remains a robust choice for scenarios demanding the highest possible accuracy and information retention.

For developers seeking the latest and most versatile solution, we also recommend exploring [**YOLO11**](https://docs.ultralytics.com/models/yolo11/). YOLO11 builds upon the strengths of these predecessors, offering a refined balance of speed, accuracy, and features for detection, segmentation, and pose estimation tasks.

### Explore Other Models

- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) - The latest state-of-the-art model.
- [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) - A versatile and mature model for various vision tasks.
- [RT-DETR](https://docs.ultralytics.com/models/rtdetr/) - A transformer-based detector for high-accuracy applications.
