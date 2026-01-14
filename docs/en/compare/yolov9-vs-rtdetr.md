---
comments: true
description: Compare YOLOv9 and RTDETRv2 for object detection. Explore speed, accuracy, use cases, and architectures to choose the best for your project.
keywords: YOLOv9, RTDETRv2, object detection, model comparison, AI models, computer vision, YOLO, real-time detection, transformers, efficiency
---

# YOLOv9 vs. RTDETRv2: Architectures, Performance, and Applications

In the rapidly evolving landscape of real-time object detection, selecting the right model architecture is crucial for optimizing performance and resource utilization. This detailed comparison explores **YOLOv9**, a CNN-based powerhouse, and **RTDETRv2**, a state-of-the-art transformer-based detector. By analyzing their architectural innovations, training efficiencies, and deployment capabilities, developers can make informed decisions for their computer vision applications.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv9", "RTDETRv2"]'></canvas>

## Model Overview

### YOLOv9: Programmable Gradient Information

YOLOv9, released in early 2024, represents a significant leap in the YOLO (You Only Look Once) family. Developed by the Institute of Information Science at Academia Sinica, it addresses the fundamental challenge of information loss in deep neural networks.

The core innovation of YOLOv9 lies in **Programmable Gradient Information (PGI)** and the **Generalized Efficient Layer Aggregation Network (GELAN)**. PGI prevents the loss of crucial semantic information during the training process of deep networks, while GELAN maximizes parameter utilization. This combination allows YOLOv9 to achieve top-tier [accuracy](https://www.ultralytics.com/glossary/accuracy) with fewer parameters compared to its predecessors.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

### RTDETRv2: Real-Time Transformers

RTDETRv2 builds upon the success of the original Real-Time Detection Transformer (RT-DETR) by Baidu. While traditional transformers like [DETR](https://arxiv.org/abs/2005.12872) offered high accuracy, they suffered from slow convergence and high computational costs. RTDETRv2 optimizes this by introducing a hybrid encoder and an IoU-aware query selection mechanism.

Distinct from CNN-based approaches, RTDETRv2 eliminates the need for [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) post-processing, offering an end-to-end detection pipeline. This makes it particularly attractive for scenarios where varying post-processing latencies can disrupt real-time pipelines.

[Learn more about RTDETR](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

## Performance Analysis

The following table provides a direct comparison of key metrics. YOLOv9 generally offers superior speed on CPU-based edge devices, while RTDETRv2 shines in complex scenes requiring global context, often at the cost of higher memory usage.

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv9t    | 640                   | 38.3                 | -                              | 2.3                                 | **2.0**            | **7.7**           |
| YOLOv9s    | 640                   | 46.8                 | -                              | **3.54**                            | 7.1                | 26.4              |
| YOLOv9m    | 640                   | 51.4                 | -                              | **6.43**                            | 20.0               | 76.3              |
| YOLOv9c    | 640                   | 53.0                 | -                              | **7.16**                            | 25.3               | 102.1             |
| YOLOv9e    | 640                   | **55.6**             | -                              | 16.77                               | 57.3               | 189.0             |
|            |                       |                      |                                |                                     |                    |                   |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | 54.3                 | -                              | 15.03                               | 76                 | 259               |

!!! note "Performance Trade-offs"

    While RTDETRv2-x achieves competitive accuracy, notice the significant jump in FLOPs (259B) compared to YOLOv9e (189.0B). This difference highlights the computational intensity of transformer attention mechanisms versus efficient CNN convolutions.

## Architectural Deep Dive

### YOLOv9: Efficiency through GELAN

YOLOv9's architecture is centered around **GELAN**, which allows for flexible stacking of convolutional blocks (like ResBlocks or CSP blocks) without sacrificing speed. This is crucial for deployment on limited hardware. The [backbone](https://www.ultralytics.com/glossary/backbone) efficiently extracts features, while the PGI auxiliary branch ensures that the gradient updates during training are informative, even for the deepest layers.

Key features include:

- **Auxiliary Reversible Branch:** Maintains complete information flow during training.
- **Multi-Level Auxiliary Information:** Aggregates gradients from different prediction heads to boost convergence.
- **Lower Memory Footprint:** Generally requires less CUDA memory during training compared to transformer architectures.

### RTDETRv2: The Power of Attention

RTDETRv2 employs a hybrid encoder that processes multi-scale features. Unlike pure CNNs that rely on local receptive fields, the transformer components in RTDETRv2 utilize self-attention to understand global context. This is particularly effective for detecting objects that are occluded or spatially separated but contextually related.

Key features include:

- **IoU-Aware Query Selection:** Prioritizes high-quality features for the decoder, improving initialization.
- **NMS-Free:** The model predicts a set of bounding boxes directly, removing the need for heuristic post-processing steps like NMS.
- **Adaptable Inference:** Inference speed can be adjusted by changing the number of decoder layers without retraining.

## Integration with Ultralytics

Both models are fully integrated into the Ultralytics ecosystem, providing a unified API for [training](https://docs.ultralytics.com/modes/train/), validation, and deployment. This integration ensures that developers can easily switch between architectures to benchmark performance on their specific [datasets](https://docs.ultralytics.com/datasets/).

### Ease of Use and Ecosystem

The Ultralytics Python package abstracts the complexities of these differing architectures. Whether you are using the CNN-based YOLOv9 or the transformer-based RTDETRv2, the workflow remains consistent.

```python
from ultralytics import RTDETR, YOLO

# Load a YOLOv9 model
model_yolo = YOLO("yolov9c.pt")
results_yolo = model_yolo.train(data="coco8.yaml", epochs=100)

# Load an RTDETRv2 model (accessed via the RTDETR class)
model_rtdetr = RTDETR("rtdetr-l.pt")
results_rtdetr = model_rtdetr.train(data="coco8.yaml", epochs=100)
```

### Deployment and Export

A significant advantage of the Ultralytics ecosystem is the seamless [export capability](https://docs.ultralytics.com/modes/export/). Both models can be exported to formats like [ONNX](https://docs.ultralytics.com/integrations/onnx/), TensorRT, and CoreML. However, transformer models like RTDETRv2 may require more careful tuning for export to certain edge accelerators compared to the straightforward convolution operations of YOLOv9.

## Use Case Recommendations

Choosing between YOLOv9 and RTDETRv2 depends heavily on the specific constraints of your project.

### When to Choose YOLOv9

- **Edge Deployment:** If you are deploying on Raspberry Pi, mobile devices, or older GPUs, YOLOv9's lower FLOPs and parameter count (specifically the `t` and `s` variants) make it the superior choice.
- **Training Resources:** YOLOv9 generally consumes less VRAM during training, allowing for larger [batch sizes](https://www.ultralytics.com/glossary/batch-size) on consumer-grade hardware.
- **Versatility:** Beyond detection, the YOLO family often extends more naturally to tasks like [segmentation](https://docs.ultralytics.com/tasks/segment/) and pose estimation within the same architectural framework.

### When to Choose RTDETRv2

- **Crowded Scenes:** The global attention mechanism helps in dense environments where objects overlap significantly.
- **NMS-Free Requirements:** In industrial pipelines where deterministic latency is critical, eliminating NMS removes a variable time component.
- **High-End GPU Availability:** If you have access to powerful [GPUs](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit) (like NVIDIA T4, A100) and prioritize accuracy over raw inference speed, RTDETRv2 is a strong contender.

## Conclusion

Both YOLOv9 and RTDETRv2 represent the pinnacle of their respective architectural lineages. YOLOv9 offers an exceptional balance of speed and accuracy, making it a versatile choice for a wide range of real-time applications, particularly those constrained by hardware. RTDETRv2 provides a robust alternative for scenarios requiring the global context awareness of transformers and NMS-free output.

For developers looking for the absolute latest in performance and ease of use, we also recommend exploring **YOLO26**. Released in January 2026, YOLO26 incorporates an end-to-end NMS-free design natively within the YOLO architecture, combining the best of both worlds—transformer-like detection logic with CNN efficiency.

[Learn more about YOLO26](https://docs.ultralytics.com/models/yolo26/){ .md-button }

By leveraging the Ultralytics platform, users can experiment with these models side-by-side, ensuring the selected solution perfectly aligns with their operational requirements.

## Citations

**YOLOv9:**
Authors: Chien-Yao Wang, Hong-Yuan Mark Liao
Organization: Institute of Information Science, Academia Sinica, Taiwan
Date: 2024-02-21
[Arxiv Link](https://arxiv.org/abs/2402.13616)

**RTDETRv2:**
Authors: Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, Yi Liu
Organization: Baidu
Date: 2023-04-17 (v1), 2024-07 (v2)
[Arxiv Link](https://arxiv.org/abs/2407.17140)
