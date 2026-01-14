---
comments: true
description: Compare YOLOv5 and RTDETRv2 for object detection. Explore their architectures, performance metrics, strengths, and best use cases in computer vision.
keywords: YOLOv5, RTDETRv2, object detection, model comparison, Ultralytics, computer vision, machine learning, real-time detection, Vision Transformers, AI models
---

# YOLOv5 vs RTDETRv2: A Technical Comparison of Real-Time Detection

The landscape of [object detection](https://www.ultralytics.com/glossary/object-detection) has evolved rapidly, shifting from purely CNN-based architectures to modern Transformer hybrids. Two significant milestones in this evolution are **Ultralytics YOLOv5** and **Baidu's RTDETRv2**. While YOLOv5 established the industry standard for speed and ease of deployment, RTDETRv2 introduces the power of Vision Transformers (ViT) to real-time applications.

This analysis explores their architectural differences, performance metrics, and ideal use cases to help developers choose the right tool for their [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) projects.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv5", "RTDETRv2"]'></canvas>

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv5n    | 640                   | 28.0                 | 73.6                           | 1.12                                | 2.6                | 7.7               |
| YOLOv5s    | 640                   | 37.4                 | 120.7                          | 1.92                                | 9.1                | 24.0              |
| YOLOv5m    | 640                   | 45.4                 | 233.9                          | 4.03                                | 25.1               | 64.2              |
| YOLOv5l    | 640                   | 49.0                 | 408.4                          | 6.61                                | 53.2               | 135.0             |
| YOLOv5x    | 640                   | 50.7                 | 763.2                          | 11.89                               | 97.2               | 246.4             |
|            |                       |                      |                                |                                     |                    |                   |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | 54.3                 | -                              | 15.03                               | 76                 | 259               |

## Ultralytics YOLOv5: The Industry Standard

YOLOv5 is arguably the most famous iteration of the YOLO family, renowned for its balance of engineering utility and performance. Built on PyTorch, it made advanced computer vision accessible to millions of developers through a focus on usability and deployment flexibility.

**Authors:** Glenn Jocher  
**Organization:** [Ultralytics](https://www.ultralytics.com)  
**Date:** 2020-06-26  
**GitHub:** [ultralytics/yolov5](https://github.com/ultralytics/yolov5)  
**Docs:** [YOLOv5 Documentation](https://docs.ultralytics.com/models/yolov5/)

### Architecture and Strengths

YOLOv5 utilizes a [Convolutional Neural Network (CNN)](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn) architecture. It employs a modified CSPDarknet backbone and a PA-NET neck to efficiently extract and aggregate features at multiple scales.

- **Deployment Versatility:** YOLOv5 is designed to run anywhere. It supports instant export to formats like [ONNX](https://www.ultralytics.com/glossary/onnx-open-neural-network-exchange), TensorRT, CoreML, and TFLite, making it the go-to choice for mobile and edge device deployment.
- **Low Memory Footprint:** Compared to transformer-based models, YOLOv5 requires significantly less [GPU](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit) memory during training and inference, allowing it to run on smaller hardware instances.
- **Training Efficiency:** The model incorporates "Bag of Freebies" enhancements such as mosaic data augmentation and hyperparameter evolution, which stabilize training and boost accuracy without increasing inference cost.

[Learn more about YOLOv5](https://docs.ultralytics.com/models/yolov5/){ .md-button }

## Baidu RTDETRv2: The Transformer Challenger

RTDETRv2 (Real-Time Detection Transformer version 2) represents a push to bring the accuracy of [Transformers](https://www.ultralytics.com/glossary/transformer) to real-time speeds. It builds upon the original RT-DETR to address the high computational cost usually associated with DETR-like models.

**Authors:** Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu  
**Organization:** Baidu  
**Date:** 2023-04-17 (v1), 2024-07 (v2)  
**Arxiv:** [RT-DETRv2 Paper](https://arxiv.org/abs/2407.17140)  
**GitHub:** [RT-DETR Repository](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)

### Architecture and Innovation

RTDETRv2 employs a hybrid encoder that processes multi-scale features, decoupling intra-scale interaction from cross-scale fusion. This design enables the model to capture global context more effectively than pure CNNs.

- **NMS-Free Prediction:** Unlike YOLOv5, which relies on [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) to filter overlapping bounding boxes, RTDETRv2 predicts the final set of objects directly. This simplifies post-processing pipelines.
- **Adaptable Speed:** The architecture allows for adjusting the number of decoder layers to trade off speed for accuracy without retraining, offering flexibility for different hardware constraints.
- **IoU-Aware Query Selection:** It improves how object queries are initialized, focusing the model's attention on the most relevant parts of the image early in the process.

[Learn more about RT-DETR](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

## Detailed Comparison

The choice between YOLOv5 and RTDETRv2 often depends on the specific constraints of the deployment environment and the nature of the visual data.

### 1. Accuracy vs. Speed Trade-off

RTDETRv2 generally achieves higher [Mean Average Precision (mAP)](https://www.ultralytics.com/glossary/mean-average-precision-map) on the COCO dataset compared to YOLOv5, particularly for medium and large objects. The transformer architecture excels at understanding global context, which helps in complex scenes with occlusion.

However, **YOLOv5** remains superior in pure raw inference speed on CPU and edge devices. The overhead of attention mechanisms in Transformers makes RTDETRv2 heavier to run on non-GPU hardware. For applications requiring ultra-low latency on limited hardware (like Raspberry Pi or mobile phones), YOLOv5's lightweight CNN architecture is often preferred.

### 2. Ecosystem and Ease of Use

This is where Ultralytics models shine. The **YOLOv5 ecosystem** is incredibly mature.

- **Documentation:** Extensive, beginner-friendly [docs](https://docs.ultralytics.com) cover everything from training on custom datasets to multi-GPU training.
- **Integration:** Seamless integration with tools like [Ultralytics HUB](https://www.ultralytics.com/hub), MLflow, and TensorBoard makes lifecycle management effortless.
- **Community:** A massive global community ensures that bugs are found and fixed quickly, and support is readily available.

In contrast, while RTDETRv2 is technically impressive, it lacks the same level of integrated tooling and ease of use found in the Ultralytics Python package.

### 3. Training and Resource Efficiency

Training transformer models like RTDETRv2 typically requires more memory and longer training times to converge compared to CNNs like YOLOv5. YOLOv5's efficient architecture allows for rapid iteration, enabling developers to train and fine-tune models quickly, even on single GPUs.

!!! tip "Memory Considerations"

    If you are training on consumer-grade GPUs with limited VRAM (e.g., 8GB or less), **YOLOv5** or the newer **YOLO11** are recommended. Transformer-based architectures like RTDETRv2 utilize `O(n^2)` attention mechanisms that can quickly exhaust GPU memory at higher resolutions.

## Code Example: Using Ultralytics Models

The [Ultralytics Python API](https://docs.ultralytics.com/usage/python/) provides a unified interface for working with these models. Below is an example of how to load and use a model for prediction.

```python
from ultralytics import RTDETR, YOLO

# Load a YOLOv5 model (using the 'u' suffix for the modern anchor-free version)
model_yolo = YOLO("yolov5su.pt")

# Run inference on an image
results_yolo = model_yolo("https://ultralytics.com/images/bus.jpg")

# Load an RT-DETR model (Ultralytics supports the RT-DETR architecture)
model_rtdetr = RTDETR("rtdetr-l.pt")

# Run inference with RT-DETR
results_rtdetr = model_rtdetr("https://ultralytics.com/images/bus.jpg")

# Print results
print(f"YOLOv5 Detection Count: {len(results_yolo[0].boxes)}")
print(f"RT-DETR Detection Count: {len(results_rtdetr[0].boxes)}")
```

## Conclusion: Which Should You Choose?

- **Choose YOLOv5 if:** You need a battle-tested, reliable model that runs fast on any hardware (CPU, Edge, Mobile). It is the ideal choice for developers who prioritize ease of use, fast training times, and broad compatibility with deployment targets.
- **Choose RTDETRv2 if:** You have powerful GPU hardware available and your primary goal is maximizing accuracy in complex scenes with occlusion, where the NMS-free design provides a distinct advantage.

### The Best of Both Worlds: YOLO26

For developers seeking the accuracy of transformers with the speed of CNNs, the newly released **[YOLO26](https://docs.ultralytics.com/models/yolo26/)** offers a compelling middle ground.

- **End-to-End NMS-Free:** Like RTDETRv2, YOLO26 eliminates NMS, streamlining deployment.
- **Performance:** It utilizes the MuSGD optimizer and optimized loss functions (ProgLoss) to achieve state-of-the-art accuracy.
- **Speed:** Up to 43% faster CPU inference than previous generations, retaining the efficiency Ultralytics is known for.

By combining the architectural innovations of NMS-free detection with the efficiency of CNNs, YOLO26 effectively bridges the gap between the legacy reliability of YOLOv5 and the modern accuracy of RTDETRv2.

[Learn more about YOLO26](https://docs.ultralytics.com/models/yolo26/){ .md-button }
