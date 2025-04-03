---
comments: true
description: Compare YOLOX and YOLOv9 for object detection. Explore performance, architecture, and use cases to choose the best model for your vision tasks.
keywords: YOLOX, YOLOv9, object detection, model comparison, computer vision, AI models, deep learning, performance benchmarks, architecture, real-time detection
---

# Technical Comparison: YOLOX vs YOLOv9 for Object Detection

Selecting the right object detection model is critical for achieving optimal results in computer vision tasks. This page provides a detailed technical comparison between YOLOX and YOLOv9, two advanced models known for their performance and efficiency in object detection. We will explore their architectural differences, performance benchmarks, and suitability for various applications to help you make an informed decision.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOX", "YOLOv9"]'></canvas>

## YOLOX: High-Performance Anchor-Free Detector

YOLOX is an anchor-free object detection model developed by Megvii. Introduced in July 2021, YOLOX aims for simplicity and high performance by removing the anchor box concept, which simplifies the model and potentially improves generalization.

**Authors:** Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun  
**Organization:** Megvii  
**Date:** 2021-07-18  
**Arxiv Link:** [https://arxiv.org/abs/2107.08430](https://arxiv.org/abs/2107.08430)  
**GitHub Link:** [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)  
**Docs Link:** [https://yolox.readthedocs.io/en/latest/](https://yolox.readthedocs.io/en/latest/)

### Architecture and Key Features

YOLOX distinguishes itself with an [anchor-free detection](https://www.ultralytics.com/glossary/anchor-free-detectors) mechanism, simplifying the architecture. Key features include:

- **Decoupled Head:** Separates classification and localization heads for improved performance.
- **SimOTA Label Assignment:** An advanced label assignment strategy for optimized training.
- **Strong Data Augmentation:** Utilizes techniques like MixUp and Mosaic to enhance robustness and generalization, detailed further in guides on [data augmentation](https://www.ultralytics.com/glossary/data-augmentation).

### Strengths and Weaknesses

**Strengths:**

- **Anchor-Free Design:** Simplifies the model architecture, reducing design parameters and complexity.
- **High Accuracy and Speed:** Achieves a strong balance between mean Average Precision (mAP) and inference speed.
- **Scalability:** Offers a range of model sizes (Nano to X), allowing deployment across various computational resources.

**Weaknesses:**

- **Ecosystem:** While open-source, it lacks the integrated ecosystem and tooling provided by Ultralytics, such as seamless integration with [Ultralytics HUB](https://docs.ultralytics.com/hub/).
- **Inference Speed:** Larger YOLOX models can be slower than comparable optimized models like YOLOv9, especially on certain hardware.

### Ideal Use Cases

YOLOX is well-suited for applications needing a balance of high accuracy and speed, such as:

- **Real-time object detection** in [robotics](https://www.ultralytics.com/glossary/robotics) and surveillance systems.
- **Research and development** due to its modular design.
- **Edge AI** deployments, particularly the smaller Nano and Tiny variants.

[Learn more about YOLOX](https://yolox.readthedocs.io/en/latest/){ .md-button }

## YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information

[Ultralytics YOLOv9](https://docs.ultralytics.com/models/yolov9/) represents a significant advancement in object detection, addressing information loss challenges in deep neural networks through innovative architectural designs.

**Authors:** Chien-Yao Wang and Hong-Yuan Mark Liao  
**Organization:** Institute of Information Science, Academia Sinica, Taiwan  
**Date:** 2024-02-21  
**Arxiv Link:** [https://arxiv.org/abs/2402.13616](https://arxiv.org/abs/2402.13616)  
**GitHub Link:** [https://github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)  
**Docs Link:** [https://docs.ultralytics.com/models/yolov9/](https://docs.ultralytics.com/models/yolov9/)

### Architecture and Key Features

YOLOv9 introduces Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN) to maintain crucial data throughout the network, leading to improved accuracy and efficiency.

- **Programmable Gradient Information (PGI):** Helps mitigate information loss as data flows through deep networks.
- **Generalized Efficient Layer Aggregation Network (GELAN):** A highly efficient architecture optimized for speed and accuracy.

### Strengths and Weaknesses

**Strengths:**

- **State-of-the-Art Accuracy:** Achieves leading mAP scores by effectively preserving information.
- **Efficiency:** Offers excellent performance relative to its parameter count and FLOPs.
- **Ultralytics Ecosystem:** Benefits from integration into the Ultralytics ecosystem, providing **ease of use**, extensive [documentation](https://docs.ultralytics.com/), efficient training processes, readily available pre-trained weights, and strong community support. This makes it highly accessible for both developers and researchers.
- **Performance Balance:** Delivers a favorable trade-off between speed and accuracy, suitable for diverse real-world deployment scenarios.

**Weaknesses:**

- **Resource Demand:** Larger YOLOv9 models (like YOLOv9-E) still require substantial computational resources for training and inference.

### Ideal Use Cases

YOLOv9 excels in applications demanding the highest accuracy and efficiency:

- Complex object detection tasks in fields like [autonomous vehicles](https://www.ultralytics.com/solutions/ai-in-automotive).
- High-resolution [security and surveillance](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8) systems.
- Applications where minimizing information loss is critical for performance.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

## Performance Comparison: YOLOX vs YOLOv9

The table below provides a quantitative comparison of various YOLOX and YOLOv9 model variants based on their performance on the COCO dataset. YOLOv9 models generally demonstrate superior mAP compared to YOLOX models of similar size or computational cost, highlighting the effectiveness of PGI and GELAN. For instance, YOLOv9-C achieves a higher mAP (53.0) than YOLOX-L (49.7) with fewer parameters and FLOPs. YOLOv9-E sets a high benchmark at 55.6 mAP. While YOLOX offers very small models like Nano and Tiny, YOLOv9 provides a compelling balance of high accuracy and efficiency across its range, especially within the supportive Ultralytics framework.

| Model     | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| --------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOXnano | 416                   | 25.8                 | -                              | -                                   | **0.91**           | **1.08**          |
| YOLOXtiny | 416                   | 32.8                 | -                              | -                                   | 5.06               | 6.45              |
| YOLOXs    | 640                   | 40.5                 | -                              | 2.56                                | 9.0                | 26.8              |
| YOLOXm    | 640                   | 46.9                 | -                              | 5.43                                | 25.3               | 73.8              |
| YOLOXl    | 640                   | 49.7                 | -                              | 9.04                                | 54.2               | 155.6             |
| YOLOXx    | 640                   | 51.1                 | -                              | 16.1                                | 99.1               | 281.9             |
|           |                       |                      |                                |                                     |                    |                   |
| YOLOv9t   | 640                   | 38.3                 | -                              | **2.3**                             | 2.0                | 7.7               |
| YOLOv9s   | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m   | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c   | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e   | 640                   | **55.6**             | -                              | 16.77                               | 57.3               | 189.0             |

## Conclusion

Both YOLOX and YOLOv9 are powerful object detection models. YOLOX offers a simplified anchor-free design and a range of model sizes suitable for various applications. However, YOLOv9, particularly when utilized within the Ultralytics ecosystem, provides state-of-the-art accuracy and efficiency by addressing fundamental information loss problems in deep networks. Its integration with Ultralytics tools offers significant advantages in terms of ease of use, training efficiency, and deployment flexibility.

For users prioritizing the highest accuracy and leveraging a well-supported ecosystem, YOLOv9 is the recommended choice. YOLOX remains a viable option, especially for those already invested in the Megvii ecosystem or requiring extremely lightweight models like YOLOX-Nano.

Explore other cutting-edge models in the Ultralytics documentation, such as [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/), [YOLOv10](https://docs.ultralytics.com/models/yolov10/), and [YOLO11](https://docs.ultralytics.com/models/yolo11/). Further comparisons like [YOLOv9 vs YOLOv8](https://docs.ultralytics.com/compare/yolov9-vs-yolov8/) and [YOLOv8 vs YOLOX](https://docs.ultralytics.com/compare/yolov8-vs-yolox/) can also provide additional insights.
