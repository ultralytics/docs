---
comments: true
description: Explore a detailed technical comparison of YOLOv9 and YOLOv10, covering architecture, performance, and use cases. Find the best model for your needs.
keywords: YOLOv9, YOLOv10, object detection, Ultralytics, computer vision, model comparison, AI models, deep learning, efficiency, accuracy, real-time
---

# YOLOv9 vs YOLOv10: Detailed Technical Comparison

Ultralytics is committed to pushing the boundaries of computer vision, and a crucial part of this is developing and refining our YOLO models. This page offers a detailed technical comparison between [YOLOv9](https://docs.ultralytics.com/models/yolov9/) and [YOLOv10](https://docs.ultralytics.com/models/yolov10/), two state-of-the-art object detection models. We'll delve into their architectural nuances, performance benchmarks, and suitable applications to assist you in selecting the optimal model for your specific computer vision tasks.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv9", "YOLOv10"]'></canvas>

## YOLOv9: Programmable Gradient Information

[YOLOv9](https://docs.ultralytics.com/models/yolov9/), introduced in February 2024 by Chien-Yao Wang and Hong-Yuan Mark Liao from the Institute of Information Science, Academia Sinica, Taiwan, represents a significant advancement in efficient object detection. The core innovation of YOLOv9 lies in its **Programmable Gradient Information (PGI)**, designed to address information loss during the deep learning process. This is achieved through techniques like **Generalized Efficient Layer Aggregation Networks (GELAN)**, ensuring that the model learns exactly what you intend it to learn.

**Architecture and Key Features:**
YOLOv9 leverages GELAN to enhance feature extraction and maintain information integrity throughout the network. This approach leads to a model that is not only accurate but also parameter-efficient, making it suitable for deployments where computational resources are limited. YOLOv9 is implemented from the paper "[YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)".

**Performance Metrics:**
YOLOv9 demonstrates impressive performance on the COCO dataset. For example, YOLOv9c achieves a mAPval50-95 of 53.0% with 25.3M parameters and 102.1B FLOPs. The model's architecture is designed for efficiency, allowing it to achieve high accuracy with fewer parameters and computations compared to previous models.

**Strengths:**

- **High Accuracy:** Achieves state-of-the-art accuracy in object detection.
- **Parameter Efficiency:** Utilizes parameters and computations effectively due to GELAN and PGI.
- **Novel Approach:** Introduces Programmable Gradient Information for better learning.

**Weaknesses:**

- **Relatively New:** Being a more recent model, it might have a smaller community and fewer deployment examples compared to more established models.

**Use Cases:**
YOLOv9 is well-suited for applications requiring high accuracy and efficiency, such as:

- **Advanced Robotics:** Object detection in complex robotic systems.
- **High-Resolution Image Analysis:** Scenarios demanding detailed analysis of large images.
- **Resource-Constrained Environments:** Edge devices and mobile applications where computational power is limited.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

**Authors and Resources:**

- **Authors:** Chien-Yao Wang, Hong-Yuan Mark Liao
- **Organization:** Institute of Information Science, Academia Sinica, Taiwan
- **Date:** 2024-02-21
- **Arxiv:** [arXiv:2402.13616](https://arxiv.org/abs/2402.13616)
- **GitHub:** [github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)
- **Docs:** [docs.ultralytics.com/models/yolov9/](https://docs.ultralytics.com/models/yolov9/)

## YOLOv10: Holistic Efficiency-Accuracy Driven Design

[YOLOv10](https://docs.ultralytics.com/models/yolov10/), released in May 2024 by Ao Wang, Hui Chen, Lihao Liu, et al. from Tsinghua University, is engineered for real-time end-to-end object detection, emphasizing maximal efficiency and speed. YOLOv10 introduces several key methodological improvements to enhance both accuracy and efficiency, including **Consistent Dual Assignments** for NMS-free training and a **Holistic Efficiency-Accuracy Driven Model Design**.

**Architecture and Key Features:**
YOLOv10's architecture is meticulously designed to minimize computational redundancy and maximize performance. Key efficiency enhancements include a **Lightweight Classification Head**, **Spatial-Channel Decoupled Downsampling**, and **Rank-Guided Block Design**. Accuracy is boosted through **Large-Kernel Convolutions** and **Partial Self-Attention (PSA)**. These innovations allow YOLOv10 to achieve state-of-the-art speed and efficiency without sacrificing accuracy. YOLOv10 is detailed in the paper "[YOLOv10: Real-Time End-to-End Object Detection](https://arxiv.org/abs/2405.14458)".

**Performance Metrics:**
YOLOv10 sets new benchmarks in real-time object detection. For instance, YOLOv10-S is reported to be 1.8x faster than RT-DETR-R18 with comparable AP on the COCO dataset. YOLOv10-B demonstrates 46% less latency and 25% fewer parameters than YOLOv9-C while maintaining similar performance levels. YOLOv10-N achieves an impressive latency of just 1.84ms on a T4 GPU.

**Strengths:**

- **Extreme Efficiency:** Optimized for minimal latency and computational cost.
- **NMS-Free Training:** Consistent Dual Assignments enable end-to-end deployment without Non-Maximum Suppression, reducing inference time.
- **High Speed:** Achieves significantly faster inference speeds compared to previous YOLO versions and other models.
- **Good Accuracy:** Maintains competitive accuracy while prioritizing efficiency.

**Weaknesses:**

- **Very Recent Model:** As a very new model, it is still under active development and community support is growing.

**Use Cases:**
YOLOv10 is ideally suited for applications where real-time performance and efficiency are paramount:

- **Edge Computing:** Deployment on edge devices with limited resources.
- **Real-Time Video Analytics:** Applications requiring immediate object detection in video streams.
- **Mobile and Embedded Systems:** Integration into mobile apps and embedded systems where speed and power consumption are critical.

[Learn more about YOLOv10](https://docs.ultralytics.com/models/yolov10/){ .md-button }

**Authors and Resources:**

- **Authors:** Ao Wang, Hui Chen, Lihao Liu, et al.
- **Organization:** Tsinghua University
- **Date:** 2024-05-23
- **Arxiv:** [arXiv:2405.14458](https://arxiv.org/abs/2405.14458)
- **GitHub:** [github.com/THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)
- **Docs:** [docs.ultralytics.com/models/yolov10/](https://docs.ultralytics.com/models/yolov10/)

## Comparison Table

| Model    | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| -------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv9t  | 640                   | 38.3                 | -                              | 2.3                                 | 2.0                | 7.7               |
| YOLOv9s  | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m  | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c  | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e  | 640                   | 55.6                 | -                              | 16.77                               | 57.3               | 189.0             |
|          |                       |                      |                                |                                     |                    |                   |
| YOLOv10n | 640                   | 39.5                 | -                              | 1.56                                | 2.3                | 6.7               |
| YOLOv10s | 640                   | 46.7                 | -                              | 2.66                                | 7.2                | 21.6              |
| YOLOv10m | 640                   | 51.3                 | -                              | 5.48                                | 15.4               | 59.1              |
| YOLOv10b | 640                   | 52.7                 | -                              | 6.54                                | 24.4               | 92.0              |
| YOLOv10l | 640                   | 53.3                 | -                              | 8.33                                | 29.5               | 120.3             |
| YOLOv10x | 640                   | 54.4                 | -                              | 12.2                                | 56.9               | 160.4             |

## Conclusion

Both YOLOv9 and YOLOv10 represent cutting-edge advancements in object detection, each with unique strengths. YOLOv9 excels in accuracy and parameter efficiency through its Programmable Gradient Information, making it suitable for complex and detailed analysis. YOLOv10, on the other hand, prioritizes speed and real-time performance with its holistic efficiency-accuracy driven design and NMS-free training, making it ideal for edge and real-time applications.

For users seeking a balance of maturity and versatility, [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) remains a robust choice. For those interested in the latest advancements and highest accuracy, [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) also presents a compelling option, building upon the strengths of previous YOLO iterations. Ultimately, the best model depends on the specific requirements of your project, balancing accuracy, speed, and resource constraints.
