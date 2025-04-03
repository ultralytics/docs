---
comments: true
description: Explore a detailed technical comparison of YOLOv9 and YOLOv10, covering architecture, performance, and use cases. Find the best model for your needs.
keywords: YOLOv9, YOLOv10, object detection, Ultralytics, computer vision, model comparison, AI models, deep learning, efficiency, accuracy, real-time
---

# YOLOv9 vs YOLOv10: Detailed Technical Comparison

Ultralytics is committed to advancing the field of computer vision by developing and integrating state-of-the-art models. This page provides a detailed technical comparison between [YOLOv9](https://docs.ultralytics.com/models/yolov9/) and [YOLOv10](https://docs.ultralytics.com/models/yolov10/), two powerful object detection models. We will explore their architectural differences, performance benchmarks, and ideal use cases to help you choose the best model for your specific computer vision tasks.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv9", "YOLOv10"]'></canvas>

## YOLOv9: Programmable Gradient Information

[YOLOv9](https://docs.ultralytics.com/models/yolov9/) was introduced in February 2024 by Chien-Yao Wang and Hong-Yuan Mark Liao from the Institute of Information Science, Academia Sinica, Taiwan. It represents a significant step forward in efficient object detection by tackling information loss in deep neural networks. The core innovation is **Programmable Gradient Information (PGI)**, which helps preserve crucial data throughout the network layers. This is complemented by the **Generalized Efficient Layer Aggregation Network (GELAN)**, an optimized network architecture.

**Architecture and Key Features:**
YOLOv9's architecture leverages GELAN for efficient parameter utilization and PGI to ensure reliable gradient information for updates, leading to better model convergence and performance. This design aims to mitigate the information bottleneck problem common in deep networks, allowing the model to learn more effectively. The model details are outlined in the paper "[YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)".

**Performance Metrics:**
YOLOv9 demonstrates strong performance on the COCO dataset. For example, the YOLOv9c model achieves a mAP<sup>val</sup> 50-95 of 53.0% with 25.3 million parameters and 102.1 billion FLOPs. Its architecture is designed for a balance of high accuracy and efficiency.

**Strengths:**

- **High Accuracy:** Achieves state-of-the-art accuracy by effectively managing information flow.
- **Parameter Efficiency:** GELAN and PGI contribute to better performance with fewer parameters compared to some predecessors.
- **Novel Approach:** Introduces PGI to address fundamental information loss issues in deep learning.

**Weaknesses:**

- **Relatively New:** As a more recent model, the community support and range of deployment examples might be less extensive than for models like [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/).

**Use Cases:**
YOLOv9 is well-suited for tasks demanding high accuracy and efficiency:

- **Advanced Robotics:** Object detection in complex robotic systems requiring precise perception.
- **High-Resolution Image Analysis:** Scenarios needing detailed analysis where preserving information is key.
- **Resource-Constrained Environments:** Suitable for deployment on edge devices where computational power is limited, thanks to its efficiency.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

**Authors and Resources:**

- Authors: Chien-Yao Wang, Hong-Yuan Mark Liao
- Organization: Institute of Information Science, Academia Sinica, Taiwan
- Date: 2024-02-21
- Arxiv: [arXiv:2402.13616](https://arxiv.org/abs/2402.13616)
- GitHub: [github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)
- Docs: [docs.ultralytics.com/models/yolov9/](https://docs.ultralytics.com/models/yolov9/)

## YOLOv10: Holistic Efficiency-Accuracy Driven Design

[YOLOv10](https://docs.ultralytics.com/models/yolov10/), released in May 2024 by Ao Wang, Hui Chen, Lihao Liu, et al. from Tsinghua University, is engineered for real-time, end-to-end object detection. It prioritizes maximizing efficiency and inference speed by eliminating the need for Non-Maximum Suppression (NMS) post-processing.

**Architecture and Key Features:**
YOLOv10 introduces **Consistent Dual Assignments** for NMS-free training, which simplifies the deployment pipeline and reduces latency. It also employs a **Holistic Efficiency-Accuracy Driven Model Design**, optimizing various components:

- **Lightweight Classification Head:** Reduces computational cost.
- **Spatial-Channel Decoupled Downsampling:** Minimizes information loss during downsampling.
- **Rank-Guided Block Design:** Optimizes structure based on stage redundancy.
- **Accuracy Enhancements:** Incorporates large-kernel convolutions and Partial Self-Attention (PSA) to boost accuracy with minimal overhead.
  The model is detailed in the paper "[YOLOv10: Real-Time End-to-End Object Detection](https://arxiv.org/abs/2405.14458)".

**Performance Metrics:**
YOLOv10 sets new benchmarks for real-time object detection efficiency. YOLOv10-S is reported to be 1.8x faster than [RT-DETR](https://docs.ultralytics.com/models/rtdetr/)-R18 with comparable AP. Compared to YOLOv9-C, YOLOv10-B shows 46% less latency and 25% fewer parameters for similar performance. The smallest variant, YOLOv10-N, achieves a latency of just 1.84ms on an NVIDIA T4 GPU.

| Model    | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| -------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv9t  | 640                   | 38.3                 | -                              | 2.3                                 | **2.0**            | 7.7               |
| YOLOv9s  | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m  | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c  | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e  | 640                   | **55.6**             | -                              | 16.77                               | 57.3               | 189.0             |
|          |                       |                      |                                |                                     |                    |                   |
| YOLOv10n | 640                   | 39.5                 | -                              | **1.56**                            | 2.3                | **6.7**           |
| YOLOv10s | 640                   | 46.7                 | -                              | 2.66                                | 7.2                | 21.6              |
| YOLOv10m | 640                   | 51.3                 | -                              | 5.48                                | 15.4               | 59.1              |
| YOLOv10b | 640                   | 52.7                 | -                              | 6.54                                | 24.4               | 92.0              |
| YOLOv10l | 640                   | 53.3                 | -                              | 8.33                                | 29.5               | 120.3             |
| YOLOv10x | 640                   | 54.4                 | -                              | 12.2                                | 56.9               | 160.4             |

**Strengths:**

- **Extreme Efficiency:** Optimized for minimal latency and computational cost.
- **NMS-Free Training:** Enables true end-to-end deployment, reducing inference overhead.
- **High Speed:** Achieves significantly faster inference speeds.
- **Competitive Accuracy:** Maintains strong accuracy while prioritizing efficiency.

**Weaknesses:**

- **Very Recent Model:** As the newest model, community adoption and resources are still growing. Integration into existing workflows might require adjustments compared to more established models integrated within the Ultralytics ecosystem.

**Use Cases:**
YOLOv10 excels in applications where real-time performance and efficiency are critical:

- **Edge Computing:** Deployment on devices with limited resources like [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/).
- **Real-Time Video Analytics:** Applications needing immediate object detection in video streams, such as [traffic monitoring](https://www.ultralytics.com/blog/ai-in-traffic-management-from-congestion-to-coordination).
- **Mobile and Embedded Systems:** Integration into apps where speed and power consumption are crucial factors.

[Learn more about YOLOv10](https://docs.ultralytics.com/models/yolov10/){ .md-button }

**Authors and Resources:**

- Authors: Ao Wang, Hui Chen, Lihao Liu, et al.
- Organization: Tsinghua University
- Date: 2024-05-23
- Arxiv: [arXiv:2405.14458](https://arxiv.org/abs/2405.14458)
- GitHub: [github.com/THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)
- Docs: [docs.ultralytics.com/models/yolov10/](https://docs.ultralytics.com/models/yolov10/)

## Conclusion

Both YOLOv9 and YOLOv10 offer significant advancements in object detection. YOLOv9 focuses on improving accuracy and efficiency through novel techniques like PGI and GELAN, making it a strong choice for applications needing high precision. YOLOv10 prioritizes end-to-end real-time performance by eliminating NMS and optimizing the architecture for speed, ideal for latency-critical deployments. The choice between them depends on the specific requirements of your project regarding accuracy, speed, and computational resources.

Users might also be interested in exploring other models within the Ultralytics ecosystem, such as the versatile [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) or the latest [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/), which offer a balance of performance, ease of use, and support for multiple vision tasks.
