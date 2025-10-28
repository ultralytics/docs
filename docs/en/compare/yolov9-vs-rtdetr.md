---
comments: true
description: Compare YOLOv9 and RTDETRv2 for object detection. Explore speed, accuracy, use cases, and architectures to choose the best for your project.
keywords: YOLOv9, RTDETRv2, object detection, model comparison, AI models, computer vision, YOLO, real-time detection, transformers, efficiency
---

# YOLOv9 vs. RTDETRv2: A Technical Comparison for Object Detection

Choosing the optimal object detection model is a critical decision for any computer vision project, requiring a careful balance between accuracy, inference speed, and computational cost. This page provides a detailed technical comparison between two powerful models: [YOLOv9](https://docs.ultralytics.com/models/yolov9/), a state-of-the-art model known for its efficiency and accuracy, and RTDETRv2, a transformer-based model praised for its high precision. This analysis will help you determine which model best suits your project's specific requirements.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv9", "RTDETRv2"]'></canvas>

## YOLOv9: Advancing Real-Time Detection with Efficiency

[YOLOv9](https://docs.ultralytics.com/models/yolov9/) is a significant leap forward in the YOLO series, introducing groundbreaking techniques to enhance performance and efficiency. Developed by leading researchers, it addresses key challenges in deep learning to deliver superior results.

- **Authors:** Chien-Yao Wang, Hong-Yuan Mark Liao
- **Organization:** [Institute of Information Science, Academia Sinica, Taiwan](https://www.iis.sinica.edu.tw/en/index.html)
- **Date:** 2024-02-21
- **Arxiv:** <https://arxiv.org/abs/2402.13616>
- **GitHub:** <https://github.com/WongKinYiu/yolov9>
- **Docs:** <https://docs.ultralytics.com/models/yolov9/>

### Architecture and Key Features

YOLOv9's architecture introduces two major innovations: **Programmable Gradient Information (PGI)** and the **Generalized Efficient Layer Aggregation Network (GELAN)**. PGI is designed to combat the problem of information loss as data flows through deep neural networks, ensuring that the model receives reliable gradient information for accurate updates. GELAN is a novel network architecture that optimizes parameter utilization and computational efficiency, allowing YOLOv9 to achieve high accuracy without a massive number of parameters.

When integrated into the Ultralytics ecosystem, YOLOv9's power is amplified. Developers benefit from a **streamlined user experience** with a simple [Python API](https://docs.ultralytics.com/usage/python/) and extensive [documentation](https://docs.ultralytics.com/models/yolov9/). This ecosystem ensures **efficient training** with readily available pre-trained weights and benefits from active development and strong community support.

### Strengths and Weaknesses

**Strengths:**

- **State-of-the-Art Accuracy:** Achieves leading mAP scores on benchmarks like [COCO](https://docs.ultralytics.com/datasets/detect/coco/), often outperforming models with more parameters.
- **High Efficiency:** GELAN and PGI deliver exceptional performance with fewer parameters and FLOPs, making it ideal for deployment on [edge AI](https://www.ultralytics.com/glossary/edge-ai) devices.
- **Information Preservation:** PGI effectively mitigates information loss, leading to more robust learning and better feature representation.
- **Well-Maintained Ecosystem:** Benefits from active development, comprehensive resources, [Ultralytics HUB](https://hub.ultralytics.com/) integration for MLOps, and strong community support.
- **Lower Memory Requirements:** Compared to transformer-based models, YOLOv9 typically requires significantly less memory during training and inference, making it more accessible for users with limited hardware.
- **Versatility:** While the original paper focuses on [object detection](https://docs.ultralytics.com/tasks/detect/), the architecture supports multiple tasks like [instance segmentation](https://docs.ultralytics.com/tasks/segment/), aligning with the multi-task capabilities of other Ultralytics models like [YOLOv8](https://docs.ultralytics.com/models/yolov8/).

**Weaknesses:**

- **Novelty:** As a newer model, the number of community-driven deployment examples may be smaller than for long-established models, though its integration within Ultralytics accelerates adoption rapidly.

### Ideal Use Cases

YOLOv9 is ideally suited for applications where both high accuracy and real-time efficiency are paramount:

- **Autonomous Systems:** Perfect for [autonomous vehicles](https://www.ultralytics.com/solutions/ai-in-automotive) and drones that require fast and accurate perception.
- **Advanced Security:** Powers sophisticated [security systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8) with real-time threat detection.
- **Industrial Automation:** Excellent for [quality control in manufacturing](https://www.ultralytics.com/blog/improving-manufacturing-with-computer-vision) and complex [robotic tasks](https://www.ultralytics.com/glossary/robotics).
- **Edge Computing:** Its efficient design makes it suitable for deployment in resource-constrained environments.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

## RTDETRv2: Precision-Focused Real-Time Detection

**RTDETRv2** ([Real-Time Detection Transformer v2](https://docs.ultralytics.com/models/rtdetr/)) is a model designed for applications demanding high accuracy in real-time object detection, leveraging the power of transformer architectures.

- **Authors:** Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu
- **Organization:** [Baidu](https://www.baidu.com/)
- **Date:** 2023-04-17 (Original RT-DETR), 2024-07-24 (RTDETRv2 paper)
- **Arxiv:** <https://arxiv.org/abs/2304.08069> (Original), <https://arxiv.org/abs/2407.17140> (v2)
- **GitHub:** <https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch>
- **Docs:** <https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme>

### Architecture and Key Features

RTDETRv2's architecture is built upon [Vision Transformers (ViT)](https://www.ultralytics.com/glossary/vision-transformer-vit), allowing it to capture global context within images through [self-attention mechanisms](https://www.ultralytics.com/glossary/self-attention). This transformer-based approach enables superior feature extraction compared to traditional [Convolutional Neural Networks (CNNs)](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn), leading to higher accuracy, especially in complex scenes with intricate object relationships.

### Strengths and Weaknesses

**Strengths:**

- **High Accuracy:** The transformer architecture provides excellent object detection accuracy, making it a strong choice for precision-focused tasks.
- **Robust Feature Extraction:** Effectively captures global context and long-range dependencies in images.
- **Real-Time Capable:** Achieves competitive inference speeds suitable for real-time applications, provided adequate hardware is available.

**Weaknesses:**

- **Higher Resource Demand:** RTDETRv2 models have significantly higher parameter counts and FLOPs, requiring more computational power and memory.
- **Slower Inference:** Generally slower than YOLOv9, particularly on non-GPU hardware or less powerful devices.
- **High Memory Usage:** Transformer architectures are known to be memory-intensive, especially during training, which often demands high CUDA memory and can be a barrier for many users.
- **Less Versatile:** Primarily focused on object detection, lacking the built-in multi-task versatility of models in the Ultralytics ecosystem.
- **Complexity:** Can be more complex to train, tune, and deploy compared to the streamlined and user-friendly Ultralytics YOLO models.

### Ideal Use Cases

RTDETRv2 is best suited for scenarios where achieving the highest possible accuracy is the primary goal and computational resources are not a major constraint:

- **Medical Imaging:** Analyzing complex medical scans where precision is critical for diagnosis.
- **Satellite Imagery:** Detecting small or obscured objects in high-resolution [satellite images](https://www.ultralytics.com/blog/using-computer-vision-to-analyze-satellite-imagery).
- **Scientific Research:** Used in research environments where model performance is prioritized over deployment efficiency.

[Learn more about RT-DETR](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

## Performance Head-to-Head: YOLOv9 vs. RTDETRv2

The following table provides a detailed performance comparison between various sizes of YOLOv9 and RTDETRv2 models on the COCO val dataset.

| Model       | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ----------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv9t     | 640                   | 38.3                 | -                              | 2.3                                 | 2.0                | 7.7               |
| YOLOv9s     | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m     | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c     | 640                   | 53.0                 | -                              | **7.16**                            | 25.3               | 102.1             |
| **YOLOv9e** | 640                   | **55.6**             | -                              | 16.77                               | 57.3               | 189.0             |
|             |                       |                      |                                |                                     |                    |                   |
| RTDETRv2-s  | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m  | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l  | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x  | 640                   | 54.3                 | -                              | 15.03                               | 76                 | 259               |

From the data, several key insights emerge:

- **Peak Accuracy:** YOLOv9-E achieves the highest mAP of 55.6%, surpassing all other models in the comparison.
- **Efficiency:** When comparing models with similar accuracy, YOLOv9 consistently demonstrates superior efficiency. For instance, YOLOv9-C (53.0 mAP) is faster and requires significantly fewer parameters (25.3M vs. 42M) and FLOPs (102.1B vs. 136B) than RTDETRv2-L (53.4 mAP).
- **Speed:** YOLOv9 models generally offer faster inference speeds on GPU with TensorRT. The YOLOv9-C model is notably faster than the comparable RTDETRv2-L.

## Conclusion: Which Model Should You Choose?

For the vast majority of real-world applications, **YOLOv9 is the recommended choice**. It offers a superior combination of accuracy, speed, and efficiency. Its innovative architecture ensures state-of-the-art performance while being mindful of computational resources. The key advantages of choosing YOLOv9, especially within the Ultralytics framework, are its **ease of use, lower memory requirements, versatility across multiple tasks, and the robust support of a well-maintained ecosystem**.

RTDETRv2 is a powerful model for niche applications where precision is the absolute priority and the higher computational and memory costs are acceptable. However, its complexity and resource-intensive nature make it less practical for widespread deployment compared to the highly optimized and user-friendly YOLOv9.

## Other Models to Consider

If you are exploring different options, you might also be interested in other state-of-the-art models available in the Ultralytics ecosystem:

- **[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)**: The latest and most advanced model from Ultralytics, pushing the boundaries of speed and accuracy even further.
- **[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/)**: A mature and highly popular model known for its exceptional balance of performance and versatility across a wide range of vision tasks.
- **[YOLOv5](https://docs.ultralytics.com/models/yolov5/)**: An industry-standard model, renowned for its reliability, speed, and ease of deployment, especially on [edge devices](https://docs.ultralytics.com/guides/nvidia-jetson/).
