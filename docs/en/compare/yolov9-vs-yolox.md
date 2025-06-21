---
comments: true
description: Discover a detailed comparison of YOLOv9 and YOLOX, covering architectures, benchmarks, and use cases to help you choose the best object detection model.
keywords: YOLOv9, YOLOX, object detection, model comparison, computer vision, YOLO models, architecture, benchmarks, deep learning
---

# YOLOv9 vs. YOLOX: A Technical Comparison

Selecting the optimal object detection model is crucial for achieving desired outcomes in computer vision projects. Models differ significantly in architecture, performance, and resource requirements. This page provides a detailed technical comparison between [YOLOv9](https://docs.ultralytics.com/models/yolov9/) and YOLOX, analyzing their key features to help you choose the best fit for your needs.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv9", "YOLOX"]'></canvas>

## YOLOv9: Advancing Real-Time Object Detection

**Authors:** Chien-Yao Wang, Hong-Yuan Mark Liao  
**Organization:** Institute of Information Science, Academia Sinica, Taiwan  
**Date:** 2024-02-21  
**Arxiv:** [arXiv:2402.13616](https://arxiv.org/abs/2402.13616)  
**GitHub:** [github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)  
**Docs:** [docs.ultralytics.com/models/yolov9/](https://docs.ultralytics.com/models/yolov9/)

Ultralytics [YOLOv9](https://docs.ultralytics.com/models/yolov9/) represents a significant leap in object detection, introducing innovative techniques like **Programmable Gradient Information (PGI)** and the **Generalized Efficient Layer Aggregation Network (GELAN)**. Developed by Chien-Yao Wang and Hong-Yuan Mark Liao, YOLOv9 tackles information loss in deep neural networks, enhancing both accuracy and efficiency. Integrated into the Ultralytics ecosystem, YOLOv9 benefits from a streamlined user experience, comprehensive [documentation](https://docs.ultralytics.com/models/yolov9/), and robust community support.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

### Architecture and Key Features

YOLOv9's architecture is designed to preserve crucial information flow through deep layers using PGI. This helps mitigate the information bottleneck problem common in deep networks. GELAN optimizes the network structure for better parameter utilization and computational efficiency, building on concepts from CSPNet and ELAN. This results in state-of-the-art performance with remarkable efficiency. The Ultralytics implementation ensures **ease of use** with a simple [Python API](https://docs.ultralytics.com/usage/python/) and efficient [training processes](https://docs.ultralytics.com/modes/train/), leveraging readily available pre-trained weights.

### Strengths

- **State-of-the-Art Accuracy:** Achieves leading mAP scores on benchmarks like [COCO](https://docs.ultralytics.com/datasets/detect/coco/), often outperforming other models at similar sizes.
- **High Efficiency:** Delivers high accuracy with fewer parameters and FLOPs compared to many alternatives, making it suitable for [edge AI](https://www.ultralytics.com/glossary/edge-ai) deployment.
- **Information Preservation:** PGI effectively mitigates information loss, improving the model's learning capacity and final performance.
- **Ultralytics Ecosystem:** Benefits from active development, extensive resources, [Ultralytics HUB](https://hub.ultralytics.com/) integration for MLOps, and lower memory requirements during training.
- **Versatility:** While the original paper focuses on detection, the architecture shows potential for tasks like [instance segmentation](https://docs.ultralytics.com/tasks/segment/) and more, aligning with the multi-task capabilities of models like [YOLOv8](https://docs.ultralytics.com/models/yolov8/).

### Weaknesses

- As a newer model, the range of community-driven deployment examples might still be growing compared to long-established models. However, its integration within the Ultralytics framework significantly accelerates adoption and provides a robust support system.

## YOLOX: High-Performance Anchor-Free Detector

**Authors:** Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun  
**Organization:** [Megvii](https://en.megvii.com/)  
**Date:** 2021-07-18  
**Arxiv:** [https://arxiv.org/abs/2107.08430](https://arxiv.org/abs/2107.08430)  
**GitHub:** [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)  
**Docs:** [https://yolox.readthedocs.io/en/latest/](https://yolox.readthedocs.io/en/latest/)

YOLOX, developed by Megvii, is an [anchor-free](https://www.ultralytics.com/glossary/anchor-free-detectors) object detection model that aims for simplicity and high performance. By removing the anchor box mechanism, YOLOX simplifies the training pipeline and reduces the number of design parameters, which can improve generalization.

[Learn more about YOLOX](https://yolox.readthedocs.io/en/latest/){ .md-button }

### Architecture and Key Features

YOLOX distinguishes itself with several key architectural choices. The most significant is its anchor-free design, which treats object detection as a per-pixel prediction problem. Other key features include a **decoupled head** that separates the classification and localization tasks, an advanced label assignment strategy called **SimOTA**, and the use of strong [data augmentation](https://www.ultralytics.com/glossary/data-augmentation) techniques like MixUp and Mosaic.

### Strengths

- **Anchor-Free Design:** Simplifies the model architecture and training process by eliminating the need for anchor box tuning.
- **Strong Performance:** Achieves a competitive balance between mean Average Precision (mAP) and inference speed for its time.
- **Scalability:** Offers a range of model sizes, from YOLOX-Nano to YOLOX-X, allowing deployment across various computational resources.

### Weaknesses

- **Outperformed by Newer Models:** While innovative, YOLOX has been surpassed in both accuracy and efficiency by newer models like YOLOv9.
- **Fragmented Ecosystem:** While open-source, it lacks the integrated ecosystem and streamlined tooling provided by Ultralytics, such as seamless integration with [Ultralytics HUB](https://hub.ultralytics.com/) for MLOps.
- **Higher Computational Cost:** For a given accuracy level, larger YOLOX models tend to have more parameters and FLOPs than comparable YOLOv9 models.

## Performance Comparison: YOLOv9 vs. YOLOX

When comparing performance on the [COCO dataset](https://docs.ultralytics.com/datasets/detect/coco/), YOLOv9 demonstrates a clear advantage in both accuracy and efficiency. The table below shows that YOLOv9 models consistently achieve higher mAP scores with fewer parameters and FLOPs than their YOLOX counterparts. For instance, YOLOv9-C achieves a 53.0% mAP with 25.3M parameters, outperforming YOLOX-L (49.7% mAP with 54.2M parameters) and YOLOX-X (51.1% mAP with 99.1M parameters) while being significantly more efficient. The largest model, YOLOv9-E, pushes the accuracy boundary to 55.6% mAP, a level that YOLOX does not reach. This superior performance-per-computation makes YOLOv9 a more powerful and resource-friendly choice for modern applications.

| Model     | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| --------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv9t   | 640                   | 38.3                 | -                              | **2.3**                             | 2.0                | 7.7               |
| YOLOv9s   | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m   | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c   | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e   | 640                   | **55.6**             | -                              | 16.77                               | 57.3               | 189.0             |
|           |                       |                      |                                |                                     |                    |                   |
| YOLOXnano | 416                   | 25.8                 | -                              | -                                   | **0.91**           | **1.08**          |
| YOLOXtiny | 416                   | 32.8                 | -                              | -                                   | 5.06               | 6.45              |
| YOLOXs    | 640                   | 40.5                 | -                              | 2.56                                | 9.0                | 26.8              |
| YOLOXm    | 640                   | 46.9                 | -                              | 5.43                                | 25.3               | 73.8              |
| YOLOXl    | 640                   | 49.7                 | -                              | 9.04                                | 54.2               | 155.6             |
| YOLOXx    | 640                   | 51.1                 | -                              | 16.1                                | 99.1               | 281.9             |

## Ideal Use Cases

### YOLOv9

YOLOv9's superior accuracy and efficiency make it the ideal choice for demanding applications where performance is critical. It excels in scenarios such as:

- **Advanced Driver-Assistance Systems (ADAS):** Detecting vehicles, pedestrians, and road signs with high precision for [autonomous driving](https://www.ultralytics.com/solutions/ai-in-automotive).
- **High-Fidelity Security:** Monitoring complex scenes in [security systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8) with low false positive rates.
- **Industrial Automation:** Performing detailed [quality control in manufacturing](https://www.ultralytics.com/blog/improving-manufacturing-with-computer-vision) by identifying small defects.
- **Medical Imaging:** Assisting in the analysis of medical scans by providing accurate [object detection](https://www.ultralytics.com/glossary/object-detection) of anomalies.

### YOLOX

YOLOX is well-suited for applications that require a solid balance of accuracy and speed, particularly where its anchor-free design might offer benefits for specific datasets. Ideal use cases include:

- **Real-time Tracking:** Applications in [robotics](https://www.ultralytics.com/glossary/robotics) and surveillance systems where real-time object tracking is needed.
- **Academic Research:** Its modular and anchor-free design makes it an interesting model for research and experimentation in object detection architectures.
- **Edge Deployments:** The smaller YOLOX-Nano and YOLOX-Tiny variants can be deployed on resource-constrained devices, although newer models like YOLOv9 often provide better performance for the same resource cost.

## Conclusion and Recommendation

Both YOLOv9 and YOLOX have contributed significantly to the field of object detection. YOLOX pushed the boundaries with its anchor-free design and decoupled head, offering a strong baseline for real-time detection. However, YOLOv9 has set a new standard for both accuracy and efficiency. Its innovative PGI and GELAN architectures allow it to achieve superior performance with fewer computational resources.

For developers and researchers looking for the best performance, efficiency, and ease of use, **YOLOv9 is the clear choice**. Its integration into the Ultralytics ecosystem provides unparalleled advantages:

- **Ease of Use:** A streamlined [Python API](https://docs.ultralytics.com/usage/python/), extensive [documentation](https://docs.ultralytics.com/), and straightforward [CLI usage](https://docs.ultralytics.com/usage/cli/) simplify development.
- **Well-Maintained Ecosystem:** Active development, strong community support, frequent updates, and integration with [Ultralytics HUB](https://hub.ultralytics.com/) for seamless [MLOps](https://www.ultralytics.com/glossary/machine-learning-operations-mlops).
- **Performance Balance:** An excellent trade-off between speed and accuracy, making it suitable for diverse real-world scenarios from [edge](https://docs.ultralytics.com/guides/nvidia-jetson/) to cloud.
- **Training Efficiency:** Faster training times, readily available pre-trained weights, and efficient resource utilization.

## Explore Other Models

While this page focuses on YOLOv9 and YOLOX, the field of computer vision is vast. We encourage you to explore other state-of-the-art models available within the Ultralytics ecosystem. Consider checking out our comparisons of [YOLOv9 vs. YOLOv8](https://docs.ultralytics.com/compare/yolov8-vs-yolov9/) for insights into the latest Ultralytics models, or [YOLOv9 vs. YOLOv5](https://docs.ultralytics.com/compare/yolov5-vs-yolov9/) to see how far the technology has progressed from an established industry standard. For those interested in transformer-based architectures, our [RT-DETR vs. YOLOv9](https://docs.ultralytics.com/compare/rtdetr-vs-yolov9/) comparison offers a detailed analysis.
