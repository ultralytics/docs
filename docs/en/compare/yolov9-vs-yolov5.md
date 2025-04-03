---
comments: true
description: Compare YOLOv9 and YOLOv5 models for object detection. Explore their architecture, performance, use cases, and key differences to choose the best fit.
keywords: YOLOv9 vs YOLOv5, YOLO comparison, Ultralytics models, YOLO object detection, YOLO performance, real-time detection, model differences, computer vision
---

# YOLOv9 vs YOLOv5: A Detailed Comparison

This page provides a technical comparison between YOLOv9 and Ultralytics YOLOv5, two significant models in the YOLO (You Only Look Once) series. We focus on their object detection capabilities, delving into architectural differences, performance metrics, training methodologies, and suitable use cases to help you select the right model for your computer vision tasks.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv9", "YOLOv5"]'></canvas>

## YOLOv9: Programmable Gradient Information

YOLOv9 was introduced in February 2024 by Chien-Yao Wang and Hong-Yuan Mark Liao from the Institute of Information Science, Academia Sinica, Taiwan. It represents a notable advancement in real-time object detection, aiming to overcome information loss in deep neural networks.

**Authors:** Chien-Yao Wang and Hong-Yuan Mark Liao  
**Organization:** Institute of Information Science, Academia Sinica, Taiwan  
**Date:** 2024-02-21  
**Arxiv:** [https://arxiv.org/abs/2402.13616](https://arxiv.org/abs/2402.13616)  
**GitHub:** [https://github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)  
**Docs:** [https://docs.ultralytics.com/models/yolov9/](https://docs.ultralytics.com/models/yolov9/)

### Architecture and Innovations

YOLOv9 introduces two key innovations detailed in their paper "[YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)":

- **Programmable Gradient Information (PGI):** Addresses the information bottleneck problem in deep networks, ensuring crucial gradient information is preserved across layers for more effective learning.
- **Generalized Efficient Layer Aggregation Network (GELAN):** An optimized network architecture that improves parameter utilization and computational efficiency, enhancing accuracy without significantly increasing computational cost.

### Performance

YOLOv9 achieves state-of-the-art performance on the [MS COCO dataset](https://docs.ultralytics.com/datasets/detect/coco/), demonstrating superior accuracy and efficiency compared to many previous real-time object detectors. For instance, the YOLOv9c model achieves 53.0% mAP<sup>val</sup><sub>50-95</sub> with 25.3 million parameters.

### Strengths

- **High Accuracy:** Delivers excellent mAP scores, particularly with larger variants like YOLOv9e.
- **Efficient Design:** GELAN and PGI contribute to better parameter and computational efficiency compared to models with similar accuracy levels.

### Weaknesses

- **Higher Training Resource Demand:** Training YOLOv9 models generally requires more computational resources and time compared to YOLOv5, as noted in the [YOLOv9 documentation](https://docs.ultralytics.com/models/yolov9/).
- **Relatively Newer Model:** As a newer model, the community support and third-party integrations might be less extensive than the well-established YOLOv5.

### Use Cases

YOLOv9 is well-suited for applications demanding high accuracy and efficiency:

- **High-precision object detection:** Scenarios where accuracy is critical, such as [autonomous vehicles](https://www.ultralytics.com/blog/ai-in-self-driving-cars), advanced surveillance, and robotic vision.
- **Resource-constrained environments:** While training is demanding, its efficient architecture allows deployment on edge devices with optimized inference speed.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

## YOLOv5: Versatility and Speed

Ultralytics YOLOv5, authored by Glenn Jocher and released in June 2020, quickly became an industry standard due to its remarkable balance of speed, ease of use, and accuracy. It is developed and maintained by [Ultralytics](https://www.ultralytics.com/).

**Author:** Glenn Jocher  
**Organization:** Ultralytics  
**Date:** 2020-06-26  
**GitHub:** [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)  
**Docs:** [https://docs.ultralytics.com/models/yolov5/](https://docs.ultralytics.com/models/yolov5/)

### Architecture and Features

YOLOv5 is built with a focus on practicality and performance:

- **Architecture:** Utilizes architectures like CSPDarknet53 in the backbone and PANet in the neck for efficient feature extraction and fusion.
- **Ease of Use:** Implemented in [PyTorch](https://pytorch.org/), YOLOv5 is known for its straightforward API, extensive [documentation](https://docs.ultralytics.com/yolov5/), and streamlined workflows for [training](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/), validation, and [deployment](https://docs.ultralytics.com/yolov5/tutorials/model_export/).
- **Scalability:** Offers a range of model sizes (YOLOv5n, s, m, l, x) to cater to different computational budgets and performance needs, from lightweight edge deployments to high-accuracy cloud solutions.
- **Ecosystem:** Benefits from the robust [Ultralytics ecosystem](https://docs.ultralytics.com/integrations/), including [Ultralytics HUB](https://hub.ultralytics.com/) for dataset management and training, active development, and strong community support.
- **Efficiency:** Known for efficient training processes, readily available pre-trained weights, and relatively lower memory requirements compared to more complex architectures like some [Transformer models](https://www.ultralytics.com/glossary/transformer).

### Performance

YOLOv5 provides an excellent trade-off between speed and accuracy, making it highly suitable for a wide array of real-world applications. The smaller YOLOv5s variant achieves 37.4% mAP<sup>val</sup><sub>50-95</sub> with very fast inference speeds, ideal for real-time tasks.

### Strengths

- **High Speed:** Offers exceptionally fast inference speeds, particularly the smaller models (n, s).
- **Ease of Use:** Renowned for its user-friendly design, simple API, comprehensive documentation, and large, active community.
- **Versatility:** Adaptable to various tasks including [object detection](https://docs.ultralytics.com/tasks/detect/), [instance segmentation](https://docs.ultralytics.com/tasks/segment/) (introduced later), and [image classification](https://docs.ultralytics.com/tasks/classify/).
- **Mature Ecosystem:** Benefits from years of development, extensive community resources, tutorials, and third-party integrations.

### Weaknesses

- **Lower Accuracy Compared to YOLOv9:** Generally, YOLOv5 models do not achieve the same peak accuracy as the latest YOLOv9 models, especially on challenging datasets.
- **Architecture:** While highly effective, its architecture doesn't incorporate the novel PGI and GELAN concepts found in YOLOv9.

### Use Cases

YOLOv5's versatility and speed make it exceptionally well-suited for:

- **Real-time applications:** Ideal for tasks requiring fast inference, such as live video processing, robotics, and [drone vision](https://www.ultralytics.com/blog/computer-vision-applications-ai-drone-uav-operations).
- **Edge deployment:** Smaller models (YOLOv5n, YOLOv5s) are excellent for deployment on [edge devices](https://www.ultralytics.com/blog/edge-ai-and-edge-computing-powering-real-time-intelligence) and mobile platforms due to lower computational demands.
- **Rapid prototyping and development:** Its ease of use and extensive resources make it perfect for quick development cycles and educational purposes.

[Learn more about YOLOv5](https://docs.ultralytics.com/models/yolov5/){ .md-button }

## Performance Comparison

Below is a comparison of various YOLOv9 and YOLOv5 model variants, evaluated on the COCO val 2017 dataset.

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| :------ | :-------------------- | :------------------- | :----------------------------- | :---------------------------------- | :----------------- | :---------------- |
| YOLOv9t | 640                   | 38.3                 | -                              | 2.3                                 | **2.0**            | **7.7**           |
| YOLOv9s | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e | 640                   | **55.6**             | -                              | 16.77                               | 57.3               | 189.0             |
|         |                       |                      |                                |                                     |                    |                   |
| YOLOv5n | 640                   | 28.0                 | **73.6**                       | **1.12**                            | 2.6                | **7.7**           |
| YOLOv5s | 640                   | 37.4                 | 120.7                          | 1.92                                | 9.1                | 24.0              |
| YOLOv5m | 640                   | 45.4                 | 233.9                          | 4.03                                | 25.1               | 64.2              |
| YOLOv5l | 640                   | 49.0                 | 408.4                          | 6.61                                | 53.2               | 135.0             |
| YOLOv5x | 640                   | 50.7                 | 763.2                          | 11.89                               | 97.2               | 246.4             |

## Conclusion

YOLOv9 represents a step forward in accuracy and architectural innovation, particularly with its PGI and GELAN components aimed at maximizing information flow and efficiency. It's an excellent choice for applications where achieving the highest possible accuracy is paramount, provided sufficient training resources are available.

Ultralytics YOLOv5 remains a highly competitive and practical choice, especially valued for its speed, ease of use, and robust ecosystem. Its versatility across different model sizes and straightforward deployment process make it ideal for real-time applications, edge computing, and projects requiring rapid development and iteration. The strong community support and extensive documentation further solidify its position as a go-to model for many developers and researchers.

For users exploring other options, Ultralytics also offers models like [YOLOv8](https://docs.ultralytics.com/models/yolov8/), [YOLOv10](https://docs.ultralytics.com/models/yolov10/), and the latest [YOLO11](https://docs.ultralytics.com/models/yolo11/), each providing unique advantages. You can find more comparisons on the [model comparison page](https://docs.ultralytics.com/compare/).
