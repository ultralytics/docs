---
comments: true
description: Discover key differences between YOLOv8 and YOLOv5. Compare speed, accuracy, use cases, and more to choose the ideal model for your computer vision needs.
keywords: YOLOv8, YOLOv5, object detection, YOLO comparison, computer vision, model comparison, speed, accuracy, Ultralytics, deep learning
---

# YOLOv8 vs YOLOv5: A Detailed Comparison

Comparing Ultralytics YOLOv8 and [Ultralytics YOLOv5](https://docs.ultralytics.com/models/yolov5/) for object detection reveals both the consistent evolution of the YOLO architecture and the distinct strengths of each model. Both models, developed by [Ultralytics](https://www.ultralytics.com/), are renowned for their exceptional balance of speed and accuracy. However, they cater to different priorities in the field of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv). This page provides a detailed technical comparison to help developers and researchers make an informed decision based on their project requirements, highlighting the advantages of the Ultralytics ecosystem.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv8", "YOLOv5"]'></canvas>

## YOLOv8: The Cutting-Edge Solution

**Authors**: Glenn Jocher, Ayush Chaurasia, and Jing Qiu  
**Organization**: [Ultralytics](https://www.ultralytics.com/)  
**Date**: 2023-01-10  
**GitHub**: <https://github.com/ultralytics/ultralytics>  
**Docs**: <https://docs.ultralytics.com/models/yolov8/>

[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) is the latest major release in the YOLO series, engineered as a unified framework to support a full range of vision AI tasks. These include [object detection](https://docs.ultralytics.com/tasks/detect/), [instance segmentation](https://docs.ultralytics.com/tasks/segment/), [image classification](https://docs.ultralytics.com/tasks/classify/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), and oriented bounding boxes (OBB). YOLOv8 introduces significant architectural innovations, such as an [anchor-free detector](https://www.ultralytics.com/glossary/anchor-free-detectors) and a refined C2f neck, which enhance its performance and flexibility.

### Strengths

- **Superior Accuracy and Performance**: YOLOv8 consistently delivers higher accuracy (mAP) than YOLOv5 across all model sizes while maintaining competitive inference speeds. This provides an excellent balance of performance for demanding applications.
- **Enhanced Versatility**: Designed as a comprehensive framework, YOLOv8 natively supports multiple vision tasks. This versatility allows developers to use a single, consistent model architecture for complex, multi-faceted projects, streamlining development and deployment.
- **Modern Architecture**: YOLOv8's anchor-free design reduces the complexity of the training process and the number of hyperparameters to tune, often leading to better generalization on diverse datasets. The updated C2f module provides more efficient feature fusion compared to YOLOv5's C3 module.
- **Streamlined User Experience**: As with all Ultralytics models, YOLOv8 benefits from a simple [Python API](https://docs.ultralytics.com/usage/python/) and [CLI](https://docs.ultralytics.com/usage/cli/), extensive [documentation](https://docs.ultralytics.com/models/yolov8/), and a well-maintained ecosystem. This includes integration with [Ultralytics HUB](https://www.ultralytics.com/hub) for no-code training and [MLOps](https://www.ultralytics.com/glossary/machine-learning-operations-mlops) management.
- **Memory Efficiency**: Ultralytics YOLO models are optimized for low memory usage during both training and inference, making them more accessible than many resource-intensive architectures like Transformers.

### Weaknesses

- **Computational Requirements**: While efficient, the larger YOLOv8 models (L/X) require substantial computational power, which could be a constraint for deployment on severely resource-limited edge devices.

### Ideal Use Cases

YOLOv8 is the recommended choice for new projects that require state-of-the-art performance and flexibility.

- **Advanced Robotics**: For complex scene understanding and object interaction where high accuracy is critical.
- **High-Resolution Image Analysis**: Excels in applications like [medical image analysis](https://www.ultralytics.com/glossary/medical-image-analysis) where detecting fine-grained details is crucial.
- **Multi-Task Vision Systems**: Ideal for systems that need to perform detection, segmentation, and pose estimation simultaneously, such as in [smart retail analytics](https://www.ultralytics.com/blog/achieving-retail-efficiency-with-ai).

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## YOLOv5: The Established and Versatile Standard

**Author**: Glenn Jocher  
**Organization**: [Ultralytics](https://www.ultralytics.com/)  
**Date**: 2020-06-26  
**GitHub**: <https://github.com/ultralytics/yolov5>  
**Docs**: <https://docs.ultralytics.com/models/yolov5/>

Ultralytics YOLOv5 became an industry standard shortly after its release, celebrated for its exceptional balance of speed, accuracy, and remarkable ease of use. Built on [PyTorch](https://pytorch.org/), it features a CSPDarknet53 backbone and a PANet neck. Its anchor-based detection head is highly efficient, and the model scales across various sizes (n, s, m, l, x) to fit different computational budgets.

### Strengths

- **Exceptional Inference Speed**: YOLOv5 is highly optimized for rapid inference, making it a go-to choice for real-time systems, especially on CPU and [edge devices](https://www.ultralytics.com/blog/edge-ai-and-aiot-upgrade-any-camera-with-ultralytics-yolov8-in-a-no-code-way).
- **Mature and Stable Ecosystem**: Having been in the field for several years, YOLOv5 has a vast user base, extensive community support, and a wealth of tutorials and third-party integrations. Its stability makes it a reliable choice for production environments.
- **Ease of Use**: YOLOv5 is renowned for its simple API and straightforward training pipeline, which made it incredibly popular for both beginners and experts. The [Ultralytics ecosystem](https://docs.ultralytics.com/) ensures a smooth user experience from training to deployment.
- **Training Efficiency**: The model offers an efficient training process with readily available [pre-trained weights](https://github.com/ultralytics/yolov5/releases), enabling rapid prototyping and development.

### Weaknesses

- **Lower Accuracy**: Compared to YOLOv8, YOLOv5 models generally have lower mAP scores for a given size. The performance gap becomes more noticeable with larger models.
- **Anchor-Based Detection**: Its reliance on predefined anchor boxes can sometimes require manual tuning for optimal performance on datasets with unusually shaped or scaled objects.

### Ideal Use Cases

YOLOv5 remains a powerful and relevant model, particularly for applications where speed and stability are paramount.

- **Edge Computing**: Its smaller variants (n/s) are perfect for deployment on resource-constrained devices like [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/) and [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/).
- **Real-Time Surveillance**: Ideal for [security systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8) and live video monitoring where low latency is essential.
- **Mobile Applications**: Suitable for on-device [object detection](https://www.ultralytics.com/glossary/object-detection) tasks where computational resources are limited.

[Learn more about YOLOv5](https://docs.ultralytics.com/models/yolov5/){ .md-button }

## Performance Face-Off: YOLOv8 vs. YOLOv5

The performance benchmarks on the [COCO dataset](https://docs.ultralytics.com/datasets/detect/coco/) clearly illustrate the advancements made with YOLOv8. Across the board, YOLOv8 models deliver superior accuracy with comparable or improved performance characteristics.

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv8n | 640                   | 37.3                 | 80.4                           | 1.47                                | **3.2**            | **8.7**           |
| YOLOv8s | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x | 640                   | **53.9**             | 479.1                          | 14.37                               | 68.2               | 257.8             |
|         |                       |                      |                                |                                     |                    |                   |
| YOLOv5n | 640                   | 28.0                 | **73.6**                       | **1.12**                            | 2.6                | 7.7               |
| YOLOv5s | 640                   | 37.4                 | 120.7                          | 1.92                                | 9.1                | 24.0              |
| YOLOv5m | 640                   | 45.4                 | 233.9                          | 4.03                                | 25.1               | 64.2              |
| YOLOv5l | 640                   | 49.0                 | 408.4                          | 6.61                                | 53.2               | 135.0             |
| YOLOv5x | 640                   | 50.7                 | 763.2                          | 11.89                               | 97.2               | 246.4             |

From the table, it's evident that YOLOv8 models offer a significant mAP boost. For instance, YOLOv8s achieves 44.9 mAP, far surpassing YOLOv5s's 37.4 mAP with only a marginal increase in parameters and latency. Similarly, YOLOv8x reaches 53.9 mAP, outperforming YOLOv5x's 50.7 mAP while being more computationally efficient.

## Conclusion: Which Model Should You Choose?

Both YOLOv5 and YOLOv8 are excellent models, but they serve different needs.

- **YOLOv5** is a fantastic choice for applications where **maximum inference speed** and a mature, stable platform are the highest priorities. It remains a strong contender for deployment on resource-constrained devices and for projects that benefit from its extensive ecosystem.

- **YOLOv8** represents the next generation of YOLO technology. It is the recommended choice for **new projects** seeking the **highest accuracy** and **versatility** across multiple vision tasks. Its modern, anchor-free architecture and unified framework make it a more powerful and flexible solution for a wide range of applications, from research to production.

For most use cases, the superior performance and flexibility of YOLOv8 make it the preferred option.

## Explore Other Models

Ultralytics continues to innovate in the field of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv). For users exploring other state-of-the-art options, we also offer models like [YOLOv9](https://docs.ultralytics.com/models/yolov9/), [YOLOv10](https://docs.ultralytics.com/models/yolov10/), and the latest [YOLO11](https://docs.ultralytics.com/models/yolo11/), each providing unique advantages. You can find more detailed analyses on our [model comparison page](https://docs.ultralytics.com/compare/).
