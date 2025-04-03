---
comments: true
description: Compare YOLOv8 and YOLOv9 models for object detection. Explore their accuracy, speed, resources, and use cases to choose the best model for your needs.
keywords: YOLOv8, YOLOv9, object detection, model comparison, Ultralytics, performance metrics, real-time AI, computer vision, YOLO series
---

# Model Comparison: YOLOv8 vs YOLOv9 for Object Detection

Choosing the right object detection model is crucial for balancing accuracy, speed, and computational resources. This page offers a detailed technical comparison between Ultralytics YOLOv8 and YOLOv9, both cutting-edge models in the YOLO series. We will analyze their architectures, performance, and use cases to help you determine the best fit for your needs.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv8", "YOLOv9"]'></canvas>

## YOLOv8: Streamlined and Versatile

[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) is a highly successful model developed by Glenn Jocher, Ayush Chaurasia, and Jing Qiu at Ultralytics, released on January 10, 2023. It's known for its excellent balance of speed and accuracy, designed to be user-friendly and versatile. YOLOv8 supports a wide range of vision tasks beyond [object detection](https://www.ultralytics.com/glossary/object-detection), including [instance segmentation](https://docs.ultralytics.com/tasks/segment/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), image classification, and oriented bounding boxes (OBB).

**Authors:** Glenn Jocher, Ayush Chaurasia, and Jing Qiu  
**Organization:** Ultralytics  
**Date:** 2023-01-10  
**GitHub:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)  
**Docs:** [https://docs.ultralytics.com/models/yolov8/](https://docs.ultralytics.com/models/yolov8/)

### Architecture and Key Features

YOLOv8 builds upon previous YOLO versions with architectural refinements like an anchor-free detection head and a modified CSPDarknet backbone. It focuses on ease of use and integration within the robust Ultralytics ecosystem. Key advantages include:

- **Ease of Use:** Offers a streamlined user experience via a simple [Python API](https://docs.ultralytics.com/usage/python/) and [CLI](https://docs.ultralytics.com/usage/cli/), backed by extensive [documentation](https://docs.ultralytics.com/).
- **Well-Maintained Ecosystem:** Benefits from continuous development, a strong open-source community, frequent updates, and integration with [Ultralytics HUB](https://www.ultralytics.com/hub) for MLOps workflows.
- **Performance Balance:** Achieves a strong trade-off between speed and accuracy, making it suitable for diverse real-world deployment scenarios from edge devices to cloud servers.
- **Memory Efficiency:** Generally requires less CUDA memory for training and inference compared to larger architectures like transformers.
- **Versatility:** Excels in handling multiple vision tasks (detection, segmentation, classification, pose, OBB) within a single framework.
- **Training Efficiency:** Features efficient training processes and readily available pre-trained weights on datasets like [COCO](https://docs.ultralytics.com/datasets/detect/coco/).

### Strengths

- **State-of-the-Art Performance:** Delivers excellent mAP and fast inference speeds.
- **Versatile Task Support:** Handles multiple vision tasks effectively.
- **User-Friendly:** Comprehensive documentation and simple API.
- **Strong Community & Ecosystem:** Actively maintained with extensive resources and integrations like [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/) and [OpenVINO](https://docs.ultralytics.com/integrations/openvino/).

### Weaknesses

- **Resource Intensive:** Larger YOLOv8 models (L, X) require significant computational resources for training and inference.
- **Optimization Needs:** May require further optimization like [model pruning](https://www.ultralytics.com/glossary/pruning) for extremely resource-constrained devices.

### Use Cases

YOLOv8's versatility makes it ideal for a broad spectrum of applications:

- Real-time object detection in [security systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8) and [smart cities](https://www.ultralytics.com/blog/computer-vision-ai-in-smart-cities).
- Complex tasks like [pose estimation](https://www.ultralytics.com/blog/pose-estimation-with-ultralytics-yolov8) in [healthcare](https://www.ultralytics.com/solutions/ai-in-healthcare) and [manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing).
- Rapid prototyping and development due to its ease of use.

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## YOLOv9: Advancing Efficiency with Programmable Gradient Information

YOLOv9 was introduced by Chien-Yao Wang and Hong-Yuan Mark Liao from the Institute of Information Science, Academia Sinica, Taiwan, on February 21, 2024. It aims to address information loss challenges in deep neural networks by introducing novel concepts like Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN).

**Authors:** Chien-Yao Wang and Hong-Yuan Mark Liao  
**Organization:** Institute of Information Science, Academia Sinica, Taiwan  
**Date:** 2024-02-21  
**Arxiv:** [https://arxiv.org/abs/2402.13616](https://arxiv.org/abs/2402.13616)  
**GitHub:** [https://github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)  
**Docs:** [https://docs.ultralytics.com/models/yolov9/](https://docs.ultralytics.com/models/yolov9/)

### Architecture and Key Features

YOLOv9's core innovations are:

- **Programmable Gradient Information (PGI):** Designed to provide complete input information for the target task by managing gradient flow, aiming to alleviate information loss and ensure reliable gradient updates.
- **Generalized Efficient Layer Aggregation Network (GELAN):** A novel lightweight network architecture that leverages gradient path planning to achieve superior parameter utilization and computational efficiency.

### Strengths

- **High Accuracy:** Achieves competitive and sometimes superior mAP scores compared to previous models, especially in larger variants.
- **Efficiency Focus:** Introduces architectural innovations aimed at better parameter and computation efficiency.
- **Novel Concepts:** Pushes research boundaries with PGI and GELAN, addressing deep learning information bottlenecks.

### Weaknesses

- **Task Specificity:** Primarily focused on object detection, lacking the built-in multi-task versatility of Ultralytics YOLOv8 within its native framework.
- **Ecosystem Maturity:** As a newer model from a different research group, it lacks the extensive ecosystem, integrations, and community support surrounding Ultralytics models.
- **Complexity:** The introduction of PGI and GELAN might present a steeper learning curve for understanding and modification compared to the more established YOLOv8 architecture.

### Use Cases

YOLOv9 is well-suited for:

- Applications demanding the highest possible accuracy in object detection, where computational resources allow for larger models.
- Research exploring novel network architectures and methods to overcome information loss in deep learning.
- Scenarios where parameter and computational efficiency are critical alongside high accuracy.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

## Performance Comparison

The table below compares the performance of various YOLOv8 and YOLOv9 model sizes on the COCO dataset. Note that CPU speed data for YOLOv9 is not available in the provided table. Ultralytics YOLOv8 demonstrates excellent performance across different scales, offering very fast inference speeds, particularly on GPU with TensorRT, and efficient CPU inference. YOLOv9 models show strong mAP performance, especially the larger variants, while achieving competitive parameter counts and FLOPs.

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv8n | 640                   | 37.3                 | **80.4**                       | **1.47**                            | 3.2                | 8.7               |
| YOLOv8s | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |
|         |                       |                      |                                |                                     |                    |                   |
| YOLOv9t | 640                   | 38.3                 | -                              | 2.3                                 | **2.0**            | **7.7**           |
| YOLOv9s | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e | 640                   | **55.6**             | -                              | 16.77                               | 57.3               | 189.0             |

## Conclusion

Both YOLOv8 and YOLOv9 represent significant advancements in object detection.

**Ultralytics YOLOv8** stands out for its **versatility, ease of use, and robust ecosystem**. It offers a fantastic balance of speed and accuracy across various model sizes and supports multiple vision tasks out-of-the-box, making it an excellent choice for developers and researchers looking for a reliable, well-supported, and highly capable model for diverse applications. Its efficient training, lower memory footprint compared to transformer models, and seamless integration with tools like Ultralytics HUB further enhance its appeal.

**YOLOv9** introduces innovative architectural concepts (PGI and GELAN) targeting improved accuracy and efficiency, particularly in larger models. It's a strong contender for tasks where achieving the absolute highest object detection accuracy is paramount, provided the focus is primarily on detection and the user is comfortable working within its specific framework.

For most users, especially those needing multi-task capabilities, rapid development, and a mature ecosystem, **Ultralytics YOLOv8 remains the recommended choice**.

Explore other models in the Ultralytics ecosystem, such as [YOLOv5](https://docs.ultralytics.com/models/yolov5/), [YOLOv10](https://docs.ultralytics.com/models/yolov10/), and the latest [YOLO11](https://docs.ultralytics.com/models/yolo11/), or compare against other architectures like [RT-DETR](https://docs.ultralytics.com/models/rtdetr/).
