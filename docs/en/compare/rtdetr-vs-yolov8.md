---
comments: true
description: Compare RTDETRv2 and YOLOv8 for object detection. Explore architecture, performance, and use cases to select the best model for your needs.
keywords: RTDETRv2, YOLOv8, object detection, computer vision, model comparison, deep learning, transformer architecture, real-time AI, Ultralytics
---

# Model Comparison: RTDETRv2 vs YOLOv8 for Object Detection

When selecting a computer vision model for object detection tasks, understanding the nuances between different architectures is crucial. This page provides a detailed technical comparison between RTDETRv2 and Ultralytics YOLOv8, two state-of-the-art models in the field. We will delve into their architectural differences, performance metrics, ideal use cases, and discuss their respective strengths and weaknesses to guide you in choosing the right model for your project.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["RTDETRv2", "YOLOv8"]'></canvas>

## RTDETRv2: High Accuracy with Transformer Efficiency

**RTDETRv2** (Real-Time Detection Transformer version 2) was introduced by Baidu researchers Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu on April 17, 2023. This model leverages a **Vision Transformer (ViT)** architecture to achieve high accuracy while aiming for real-time performance.

- **Authors:** Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, Yi Liu
- **Organization:** Baidu
- **Date:** 2023-04-17
- **Arxiv Link:** [https://arxiv.org/abs/2304.08069](https://arxiv.org/abs/2304.08069) ([RT-DETRv2 update: https://arxiv.org/abs/2407.17140](https://arxiv.org/abs/2407.17140))
- **GitHub Link:** [https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)
- **Docs Link:** [https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme)

### Architecture and Features

RTDETRv2 employs a transformer-based architecture, specifically using a [Vision Transformer (ViT)](https://www.ultralytics.com/glossary/vision-transformer-vit) backbone combined with a hybrid efficient encoder. This allows the model to capture global context within images more effectively than traditional CNNs through self-attention mechanisms, potentially leading to higher accuracy, especially in complex scenes with many objects. It uses an anchor-free detection approach. More details can be found in the [RT-DETR Arxiv paper](https://arxiv.org/abs/2304.08069).

### Strengths and Weaknesses

**Strengths:**

- **High Accuracy:** The transformer architecture enables superior object detection accuracy, particularly in complex scenarios.
- **Real-time Capability:** Optimized for speed, RTDETRv2 offers a balance between accuracy and inference time, suitable for real-time applications on GPUs.
- **Robust Feature Extraction:** ViTs excel at capturing global context and intricate details.

**Weaknesses:**

- **Computational Cost:** Transformer models are generally more computationally intensive, requiring more parameters, FLOPs, and significantly more CUDA memory during training compared to CNN-based models like YOLOv8.
- **Inference Speed:** While optimized for real-time GPU inference, RTDETRv2's speed might be slower than the fastest YOLO models, especially on CPU or resource-constrained edge devices.
- **Complexity:** The architecture and training process can be more complex compared to the streamlined YOLO framework.

### Use Cases

RTDETRv2 is particularly suited for applications where the highest accuracy is paramount and sufficient computational resources (especially GPUs) are available. Ideal use cases include [AI in self-driving cars](https://www.ultralytics.com/solutions/ai-in-automotive) for precise environmental perception, advanced [robotics](https://www.ultralytics.com/glossary/robotics) requiring accurate object interaction, and [AI in healthcare](https://www.ultralytics.com/solutions/ai-in-healthcare), such as detailed [medical image analysis](https://www.ultralytics.com/glossary/medical-image-analysis).

[Learn more about RTDETRv2](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

## Ultralytics YOLOv8: Streamlined Efficiency and Versatility

**Ultralytics YOLOv8** is the latest iteration in the renowned YOLO (You Only Look Once) family, developed by Glenn Jocher, Ayush Chaurasia, and Jing Qiu at Ultralytics and released on January 10, 2023. It builds upon the success of previous versions, offering state-of-the-art performance with a focus on speed, efficiency, and ease of use.

- **Authors:** Glenn Jocher, Ayush Chaurasia, Jing Qiu
- **Organization:** Ultralytics
- **Date:** 2023-01-10
- **GitHub Link:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Docs Link:** [https://docs.ultralytics.com/models/yolov8/](https://docs.ultralytics.com/models/yolov8/)

### Architecture and Features

YOLOv8 utilizes a single-stage detector architecture based on optimized CNNs, prioritizing rapid inference. It features an anchor-free detection head, a flexible backbone (like CSPDarknet), and a refined loss function. This design achieves an excellent balance between speed and accuracy, making it highly suitable for real-time applications across various hardware platforms.

### Strengths and Weaknesses

**Strengths:**

- **Exceptional Speed:** YOLOv8 excels in inference speed on both CPU and GPU, crucial for real-time processing.
- **High Efficiency:** Computationally efficient with lower memory requirements during training and inference compared to transformer models, making it ideal for edge devices.
- **Ease of Use:** Offers a streamlined user experience with a simple [Python API](https://docs.ultralytics.com/usage/python/), extensive [documentation](https://docs.ultralytics.com/models/yolov8/), and readily available pre-trained weights.
- **Well-Maintained Ecosystem:** Benefits from the integrated Ultralytics ecosystem, including active development on [GitHub](https://github.com/ultralytics/ultralytics), strong community support, frequent updates, and platforms like [Ultralytics HUB](https://www.ultralytics.com/hub) for no-code training and deployment.
- **Versatility:** Supports multiple vision tasks beyond object detection, including [instance segmentation](https://docs.ultralytics.com/tasks/segment/), [image classification](https://docs.ultralytics.com/tasks/classify/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), and oriented bounding boxes ([OBB](https://docs.ultralytics.com/tasks/obb/)).
- **Training Efficiency:** Efficient training process with support for various [datasets](https://docs.ultralytics.com/datasets/) and features like [hyperparameter tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning/).

**Weaknesses:**

- **Accuracy in Complex Scenes:** While highly accurate, single-stage detectors like YOLOv8 might be slightly outperformed by transformers in extremely complex scenes with heavy occlusion or very small objects, though YOLOv8 significantly minimizes this gap.

### Use Cases

YOLOv8's speed, efficiency, and versatility make it ideal for a vast range of applications. It excels in real-time systems like [AI in robotics](https://www.ultralytics.com/blog/from-algorithms-to-automation-ais-role-in-robotics), surveillance ([security alarm systems](https://docs.ultralytics.com/guides/security-alarm-system/)), industrial automation ([improving manufacturing with computer vision](https://www.ultralytics.com/blog/improving-manufacturing-with-computer-vision)), and [AI in agriculture](https://www.ultralytics.com/solutions/ai-in-agriculture). Its efficiency makes it perfect for deployment on resource-constrained edge devices like [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/) or [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/). Explore more applications on the [Ultralytics Solutions](https://www.ultralytics.com/solutions) page.

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## Performance Comparison: RTDETRv2 vs YOLOv8

The table below provides a quantitative comparison of different model variants based on key performance metrics. Note that RTDETRv2 results often rely on GPU acceleration (TensorRT) for real-time speeds, while YOLOv8 demonstrates strong performance even on CPU.

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| :--------- | :-------------------- | :------------------- | :----------------------------- | :---------------------------------- | :----------------- | :---------------- |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | **54.3**             | -                              | 15.03                               | 76                 | 259               |
|            |                       |                      |                                |                                     |                    |                   |
| YOLOv8n    | 640                   | 37.3                 | **80.4**                       | **1.47**                            | **3.2**            | **8.7**           |
| YOLOv8s    | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m    | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l    | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x    | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |

## Conclusion

Both RTDETRv2 and Ultralytics YOLOv8 are powerful object detection models, but they cater to different priorities.

- Choose **RTDETRv2** when **maximum accuracy** is the absolute priority, especially in complex scenes, and you have sufficient **GPU resources** for training and deployment.
- Choose **Ultralytics YOLOv8** when **speed, efficiency, versatility, and ease of use** are critical. Its excellent performance across various hardware (including CPU and edge devices), lower resource requirements, multi-task capabilities, and robust ecosystem make it a highly practical and recommended choice for a wide array of real-world applications.

For developers and researchers seeking a balance of performance, flexibility, and a streamlined workflow, YOLOv8 presents a compelling advantage.

Explore other models within the Ultralytics ecosystem, such as [YOLOv9](https://docs.ultralytics.com/models/yolov9/), [YOLOv10](https://docs.ultralytics.com/models/yolov10/), and [YOLO11](https://docs.ultralytics.com/models/yolo11/), for the latest advancements in efficient and accurate object detection. Refer to the [Ultralytics documentation](https://docs.ultralytics.com/models/) and [GitHub repository](https://github.com/ultralytics/ultralytics) for more detailed information and comparisons.
