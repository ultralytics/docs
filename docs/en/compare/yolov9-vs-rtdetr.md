---
comments: true
description: Compare YOLOv9 and RTDETRv2 for object detection. Explore speed, accuracy, use cases, and architectures to choose the best for your project.
keywords: YOLOv9, RTDETRv2, object detection, model comparison, AI models, computer vision, YOLO, real-time detection, transformers, efficiency
---

# YOLOv9 vs RTDETRv2: A Technical Comparison for Object Detection

Choosing the optimal object detection model is a critical decision in computer vision projects. Ultralytics offers a suite of models, including the Ultralytics YOLO series known for speed and efficiency, and the RT-DETR series, emphasizing high accuracy. This page provides a detailed technical comparison between **YOLOv9** and **RTDETRv2**, two state-of-the-art models, to assist you in making an informed choice.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv9", "RTDETRv2"]'></canvas>

## YOLOv9: Ultra-Efficient Real-Time Detection

**YOLOv9** ([You Only Look Once 9](https://docs.ultralytics.com/models/yolov9/)) represents a significant advancement in the Ultralytics YOLO family, celebrated for its exceptional speed and efficiency.

- **Authors:** Chien-Yao Wang and Hong-Yuan Mark Liao
- **Organization:** Institute of Information Science, Academia Sinica, Taiwan
- **Date:** 2024-02-21
- **Arxiv Link:** [https://arxiv.org/abs/2402.13616](https://arxiv.org/abs/2402.13616)
- **GitHub Link:** [https://github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)
- **Docs Link:** [https://docs.ultralytics.com/models/yolov9/](https://docs.ultralytics.com/models/yolov9/)

### Architecture and Key Features

YOLOv9 introduces innovative concepts like Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN). These advancements allow the model to learn more effectively, addressing information loss in deep networks and improving accuracy while maintaining a lightweight CNN-based structure. Integrated within the Ultralytics ecosystem, YOLOv9 benefits from a **streamlined user experience**, a simple API, extensive [documentation](https://docs.ultralytics.com/), and readily available [pre-trained weights](https://github.com/WongKinYiu/yolov9#evaluation), facilitating **efficient training** and deployment. Its architecture is optimized for a strong **performance balance** between speed and accuracy. Furthermore, YOLOv9 typically requires **lower memory** for training and inference compared to transformer models like RTDETRv2. A key advantage is its **versatility**, supporting tasks beyond detection, including [segmentation](https://docs.ultralytics.com/tasks/segment/) and potentially others, unlike the more specialized RTDETRv2.

### Performance Metrics

YOLOv9 models achieve a remarkable balance between speed and accuracy. As shown in the comparison table, YOLOv9 models offer competitive mAP scores (e.g., 55.6% for YOLOv9e) with significantly faster inference speeds (e.g., 2.3ms for YOLOv9t on T4 TensorRT) and smaller model sizes compared to RTDETRv2. This efficiency makes it ideal for deployment on resource-constrained devices and in latency-critical applications.

### Strengths and Weaknesses

**Strengths:**

- **High Speed and Efficiency:** Optimized for extremely fast inference, crucial for real-time applications.
- **Smaller Model Size:** Requires less computational resources and memory, facilitating deployment on [edge devices](https://docs.ultralytics.com/guides/nvidia-jetson/).
- **Good Accuracy:** Achieves a strong balance of speed and accuracy.
- **Ease of Use:** Benefits from the well-maintained Ultralytics ecosystem, simple API, and extensive documentation.
- **Versatility:** Supports multiple computer vision tasks.
- **Training Efficiency:** Efficient training process with lower memory requirements.

**Weaknesses:**

- **Accuracy Trade-off:** May not reach the absolute highest accuracy levels of larger transformer models like RTDETRv2 in certain complex scenarios.
- **Global Context:** CNN-based architecture might capture less global context compared to transformers.

### Ideal Use Cases

YOLOv9 is ideally suited for applications where real-time detection and efficiency are paramount:

- **Edge Computing:** Deployment on devices with limited resources, explored in [Empowering Edge AI](https://www.ultralytics.com/blog/empowering-edge-ai-with-sony-imx500-and-aitrios).
- **Real-time Surveillance:** Efficient monitoring in security systems. Learn about [Computer Vision for Theft Prevention](https://www.ultralytics.com/blog/computer-vision-for-theft-prevention-enhancing-security).
- **Robotics and Drones:** Fast perception for navigation, as discussed in [Computer Vision Applications in AI Drone Operations](https://www.ultralytics.com/blog/computer-vision-applications-ai-drone-uav-operations).
- **Mobile Applications:** Integrating object detection into mobile apps.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

## RTDETRv2: Precision-Focused Real-Time Detection

**RTDETRv2** ([Real-Time Detection Transformer v2](https://docs.ultralytics.com/models/rtdetr/)) is designed for applications demanding high accuracy in real-time object detection, leveraging transformer architectures.

- **Authors:** Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu
- **Organization:** Baidu
- **Date:** 2023-04-17 (Original RT-DETR), 2024-07-24 (RTDETRv2 paper)
- **Arxiv Link:** [https://arxiv.org/abs/2304.08069](https://arxiv.org/abs/2304.08069) (Original), [https://arxiv.org/abs/2407.17140](https://arxiv.org/abs/2407.17140) (v2)
- **GitHub Link:** [https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)
- **Docs Link:** [https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme)

### Architecture and Key Features

RTDETRv2's architecture is built upon [Vision Transformers (ViT)](https://www.ultralytics.com/glossary/vision-transformer-vit), allowing it to capture global context within images through self-attention mechanisms. This transformer-based approach enables superior feature extraction compared to traditional CNNs, leading to higher accuracy, especially in complex scenes with intricate object relationships.

### Performance Metrics

RTDETRv2 models demonstrate impressive mAP scores, as highlighted in the comparison table. For instance, RTDETRv2-x achieves a mAP<sup>val</sup>50-95 of 54.3%. While inference speeds are competitive for real-time use (e.g., 5.03ms for RTDETRv2-s on T4 TensorRT), they are generally slower than comparable YOLOv9 models, especially on CPU or less powerful hardware.

### Strengths and Weaknesses

**Strengths:**

- **High Accuracy:** Transformer architecture provides excellent object detection accuracy.
- **Real-Time Capability:** Achieves competitive inference speeds suitable for real-time applications (with adequate hardware).
- **Robust Feature Extraction:** Effectively captures global context.

**Weaknesses:**

- **Larger Model Size & Higher Computation:** RTDETRv2 models have higher parameter counts and FLOPs, requiring more resources.
- **Slower Inference:** Generally slower than YOLOv9, particularly on non-GPU hardware.
- **Higher Memory Usage:** Transformers typically demand more memory during training and inference.
- **Less Versatile:** Primarily focused on object detection.
- **Complexity:** Can be more complex to train and deploy compared to the streamlined Ultralytics YOLO models.

### Ideal Use Cases

RTDETRv2 is best suited for applications where accuracy is paramount and computational resources are less constrained:

- **Autonomous Vehicles:** Demanding reliable environmental perception, detailed in [AI in self-driving cars](https://www.ultralytics.com/solutions/ai-in-automotive).
- **Robotics:** Enabling precise object interaction, relevant to [AI's Role in Robotics](https://www.ultralytics.com/blog/from-algorithms-to-automation-ais-role-in-robotics).
- **Medical Imaging:** Accurate anomaly detection, as seen in [AI in Healthcare applications](https://www.ultralytics.com/solutions/ai-in-healthcare).
- **High-Resolution Analysis:** Detailed analysis like [satellite imagery](https://www.ultralytics.com/blog/using-computer-vision-to-analyse-satellite-imagery).

[Learn more about RTDETRv2](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

## Performance Comparison

The table below provides a quantitative comparison of various YOLOv9 and RTDETRv2 model variants on the COCO dataset. YOLOv9 models consistently demonstrate superior speed (lower ms latency) and efficiency (fewer parameters and FLOPs) for comparable or even better mAP scores in some cases (e.g., YOLOv9e vs RTDETRv2-x).

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv9t    | 640                   | 38.3                 | -                              | **2.3**                             | **2.0**            | **7.7**           |
| YOLOv9s    | 640                   | 46.8                 | -                              | **3.54**                            | **7.1**            | **26.4**          |
| YOLOv9m    | 640                   | 51.4                 | -                              | **6.43**                            | 20.0               | 76.3              |
| YOLOv9c    | 640                   | 53.0                 | -                              | **7.16**                            | 25.3               | 102.1             |
| YOLOv9e    | 640                   | **55.6**             | -                              | 16.77                               | 57.3               | 189.0             |
|            |                       |                      |                                |                                     |                    |                   |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | 54.3                 | -                              | 15.03                               | 76                 | 259               |

## Conclusion

Both YOLOv9 and RTDETRv2 are powerful object detection models, but they cater to different priorities. **RTDETRv2** is a strong choice when maximum accuracy is the absolute priority and computational resources are abundant. However, **YOLOv9**, particularly within the Ultralytics ecosystem, offers a more compelling package for most real-world applications. It provides an excellent balance of **high speed, strong accuracy, and efficiency**, coupled with significant advantages in **ease of use, lower memory requirements, training efficiency, and versatility** across multiple tasks. The robust support, continuous updates, and extensive resources within the Ultralytics framework further enhance YOLOv9's appeal for developers and researchers seeking practical, high-performance computer vision solutions.

### Explore Other Models

For users exploring alternatives, Ultralytics offers a diverse range:

- **[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/):** The previous generation, still offering a great balance of speed and accuracy.
- **[YOLO11](https://docs.ultralytics.com/models/yolo11/):** The latest Ultralytics model, pushing efficiency and speed further.
- **[YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/):** Models designed via Neural Architecture Search for optimal performance.
- **[FastSAM](https://docs.ultralytics.com/models/fast-sam/) and [MobileSAM](https://docs.ultralytics.com/models/mobile-sam/):** Efficient models for instance segmentation tasks.

Refer to the comprehensive [Ultralytics Documentation](https://docs.ultralytics.com/models/) and the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics) for in-depth information.
