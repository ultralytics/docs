---
comments: true
description: Compare RTDETRv2 and YOLOv9 object detection models. Explore performance, strengths, weaknesses, and ideal use cases to make an informed decision.
keywords: RTDETRv2, YOLOv9, object detection, Ultralytics models, transformer vision, YOLO series, real-time object detection, model comparison, Vision Transformers, computer vision
---

# RTDETRv2 vs YOLOv9: A Technical Comparison for Object Detection

Choosing the optimal object detection model is a critical decision for computer vision projects. Ultralytics offers a diverse range of models, including the YOLO series known for speed and efficiency, and the RT-DETR series, emphasizing high accuracy. This page delivers a detailed technical comparison between **RTDETRv2** and **YOLOv9**, two state-of-the-art object detection models, to assist you in making an informed choice.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["RTDETRv2", "YOLOv9"]'></canvas>

## RTDETRv2: Transformer-Powered High Accuracy

**RTDETRv2** ([Real-Time Detection Transformer v2](https://docs.ultralytics.com/models/rtdetr/)) is a state-of-the-art object detection model developed by Baidu, known for its exceptional accuracy and real-time performance. Published on [arXiv](https://arxiv.org/abs/2304.08069) on 2023-04-17, and with code available on [GitHub](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch), RTDETRv2 is authored by Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu. It leverages a Vision Transformer (ViT) architecture to achieve precise object localization and classification, making it suitable for demanding applications.

### Architecture and Key Features

RTDETRv2's architecture is built upon Vision Transformers, enabling it to capture global context within images through self-attention mechanisms. This differs significantly from traditional Convolutional Neural Networks (CNNs) and allows RTDETRv2 to weigh the importance of different image regions, resulting in enhanced feature extraction and superior accuracy, especially in complex scenes. The transformer-based design allows for anchor-free detection, simplifying the detection process and potentially improving generalization.

### Performance Metrics

RTDETRv2 demonstrates strong performance, particularly in mAP. As detailed in the comparison table, the RTDETRv2-x variant achieves a mAPval50-95 of 54.3. Inference speeds are also competitive, with RTDETRv2-s reaching 5.03 ms on TensorRT, making it viable for real-time applications when using capable hardware such as NVIDIA T4 GPUs. For a deeper understanding of performance evaluation, refer to our [YOLO Performance Metrics guide](https://docs.ultralytics.com/guides/yolo-performance-metrics/).

### Strengths and Weaknesses

**Strengths:**

- **High Accuracy:** Transformer architecture provides excellent object detection accuracy, crucial for applications requiring precision.
- **Real-Time Capable:** Achieves competitive inference speeds, particularly when optimized with TensorRT and run on suitable hardware.
- **Global Context Understanding:** Vision Transformers effectively capture global context, leading to robust detection in complex environments.

**Weaknesses:**

- **Larger Model Size:** RTDETRv2 models, especially larger variants like RTDETRv2-x, have a substantial parameter count and FLOPs, demanding more computational resources.
- **Inference Speed Limitations:** While real-time is achievable, inference speed might be slower than highly optimized CNN-based models like YOLOv9, especially on resource-constrained devices.

### Ideal Use Cases

RTDETRv2 is ideally suited for applications where accuracy is paramount and computational resources are readily available. These include:

- **Autonomous Vehicles:** For precise and reliable environmental perception. Explore more about [AI in self-driving cars](https://www.ultralytics.com/solutions/ai-in-self-driving).
- **Medical Imaging:** For accurate anomaly detection in medical images, aiding in diagnostics. Learn about [AI in Healthcare](https://www.ultralytics.com/solutions/ai-in-healthcare).
- **Robotics:** To enable robots to interact with and manipulate objects in complex environments accurately. Understand [AI's Role in Robotics](https://www.ultralytics.com/blog/from-algorithms-to-automation-ais-role-in-robotics).
- **High-Resolution Image Analysis:** For detailed analysis of large images, such as in satellite imagery or industrial inspection. See how to [Analyse Satellite Imagery using Computer Vision](https://www.ultralytics.com/blog/using-computer-vision-to-analyse-satellite-imagery).

[Learn more about RTDETRv2](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

## YOLOv9: Programmable Gradient Information for Efficiency and Accuracy

**YOLOv9** ([You Only Look Once 9](https://docs.ultralytics.com/models/yolov9/)) is a cutting-edge object detection model from the renowned Ultralytics YOLO family. Introduced on [arXiv](https://arxiv.org/abs/2402.13616) on 2024-02-21, YOLOv9 is authored by Chien-Yao Wang and Hong-Yuan Mark Liao from the Institute of Information Science, Academia Sinica, Taiwan, with code available on [GitHub](https://github.com/WongKinYiu/yolov9). YOLOv9 introduces Programmable Gradient Information (PGI) and GELAN techniques, enhancing both accuracy and training efficiency compared to previous YOLO versions.

### Architecture and Key Features

YOLOv9 builds upon the efficiency of earlier YOLO models while incorporating novel architectural improvements. It utilizes GELAN (Generalized Efficient Layer Aggregation Network) to optimize network architecture and PGI to maintain gradient information integrity, addressing information loss during deep network propagation. These innovations lead to improved accuracy and more efficient training. YOLOv9 maintains an anchor-free detection head and streamlined single-stage design, focusing on real-time performance.

### Performance Metrics

YOLOv9 achieves a compelling balance of speed and accuracy. The YOLOv9-e model achieves a mAPval50-95 of 55.6, outperforming even larger RTDETRv2 models in accuracy while maintaining competitive inference speeds. The smaller YOLOv9-t variant is exceptionally fast, reaching 2.3 ms inference speed on TensorRT, making it suitable for extremely latency-sensitive applications.

### Strengths and Weaknesses

**Strengths:**

- **High Accuracy and Efficiency:** PGI and GELAN contribute to both higher accuracy and efficient parameter utilization.
- **Fast Inference Speed:** Optimized for real-time performance, especially smaller variants suitable for edge deployment.
- **Efficient Training:** PGI contributes to more stable and efficient training processes.

**Weaknesses:**

- **Lower Global Context:** CNN-based architecture might be less effective in capturing long-range dependencies compared to transformer-based models in very complex scenes.
- **Accuracy Trade-off for Speed:** While highly accurate, achieving the fastest inference speeds may involve using smaller models with slightly reduced accuracy compared to the largest models.

### Ideal Use Cases

YOLOv9 is well-suited for applications requiring a balance of high accuracy and real-time performance, especially in resource-constrained environments:

- **Real-time Surveillance:** For efficient and accurate monitoring in security systems. Explore [computer vision for theft prevention](https://www.ultralytics.com/blog/computer-vision-for-theft-prevention-enhancing-security).
- **Edge Computing:** Deployment on edge devices with limited computational resources. Learn about [Edge AI](https://www.ultralytics.com/glossary/edge-ai).
- **Robotics:** For fast and accurate perception in robotic systems. See [AI's role in robotics](https://www.ultralytics.com/blog/from-algorithms-to-automation-ais-role-in-robotics).
- **Industrial Automation:** For applications in manufacturing requiring real-time object detection for quality control and process optimization. Discover [AI in manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing).

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

## Model Comparison Table

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | 54.3                 | -                              | 15.03                               | 76                 | 259               |
|            |                       |                      |                                |                                     |                    |                   |
| YOLOv9t    | 640                   | 38.3                 | -                              | 2.3                                 | 2.0                | 7.7               |
| YOLOv9s    | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m    | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c    | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e    | 640                   | 55.6                 | -                              | 16.77                               | 57.3               | 189.0             |

## Conclusion

Both RTDETRv2 and YOLOv9 are powerful object detection models, each with unique strengths. **RTDETRv2** excels in scenarios prioritizing maximum accuracy and leveraging transformer architecture for robust feature extraction, suitable for applications with ample computational resources. **YOLOv9**, on the other hand, is ideal when real-time performance and efficiency are paramount, offering a compelling blend of accuracy and speed, particularly beneficial for deployment on edge devices and latency-sensitive systems.

For users interested in exploring other models, Ultralytics offers a wide array of options, including:

- **YOLOv8:** The previous generation [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) model, offering a balance of speed and accuracy.
- **YOLO11:** For enhanced efficiency and speed, consider [YOLO11](https://docs.ultralytics.com/models/yolo11/).
- **FastSAM and MobileSAM:** For real-time instance segmentation tasks, explore [FastSAM](https://docs.ultralytics.com/models/fast-sam/) and [MobileSAM](https://docs.ultralytics.com/models/mobile-sam/).

The choice between RTDETRv2, YOLOv9, and other Ultralytics models depends on the specific needs of your project, carefully considering the balance between accuracy, speed, and available resources. Refer to the [Ultralytics Documentation](https://docs.ultralytics.com/models/) and the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics) for comprehensive details and implementation guides.

## Comments
