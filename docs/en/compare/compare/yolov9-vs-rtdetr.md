---
description: Compare YOLOv9 and RTDETRv2 for object detection. Explore speed, accuracy, use cases, and architectures to choose the best for your project.
keywords: YOLOv9, RTDETRv2, object detection, model comparison, AI models, computer vision, YOLO, real-time detection, transformers, efficiency
---

# YOLOv9 vs RTDETRv2: A Technical Comparison for Object Detection

Choosing the optimal object detection model is a critical decision in computer vision projects. Ultralytics offers a suite of models, including the YOLO series known for speed and efficiency, and the RT-DETR series, emphasizing high accuracy. This page provides a detailed technical comparison between **YOLOv9** and **RTDETRv2**, two state-of-the-art models, to assist you in making an informed choice.

<script async src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv9", "RTDETRv2"]'></canvas>

## RTDETRv2: Precision-Focused Real-Time Detection

**RTDETRv2** ([Real-Time Detection Transformer v2](https://docs.ultralytics.com/models/rtdetr/)) is designed for applications requiring high accuracy in real-time object detection. Developed by Baidu and detailed in their [Arxiv paper](https://arxiv.org/abs/2304.08069) released on 2023-04-17, RTDETRv2 leverages a Vision Transformer (ViT) architecture to achieve state-of-the-art performance. The [RT-DETR GitHub repository](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch) provides implementation details.

### Architecture and Key Features

RTDETRv2's architecture is built upon Vision Transformers, allowing it to capture global context within images through self-attention mechanisms, which is further explained in our [Vision Transformer (ViT) glossary page](https://www.ultralytics.com/glossary/vision-transformer-vit). This transformer-based approach enables superior feature extraction compared to traditional CNN-based detectors, leading to higher accuracy, especially in complex scenes.

### Performance Metrics

RTDETRv2 models demonstrate impressive mAP scores, as highlighted in the comparison table. For instance, RTDETRv2-x achieves a mAPval50-95 of 54.3. Inference speeds are also competitive, making RTDETRv2 suitable for real-time applications, especially when using hardware acceleration like NVIDIA T4 GPUs.

### Strengths and Weaknesses

**Strengths:**

- **High Accuracy:** Transformer architecture provides excellent object detection accuracy.
- **Real-Time Capability:** Achieves competitive inference speeds suitable for real-time applications.
- **Robust Feature Extraction:** Vision Transformers effectively capture global context and intricate details.

**Weaknesses:**

- **Larger Model Size:** RTDETRv2 models, particularly larger variants like RTDETRv2-x, have a higher parameter count and FLOPs, requiring more computational resources.
- **Inference Speed:** While real-time, its inference speed may be slower than the fastest YOLO models, especially on less powerful devices.

### Ideal Use Cases

RTDETRv2 is best suited for applications where accuracy is paramount and sufficient computational resources are available. These include:

- **Autonomous Vehicles:** Demanding reliable and precise environmental perception. Explore more about [AI in self-driving cars](https://www.ultralytics.com/solutions/ai-in-self-driving).
- **Robotics:** Enabling precise object interaction and manipulation in complex environments. Learn about [AI's Role in Robotics](https://www.ultralytics.com/blog/from-algorithms-to-automation-ais-role-in-robotics).
- **Medical Imaging:** For accurate detection of anomalies, aiding in medical diagnostics. Discover [AI in Healthcare applications](https://www.ultralytics.com/solutions/ai-in-healthcare).
- **High-Resolution Image Analysis:** Applications requiring detailed analysis of large images like satellite or aerial imagery, as discussed in [Using Computer Vision to Analyse Satellite Imagery](https://www.ultralytics.com/blog/using-computer-vision-to-analyse-satellite-imagery).

[Learn more about RTDETRv2](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

## YOLOv9: Ultra-Efficient Real-Time Detection

**YOLOv9** ([You Only Look Once 9](https://docs.ultralytics.com/models/yolov9/)) is the latest iteration in the Ultralytics YOLO family, renowned for its exceptional speed and efficiency. Authored by Chien-Yao Wang and Hong-Yuan Mark Liao from the Institute of Information Science, Academia Sinica, Taiwan, YOLOv9 was introduced in their [Arxiv paper](https://arxiv.org/abs/2402.13616) on 2024-02-21. The official [YOLOv9 GitHub repository](https://github.com/WongKinYiu/yolov9) provides code and implementation details. YOLOv9 builds upon previous YOLO versions, focusing on maintaining real-time performance while enhancing accuracy and reducing model size.

### Architecture and Key Features

YOLOv9 introduces the concept of Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN). These innovations allow the model to learn more effectively and efficiently, leading to improved accuracy with fewer parameters and computations. This makes YOLOv9 exceptionally efficient for real-time object detection tasks.

### Performance Metrics

YOLOv9 achieves a remarkable balance between speed and accuracy. As shown in the comparison table, YOLOv9 models offer competitive mAP scores with significantly faster inference speeds and smaller model sizes compared to many other high-accuracy models. This efficiency makes it ideal for deployment on resource-constrained devices and in latency-critical applications.

### Strengths and Weaknesses

**Strengths:**

- **High Speed and Efficiency:** Optimized for extremely fast inference, crucial for real-time applications.
- **Smaller Model Size:** Requires less computational resources and memory, facilitating deployment on edge devices.
- **Good Accuracy:** Achieves a strong balance of speed and accuracy, often outperforming other real-time detectors in efficiency.

**Weaknesses:**

- **Accuracy Trade-off:** While highly accurate for its size and speed, it may not reach the absolute highest accuracy levels of larger, more computationally intensive models like RTDETRv2 in certain complex scenarios.

### Ideal Use Cases

YOLOv9 is ideally suited for applications where real-time detection and efficiency are paramount. These include:

- **Edge Computing:** Deployment on edge devices with limited computational resources, as explored in [Empowering Edge AI with Sony IMX500 and aitrios](https://www.ultralytics.com/blog/empowering-edge-ai-with-sony-imx500-and-aitrios).
- **Real-time Surveillance:** For efficient monitoring in security systems requiring immediate analysis. Learn about [Computer Vision for Theft Prevention](https://www.ultralytics.com/blog/computer-vision-for-theft-prevention-enhancing-security).
- **Robotics and Drones:** Applications needing fast perception for navigation and interaction. Discover [Computer Vision Applications in AI Drone Operations](https://www.ultralytics.com/blog/computer-vision-applications-ai-drone-uav-operations).
- **Mobile Applications:** Integrating object detection into mobile apps where model size and speed are critical.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

## Model Comparison Table

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv9t    | 640                   | 38.3                 | -                              | 2.3                                 | 2.0                | 7.7               |
| YOLOv9s    | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m    | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c    | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e    | 640                   | 55.6                 | -                              | 16.77                               | 57.3               | 189.0             |
|            |                       |                      |                                |                                     |                    |                   |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | 54.3                 | -                              | 15.03                               | 76                 | 259               |

## Conclusion

Both YOLOv9 and RTDETRv2 are powerful object detection models, yet they cater to different priorities. **RTDETRv2** is the optimal choice when maximum accuracy is paramount and computational resources are less of a constraint. **YOLOv9**, conversely, excels in scenarios requiring real-time performance, efficiency, and deployment on resource-limited platforms.

For users exploring other models, Ultralytics offers a diverse range, including:

- **YOLOv8 and YOLO11:** Previous and current generations of YOLO models, providing various speed-accuracy trade-offs, as highlighted in [Ultralytics YOLOv8 Turns One: A Year of Breakthroughs and Innovations](https://www.ultralytics.com/blog/ultralytics-yolov8-turns-one-a-year-of-breakthroughs-and-innovations) and [Ultralytics YOLO11 has Arrived](https://www.ultralytics.com/blog/ultralytics-yolo11-has-arrived-redefine-whats-possible-in-ai).
- **YOLO-NAS:** Models designed via Neural Architecture Search for optimal performance. Learn more about [YOLO-NAS by Deci AI](https://docs.ultralytics.com/models/yolo-nas/).
- **FastSAM and MobileSAM:** For real-time instance segmentation tasks. Check out [FastSAM documentation](https://docs.ultralytics.com/models/fast-sam/) and [MobileSAM documentation](https://docs.ultralytics.com/models/mobile-sam/).

Selecting between RTDETRv2 and YOLOv9, or other Ultralytics models, hinges on the specific demands of your computer vision project, balancing accuracy, speed, and available resources. Refer to the comprehensive [Ultralytics Documentation](https://docs.ultralytics.com/models/) and the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics) for in-depth information and implementation guides.