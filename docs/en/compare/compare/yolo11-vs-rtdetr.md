---
description: Compare RTDETRv2's accuracy with YOLO11's speed in this detailed analysis of top object detection models. Decide the best fit for your projects.
keywords: RTDETRv2, YOLO11, object detection, Ultralytics, Vision Transformer, YOLO, computer vision, real-time detection, model comparison
---

# RTDETRv2 vs YOLO11: A Technical Comparison for Object Detection

Choosing the right object detection model is crucial for computer vision projects. Ultralytics offers a range of models, including the efficient YOLO series and the high-accuracy RT-DETR series. This page provides a detailed technical comparison between **YOLO11** and **RTDETRv2**, two state-of-the-art models for object detection, to help you make an informed decision.

<script async src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO11", "RTDETRv2"]'></canvas>

## RTDETRv2: High Accuracy Real-Time Detection

**RTDETRv2** ([Real-Time Detection Transformer v2](https://docs.ultralytics.com/models/rtdetr/)) is authored by Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu from Baidu, and was released on 2023-04-17. It is documented on [GitHub](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme) and further detailed in its [arXiv paper](https://arxiv.org/abs/2304.08069). RTDETRv2 is celebrated for its high accuracy and real-time performance in object detection.

### Architecture and Key Features

RTDETRv2's architecture is built upon a Vision Transformer (ViT), which allows the model to effectively capture global context within images. This transformer-based approach distinguishes it from traditional CNN-based models like YOLO, enabling RTDETRv2 to leverage self-attention mechanisms for enhanced feature extraction. This design choice leads to superior accuracy, especially in complex visual scenes where understanding the broader context is crucial for accurate object detection.

### Performance Metrics

RTDETRv2 models demonstrate strong performance in terms of accuracy, with the larger RTDETRv2-x variant achieving a mAPval50-95 of 54.3. When using TensorRT acceleration on NVIDIA T4 GPUs, RTDETRv2 maintains competitive inference speeds. For detailed performance metrics, refer to the comparison table below.

### Strengths and Weaknesses

**Strengths:**

- **Superior Accuracy**: Transformer architecture provides excellent object detection accuracy, particularly in complex scenarios.
- **Real-Time Capable**: Achieves competitive inference speeds, especially with GPU acceleration, making it suitable for real-time applications.
- **Robust Feature Extraction**: Vision Transformers effectively capture global context and detailed features.

**Weaknesses:**

- **Larger Model Size**: RTDETRv2 models, especially larger variants like RTDETRv2-x, have a considerable parameter count and higher FLOPs, requiring more computational resources compared to smaller YOLO models.
- **Inference Speed Limitations**: While real-time, the inference speed may be slower than the fastest YOLO models, especially on devices with limited resources.

### Ideal Use Cases

RTDETRv2 is ideally suited for applications where accuracy is paramount and sufficient computational resources are available, such as:

- **Autonomous Vehicles**: For precise and reliable environmental perception. Learn more about [AI in self-driving cars](https://www.ultralytics.com/solutions/ai-in-self-driving).
- **Robotics**: To enable robots to interact accurately in complex environments. Explore [AI's role in robotics](https://www.ultralytics.com/blog/from-algorithms-to-automation-ais-role-in-robotics).
- **Medical Imaging**: For detailed anomaly detection to aid in diagnostics. Discover [AI in Healthcare](https://www.ultralytics.com/solutions/ai-in-healthcare).
- **High-Resolution Image Analysis**: For applications like satellite imagery analysis and industrial inspection. Read about [analyzing satellite imagery with computer vision](https://www.ultralytics.com/blog/using-computer-vision-to-analyse-satellite-imagery).

[Learn more about RTDETRv2](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

## YOLO11: Efficient and Versatile Object Detection

**YOLO11** ([You Only Look Once 11](https://docs.ultralytics.com/models/yolo11/)) is the latest in the Ultralytics YOLO series, developed by Glenn Jocher and Jing Qiu from Ultralytics, released on 2024-09-27. Find more details in the [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/) and the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics). YOLO11 is engineered for speed and efficiency, building upon the strengths of previous YOLO iterations while enhancing accuracy and maintaining real-time performance.

### Architecture and Key Features

YOLO11 adopts a single-stage detection approach, prioritizing inference speed while achieving a strong balance with accuracy. It improves upon predecessors like [YOLOv8](https://docs.ultralytics.com/models/yolov8/) through architectural optimizations and enhancements. YOLO models are inherently designed for efficient processing, making them highly adaptable for real-time applications across various hardware platforms, including edge devices.

### Performance Metrics

YOLO11 excels in speed, as shown in the performance table below. Models like YOLO11n and YOLO11s achieve remarkable inference speeds on both CPUs and GPUs. While its mAP scores are slightly below those of larger RTDETRv2 models, YOLO11 provides competitive accuracy suitable for a wide array of object detection tasks.

### Strengths and Weaknesses

**Strengths:**

- **Exceptional Speed**: YOLO models are famous for their fast inference speeds, crucial for real-time applications.
- **High Efficiency**: YOLO11 models are computationally efficient, enabling deployment on resource-limited devices.
- **Versatile Application**: Suitable for a broad range of object detection tasks and deployment scenarios.
- **Small Model Size**: Smaller YOLO11 variants are memory-efficient due to their reduced parameter count.

**Weaknesses:**

- **Accuracy Trade-off**: In scenarios requiring the highest possible accuracy, particularly with complex or overlapping objects, larger models such as RTDETRv2 may offer superior performance.

### Ideal Use Cases

YOLO11's speed and efficiency make it ideal for applications requiring real-time processing and edge deployment:

- **Real-time Video Surveillance**: For rapid object detection in security systems. Explore [security alarm system projects with YOLOv8](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8).
- **Edge AI Applications**: For deployment on devices like mobile phones and IoT devices. Learn about [Edge AI with YOLOv8](https://www.ultralytics.com/blog/edge-ai-and-aiot-upgrade-any-camera-with-ultralytics-yolov8-in-a-no-code-way).
- **Robotics and Automation**: For real-time perception in robotic systems. Understand [AI in Robotics](https://www.ultralytics.com/glossary/robotics).
- **Retail Analytics**: For real-time customer and inventory analysis. Discover [AI for smarter retail inventory management](https://www.ultralytics.com/blog/ai-for-smarter-retail-inventory-management).
- **Traffic Management**: For real-time vehicle detection and traffic analysis. Read about [optimizing traffic management with YOLO11](https://www.ultralytics.com/blog/optimizingtraffic-management-with-ultralytics-yolo11).

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

## Model Comparison Table

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLO11n    | 640                   | 39.5                 | 56.1                           | 1.5                                 | 2.6                | 6.5               |
| YOLO11s    | 640                   | 47.0                 | 90.0                           | 2.5                                 | 9.4                | 21.5              |
| YOLO11m    | 640                   | 51.5                 | 183.2                          | 4.7                                 | 20.1               | 68.0              |
| YOLO11l    | 640                   | 53.4                 | 238.6                          | 6.2                                 | 25.3               | 86.9              |
| YOLO11x    | 640                   | 54.7                 | 462.8                          | 11.3                                | 56.9               | 194.9             |
|            |                       |                      |                                |                                     |                    |                   |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | 54.3                 | -                              | 15.03                               | 76                 | 259               |

## Conclusion

Both RTDETRv2 and YOLO11 are powerful object detection models tailored for different priorities. **RTDETRv2** is optimal when accuracy is the primary concern and computational resources are available, while **YOLO11** excels in scenarios where real-time performance, efficiency, and deployment on less powerful hardware are key.

For users considering other models, Ultralytics offers a wide range, including:

- **YOLOv8 and YOLOv9**: Previous YOLO generations, offering various speed-accuracy trade-offs. Read about [YOLOv8's first year](https://www.ultralytics.com/blog/ultralytics-yolov8-turns-one-a-year-of-breakthroughs-and-innovations) and explore [YOLOv9](https://docs.ultralytics.com/models/yolov9/).
- **YOLO-NAS**: Models designed via Neural Architecture Search for optimized performance. Learn about [YOLO-NAS by Deci AI](https://docs.ultralytics.com/models/yolo-nas/).
- **FastSAM and MobileSAM**: For real-time instance segmentation. Check out [FastSAM](https://docs.ultralytics.com/models/fast-sam/) and [MobileSAM](https://docs.ultralytics.com/models/mobile-sam/).

The choice between RTDETRv2, YOLO11, and other Ultralytics models should be based on the specific needs of your computer vision project, carefully balancing accuracy, speed, and resource constraints. Refer to the [Ultralytics Documentation](https://docs.ultralytics.com/models/) and the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics) for further details and implementation guides.