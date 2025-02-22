---
comments: true
description: Compare RTDETRv2 & YOLOX object detection models. Discover their strengths, performance, and use cases to choose the best model for your project.
keywords: RTDETRv2,YOLOX,object detection,model comparison,Vision Transformers,real-time detection,Yolo models,Ultralytics computer vision
---

# RTDETRv2 vs YOLOX: A Technical Comparison for Object Detection

Choosing the right object detection model is crucial for computer vision projects. Ultralytics offers a diverse range of models, including the YOLO series and the RT-DETR series, each with unique strengths. This page provides a detailed technical comparison between **RTDETRv2** and **YOLOX**, two state-of-the-art models for object detection, to assist you in making an informed decision based on your project requirements.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["RTDETRv2", "YOLOX"]'></canvas>

## RTDETRv2: High Accuracy Real-Time Detection

**RTDETRv2** ([Real-Time Detection Transformer v2](https://docs.ultralytics.com/models/rtdetr/)) is an advanced object detection model developed by Baidu, known for its high accuracy and real-time performance. Introduced on 2023-04-17 and detailed in its [Arxiv paper](https://arxiv.org/abs/2304.08069), RTDETRv2 utilizes a Vision Transformer (ViT) architecture to achieve state-of-the-art results. The [official implementation](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch) is available on GitHub.

### Architecture and Key Features

RTDETRv2's architecture is based on Vision Transformers, enabling it to capture global context within images through self-attention mechanisms. This transformer-based approach allows for robust feature extraction and precise object localization, particularly in complex scenes. Unlike traditional CNN-based models, RTDETRv2 excels in understanding the relationships between different parts of an image, leading to improved detection accuracy.

### Performance Metrics

RTDETRv2 models demonstrate impressive mAP scores, with larger variants like RTDETRv2-x achieving a mAPval50-95 of 54.3. While detailed CPU ONNX speed metrics are not provided in the table below, its TensorRT speeds are competitive, making it suitable for real-time applications on capable hardware such as NVIDIA T4 GPUs. For detailed performance metrics, refer to the [model comparison table](#model-comparison-table) below.

### Strengths and Weaknesses

**Strengths:**

- **Superior Accuracy:** Transformer architecture provides excellent object detection accuracy.
- **Real-Time Capable:** Achieves competitive inference speeds with hardware acceleration, suitable for real-time systems.
- **Effective Feature Extraction:** Vision Transformers capture global context and intricate details effectively.

**Weaknesses:**

- **Larger Model Size:** RTDETRv2 models, especially larger versions, have a higher parameter count and FLOPs, demanding more computational resources.
- **Inference Speed Limitations:** While real-time, it may not be as fast as highly optimized models like YOLOX on less powerful devices.

### Ideal Use Cases

RTDETRv2 is best suited for applications where accuracy is paramount and sufficient computational resources are available. Ideal use cases include:

- **Autonomous Vehicles:** For reliable and precise environmental perception in self-driving systems. [AI in self-driving cars](https://www.ultralytics.com/solutions/ai-in-self-driving)
- **Robotics:** Enabling robots to accurately perceive and interact with objects in complex environments. [From Algorithms to Automation: AI's Role in Robotics](https://www.ultralytics.com/blog/from-algorithms-to-automation-ais-role-in-robotics)
- **Medical Imaging:** For high-precision detection of anomalies in medical images, aiding in diagnostics. [AI in Healthcare](https://www.ultralytics.com/solutions/ai-in-healthcare)
- **High-Resolution Image Analysis:** Applications requiring detailed analysis of large images, such as satellite or aerial imagery. [Using Computer Vision to Analyse Satellite Imagery](https://www.ultralytics.com/blog/using-computer-vision-to-analyse-satellite-imagery)

[Learn more about RTDETRv2](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

## YOLOX: Efficient and Versatile Object Detection

**YOLOX** ([You Only Look Once X](https://github.com/Megvii-BaseDetection/YOLOX)) is an anchor-free object detection model developed by Megvii, known for its high performance and efficiency. Introduced on 2021-07-18 and detailed in its [Arxiv paper](https://arxiv.org/abs/2107.08430), YOLOX builds upon the YOLO series, offering a simplified design with state-of-the-art results. The [official documentation](https://yolox.readthedocs.io/en/latest/) provides comprehensive details.

### Architecture and Key Features

YOLOX adopts an anchor-free approach, eliminating the need for predefined anchor boxes, which simplifies the model and reduces hyperparameters. It features a decoupled head for classification and localization, enhancing training efficiency and accuracy. Advanced data augmentation techniques like MixUp and Mosaic are utilized to improve robustness. YOLOX is designed for high speed and efficiency, making it suitable for real-time applications and deployment on various hardware platforms.

### Performance Metrics

YOLOX offers a range of model sizes, from Nano to XLarge, catering to different computational budgets and accuracy needs. YOLOX models achieve a good balance of speed and accuracy. For example, YOLOX-s achieves a mAPval50-95 of 40.5 with fast inference speeds on TensorRT. Refer to the [model comparison table](#model-comparison-table) below for detailed performance metrics across different YOLOX variants.

### Strengths and Weaknesses

**Strengths:**

- **High Efficiency and Speed:** Optimized for fast inference, making it ideal for real-time applications.
- **Anchor-Free Design:** Simplifies the architecture and training process, improving generalization.
- **Versatile Model Sizes:** Offers a range of model sizes to suit different computational constraints.
- **Strong Performance:** Achieves a good balance between speed and accuracy.

**Weaknesses:**

- **Accuracy Trade-off:** While efficient, its accuracy may be slightly lower than transformer-based models like RTDETRv2 in complex scenarios.
- **Performance in Complex Scenes:** As a single-stage detector, it might be less robust in extremely crowded scenes compared to some two-stage detectors, although YOLOX mitigates this gap significantly compared to earlier YOLO versions.

### Ideal Use Cases

YOLOX is ideally suited for applications requiring real-time object detection with a focus on speed and efficiency. These include:

- **Robotics:** Real-time perception for robot navigation and interaction in dynamic environments. [AI in Robotics](https://www.ultralytics.com/solutions)
- **Surveillance Systems:** Efficient object detection in video streams for security and monitoring applications. [Computer Vision for Theft Prevention: Enhancing Security](https://www.ultralytics.com/blog/computer-vision-for-theft-prevention-enhancing-security)
- **Industrial Inspection:** Automated visual inspection on production lines for defect detection and quality control. [Improving Manufacturing with Computer Vision](https://www.ultralytics.com/blog/improving-manufacturing-with-computer-vision)
- **Edge Devices:** Deployment on resource-constrained devices where computational efficiency is critical. [Empowering Edge AI with Sony IMX500 and Aitrios](https://www.ultralytics.com/blog/empowering-edge-ai-with-sony-imx500-and-aitrios)

[Learn more about YOLOX](https://github.com/Megvii-BaseDetection/YOLOX){ .md-button }

## Model Comparison Table

| Model      | size<sup>(pixels) | mAP<sup>val<br>50-95 | Speed<sup>CPU ONNX<br>(ms) | Speed<sup>T4 TensorRT10<br>(ms) | params<sup>(M) | FLOPs<sup>(B) |
|------------|-------------------|----------------------|----------------------------|---------------------------------|----------------|---------------|
| RTDETRv2-s | 640               | 48.1                 | -                          | 5.03                            | 20             | 60            |
| RTDETRv2-m | 640               | 51.9                 | -                          | 7.51                            | 36             | 100           |
| RTDETRv2-l | 640               | 53.4                 | -                          | 9.76                            | 42             | 136           |
| RTDETRv2-x | 640               | 54.3                 | -                          | 15.03                           | 76             | 259           |
|            |                   |                      |                            |                                 |                |               |
| YOLOXnano  | 416               | 25.8                 | -                          | -                               | 0.91           | 1.08          |
| YOLOXtiny  | 416               | 32.8                 | -                          | -                               | 5.06           | 6.45          |
| YOLOXs     | 640               | 40.5                 | -                          | 2.56                            | 9.0            | 26.8          |
| YOLOXm     | 640               | 46.9                 | -                          | 5.43                            | 25.3           | 73.8          |
| YOLOXl     | 640               | 49.7                 | -                          | 9.04                            | 54.2           | 155.6         |
| YOLOXx     | 640               | 51.1                 | -                          | 16.1                            | 99.1           | 281.9         |

## Conclusion

Both RTDETRv2 and YOLOX are powerful object detection models, but they cater to different priorities. **RTDETRv2** is the superior choice when maximum accuracy is required and computational resources are not a limiting factor. **YOLOX**, conversely, excels in scenarios where real-time performance, efficiency, and deployment on less powerful hardware are critical.

For users exploring other options, Ultralytics offers a wide range of models, including:

- **YOLOv8 and YOLOv9:** Successors in the YOLO series, offering a spectrum of speed and accuracy trade-offs. [Ultralytics YOLOv8 Turns One: A Year of Breakthroughs and Innovations](https://www.ultralytics.com/blog/ultralytics-yolov8-turns-one-a-year-of-breakthroughs-and-innovations), [YOLOv9 Documentation](https://docs.ultralytics.com/models/yolov9/)
- **YOLO-NAS:** Models designed using Neural Architecture Search for optimal performance. [YOLO-NAS by Deci AI - a State-of-the-Art Object Detection Model](https://docs.ultralytics.com/models/yolo-nas/)
- **FastSAM and MobileSAM:** For real-time instance segmentation tasks. [FastSAM Documentation](https://docs.ultralytics.com/models/fast-sam/), [MobileSAM Documentation](https://docs.ultralytics.com/models/mobile-sam/)

The choice between RTDETRv2, YOLOX, and other Ultralytics models should be guided by the specific needs of your computer vision project, carefully balancing accuracy, speed, and available resources. Explore the [Ultralytics Documentation](https://docs.ultralytics.com/models/) and [GitHub repository](https://github.com/ultralytics/ultralytics) for more in-depth information and implementation details.
