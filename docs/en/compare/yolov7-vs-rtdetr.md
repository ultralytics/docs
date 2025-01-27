---
comments: true
description: Technical comparison of YOLOv7 and RT-DETR object detection models, including architecture, performance, and use cases.
keywords: YOLOv7, RT-DETR, object detection, model comparison, computer vision, Ultralytics
---

# YOLOv7 vs RT-DETR: A Detailed Model Comparison

Choosing the right object detection model is crucial for computer vision projects. This page provides a technical comparison between YOLOv7 and RT-DETR, two state-of-the-art models, to help you make an informed decision. We delve into their architectural differences, performance metrics, and ideal applications.

<script async src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv7", "RTDETRv2"]'></canvas>

## YOLOv7: The Real-time Object Detector

Ultralytics YOLOv7 is renowned for its **speed and efficiency** in object detection tasks. It builds upon the successes of previous YOLO versions, offering a refined architecture focused on maximizing inference speed without significant compromise on accuracy.

### Architecture and Key Features

YOLOv7 employs an **efficient network architecture** with techniques like:

- **E-ELAN:** Extended Efficient Layer Aggregation Network for more effective feature extraction.
- **Model Scaling:** Compound scaling methods to adjust model depth and width for different performance requirements.
- **Auxiliary Head Training:** Utilizing auxiliary loss heads during training for improved accuracy.

These features contribute to YOLOv7's ability to achieve high performance in real-time object detection scenarios.

### Performance Metrics

YOLOv7 achieves a balance between speed and accuracy, making it suitable for applications where latency is critical. Key performance indicators include:

- **mAP<sup>val</sup><sub>50-95</sub>**: Up to 53.1%
- **Inference Speed (T4 TensorRT10)**: As low as 6.84 ms
- **Model Size (parameters)**: Starting from 36.9M

### Use Cases and Strengths

YOLOv7 excels in applications demanding **real-time object detection**, such as:

- **Robotics:** For fast perception in robotic systems.
- **Surveillance:** Real-time monitoring in security systems.
- **Edge Devices:** Deployment on resource-constrained devices requiring efficient inference.

Its strength lies in its speed and relatively small model size, making it deployable in diverse environments.

[Learn more about YOLOv7](https://docs.ultralytics.com/models/yolov7/){ .md-button }

## RT-DETR: Accuracy with Transformers

RT-DETR, standing for Real-Time DEtection TRansformer, represents a different approach to object detection by leveraging **Vision Transformers (ViT)**. Unlike YOLO's CNN-based architecture, RT-DETR utilizes transformers to capture global context within images, potentially leading to higher accuracy.

### Architecture and Key Features

RT-DETR's architecture is characterized by:

- **Transformer Encoder:** Employing a transformer encoder to process the entire image and capture long-range dependencies.
- **Hybrid CNN Feature Extraction:** Combining CNNs for initial feature extraction with transformer layers for global context.
- **Anchor-Free Detection:** Eliminating the need for predefined anchor boxes, simplifying the detection process.

This transformer-based design allows RT-DETR to potentially achieve higher accuracy, particularly in complex scenes. To learn more about Vision Transformers, refer to our explanation on [Vision Transformer (ViT)](https://www.ultralytics.com/glossary/vision-transformer-vit).

### Performance Metrics

RT-DETR prioritizes accuracy and offers competitive performance metrics:

- **mAP<sup>val</sup><sub>50-95</sub>**: Up to 54.3%
- **Inference Speed (T4 TensorRT10)**: Starting from 5.03 ms
- **Model Size (parameters)**: Starting from 20M

### Use Cases and Strengths

RT-DETR is well-suited for applications where **high accuracy** is paramount, even if it means slightly higher computational cost compared to YOLOv7 in some configurations:

- **Autonomous Driving:** Precise object detection for safety-critical applications. Explore how [AI in self-driving cars](https://www.ultralytics.com/solutions/ai-in-self-driving) relies on accurate models.
- **Medical Imaging:** Detailed analysis in [medical image analysis](https://www.ultralytics.com/glossary/medical-image-analysis) where precision is crucial.
- **High-Resolution Imagery:** Object detection in high-resolution images, potentially leveraging SAHI Tiled Inference as described in our [SAHI Tiled Inference](https://docs.ultralytics.com/guides/sahi-tiled-inference/) guide.

RT-DETR’s strength is in its transformer architecture enabling it to achieve higher accuracy in complex scenarios.

[Learn more about RT-DETR](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

## Model Comparison Table

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv7l    | 640                   | 51.4                 | -                              | 6.84                                | 36.9               | 104.7             |
| YOLOv7x    | 640                   | 53.1                 | -                              | 11.57                               | 71.3               | 189.9             |
|            |                       |                      |                                |                                     |                    |                   |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | 54.3                 | -                              | 15.03                               | 76                 | 259               |

## Conclusion

Both YOLOv7 and RT-DETR are powerful object detection models, each with its strengths. YOLOv7 prioritizes **speed and efficiency**, making it ideal for real-time applications and resource-constrained environments. RT-DETR, with its transformer-based architecture, aims for **higher accuracy**, suitable for applications where precision is paramount.

Consider your project's specific requirements when choosing between these models. If speed is the primary concern, YOLOv7 is an excellent choice. If accuracy is more critical, RT-DETR's transformer architecture may be advantageous.

For users interested in exploring other models, Ultralytics offers a range of options including:

- **YOLOv8:** The latest iteration in the YOLO series, balancing speed and accuracy. Explore [Ultralytics YOLOv8](https://www.ultralytics.com/yolo).
- **YOLOv9 & YOLOv10:** Cutting-edge models pushing the boundaries of real-time object detection. Learn more about [YOLOv9](https://docs.ultralytics.com/models/yolov9/) and [YOLOv10](https://docs.ultralytics.com/models/yolov10/).
- **YOLO-NAS:** Models optimized through Neural Architecture Search for enhanced performance. Discover [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/).
- **YOLOv6:** Another high-performance object detector focusing on speed and efficiency. Explore [YOLOv6](https://docs.ultralytics.com/models/yolov6/).

Explore our [Ultralytics documentation](https://docs.ultralytics.com/models/) to discover the full range of models and choose the best fit for your computer vision needs. You can also visit our [Ultralytics HUB](https://www.ultralytics.com/hub) for easy training and deployment of YOLO models.
