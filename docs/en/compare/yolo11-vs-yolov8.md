---
comments: true
description: Technical comparison of YOLO11 and YOLOv8 object detection models, including architecture, performance metrics like mAP and inference speed, and use cases.
keywords: YOLO11, YOLOv8, object detection, model comparison, Ultralytics, AI, computer vision, performance, architecture, use cases
---

# YOLO11 vs YOLOv8: A Technical Comparison

Ultralytics YOLO models are renowned for their cutting-edge performance in object detection tasks. This page provides a detailed technical comparison between two prominent models in the YOLO family: [YOLO11](https://docs.ultralytics.com/models/yolo11/) and [YOLOv8](https://docs.ultralytics.com/models/yolov8/). We delve into their architectural nuances, performance benchmarks, and suitability for various real-world applications.

<script async src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO11", "YOLOv8"]'></canvas>

## YOLO11

[YOLO11](https://docs.ultralytics.com/models/yolo11/), the latest iteration in the Ultralytics YOLO series, represents a significant leap forward in object detection technology. Building upon the strengths of its predecessors, YOLO11 is engineered for enhanced precision and efficiency, making it ideal for demanding computer vision applications.

### Architecture

YOLO11 introduces several architectural refinements focused on improving feature extraction and processing speed. It leverages an optimized backbone network for more effective feature learning, allowing the model to capture finer details and contextual information within images. This leads to more accurate object detection, especially in complex scenes. Furthermore, YOLO11 models are designed with parameter efficiency in mind, achieving higher accuracy with a relatively smaller model size compared to previous generations.

### Performance Metrics

YOLO11 demonstrates superior performance across various metrics, particularly in accuracy. As shown in the comparison table below, YOLO11 models generally achieve higher mAP (mean Average Precision) values than their YOLOv8 counterparts, indicating improved detection accuracy. This accuracy enhancement comes with competitive inference speeds, ensuring real-time capabilities are maintained for many applications. The model size remains efficient, allowing for deployment on resource-constrained devices.

### Use Cases

Due to its focus on accuracy and efficient performance, YOLO11 is particularly well-suited for applications where precision is paramount. Ideal use cases include:

- **Medical Imaging**: [YOLO11 in hospitals](https://www.ultralytics.com/blog/ultralytics-yolo11-in-hospitals-advancing-healthcare-with-computer-vision) can enhance diagnostic accuracy in medical image analysis.
- **Quality Control in Manufacturing**: [AI in manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing) benefits from YOLO11's precision in defect detection and product inspection.
- **Advanced Security Systems**: For applications like [computer vision for theft prevention](https://www.ultralytics.com/blog/computer-vision-for-theft-prevention-enhancing-security), YOLO11's accuracy is critical for reliable threat detection.
- **Robotics**: In robotics applications requiring precise environmental understanding, YOLO11 provides the necessary accuracy for tasks like navigation and manipulation.

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

## YOLOv8

[YOLOv8](https://docs.ultralytics.com/models/yolov8/) is a state-of-the-art object detection model that achieves an excellent balance between speed and accuracy. It is designed to be versatile and user-friendly, making it a popular choice for a wide range of computer vision tasks.

### Architecture

YOLOv8's architecture is designed for efficiency and scalability. It incorporates a streamlined network structure that allows for fast inference times without significantly compromising accuracy. This model is built to be easily adaptable and customizable, catering to diverse object detection needs. YOLOv8 supports various model sizes, from YOLOv8n (nano) for edge devices to YOLOv8x (extra-large) for high-performance servers, providing flexibility for different computational resources.

### Performance Metrics

YOLOv8 excels in providing a balanced performance profile. It offers impressive inference speeds, making it suitable for real-time object detection applications. While slightly less accurate than YOLO11 in terms of mAP, YOLOv8 maintains a strong performance level while being notably faster in many configurations. Its efficient model size and speed make it deployable across a broad spectrum of hardware, from CPUs to GPUs and edge devices.

### Use Cases

YOLOv8’s versatility and balanced performance make it suitable for numerous applications, including:

- **Real-time Object Detection**: Its speed makes [YOLOv8 for object detection](https://www.ultralytics.com/blog/object-detection-with-a-pre-trained-ultralytics-yolov8-model) ideal for applications like autonomous vehicles and real-time surveillance.
- **Retail Analytics**: [Building intelligent stores with YOLOv8](https://www.ultralytics.com/event/build-intelligent-stores-with-ultralytics-yolov8-and-seeed-studio) benefits from YOLOv8's ability to quickly analyze customer traffic and inventory.
- **Environmental Monitoring**: For tasks like [protecting biodiversity with YOLOv8](https://www.ultralytics.com/blog/protecting-biodiversity-the-kashmir-world-foundations-success-story-with-yolov5-and-yolov8), YOLOv8 provides a robust and efficient solution for wildlife detection.
- **Queue Management**: [Revolutionizing queue management with YOLOv8](https://www.ultralytics.com/blog/revolutionizing-queue-management-with-ultralytics-yolov8-and-openvino) in public spaces and retail environments.

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## Model Comparison Table

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLO11n | 640                   | 39.5                 | 56.1                           | 1.5                                 | 2.6                | 6.5               |
| YOLO11s | 640                   | 47.0                 | 90.0                           | 2.5                                 | 9.4                | 21.5              |
| YOLO11m | 640                   | 51.5                 | 183.2                          | 4.7                                 | 20.1               | 68.0              |
| YOLO11l | 640                   | 53.4                 | 238.6                          | 6.2                                 | 25.3               | 86.9              |
| YOLO11x | 640                   | 54.7                 | 462.8                          | 11.3                                | 56.9               | 194.9             |
|         |                       |                      |                                |                                     |                    |                   |
| YOLOv8n | 640                   | 37.3                 | 80.4                           | 1.47                                | 3.2                | 8.7               |
| YOLOv8s | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |

## Conclusion

Choosing between YOLO11 and YOLOv8 depends on the specific requirements of your project. If accuracy is the top priority and computational resources are sufficient, YOLO11 is the superior choice. Its enhanced architecture delivers higher precision, making it ideal for critical applications like medical imaging and security.

However, if speed and versatility are more crucial, or if deployment on lower-powered devices is necessary, YOLOv8 provides an excellent balance of performance. Its efficient design ensures real-time object detection capabilities across a wide range of applications.

For users interested in exploring other models, Ultralytics also offers a range of [YOLO models](https://docs.ultralytics.com/models/) including [YOLOv9](https://docs.ultralytics.com/models/yolov9/), [YOLOv10](https://docs.ultralytics.com/models/yolov10/), [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/), and [RT-DETR](https://docs.ultralytics.com/models/rtdetr/), each optimized for different aspects of object detection tasks.

For further details and implementation guides, refer to the [Ultralytics YOLO Documentation](https://docs.ultralytics.com/guides/) and the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics).
