---
comments: true
description: Discover the technical differences, performance benchmarks, and use cases of YOLOv8 and YOLOv9 to help you choose the best object detection model.
keywords: YOLOv8, YOLOv9, object detection, AI models comparison, computer vision, YOLO performance benchmarks, deep learning, Ultralytics models
---

# YOLOv8 vs YOLOv9: A Technical Comparison for Object Detection

Ultralytics YOLO models are at the forefront of real-time object detection, offering cutting-edge solutions for a wide array of computer vision tasks. This page provides a detailed technical comparison between two of our state-of-the-art models: [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) and [Ultralytics YOLOv9](https://docs.ultralytics.com/models/yolov9/). We will analyze their architectural differences, performance benchmarks, and ideal use cases to help you choose the best model for your specific needs.

<script async src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv8", "YOLOv9"]'></canvas>

## Architectural Overview

**YOLOv8** builds upon the successes of previous YOLO versions, introducing a refined architecture that balances speed and accuracy. It incorporates a flexible backbone, an anchor-free detection head, and a new loss function. This design makes YOLOv8 highly versatile and efficient for various object detection tasks. For more in-depth information, refer to the [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/) and the [Ultralytics YOLOv8 GitHub repository](https://github.com/ultralytics/ultralytics).

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

**YOLOv9** represents the next evolution, focusing on enhanced accuracy and efficiency through architectural innovations like Programmable Gradient Information (PGI) and Generalized Efficient Layer Aggregation Network (GELAN). These advancements allow YOLOv9 to achieve superior performance with fewer parameters, making it a powerful option for demanding applications. Detailed architectural specifics can be found in the [Ultralytics YOLOv9 documentation](https://docs.ultralytics.com/models/yolov9/).

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

## Performance Metrics

The table below summarizes the performance metrics of YOLOv8 and YOLOv9 models, highlighting key indicators for comparison.

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv8n | 640                   | 37.3                 | 80.4                           | 1.47                                | 3.2                | 8.7               |
| YOLOv8s | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |
|         |                       |                      |                                |                                     |                    |                   |
| YOLOv9t | 640                   | 38.3                 | -                              | 2.3                                 | 2.0                | 7.7               |
| YOLOv9s | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e | 640                   | 55.6                 | -                              | 16.77                               | 57.3               | 189.0             |

**mAP (Mean Average Precision):** YOLOv9 generally outperforms YOLOv8 in mAP across comparable model sizes, indicating higher accuracy in object detection. For instance, YOLOv9e achieves a significantly higher mAP of 55.6 compared to YOLOv8x's 53.9. Understanding [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) is crucial for evaluating object detection models.

**Inference Speed:** YOLOv8 demonstrates faster inference speeds on CPU ONNX in smaller models like YOLOv8n and YOLOv8s. However, with T4 TensorRT10, YOLOv9 models show competitive speeds, especially considering their higher accuracy. Optimizing [inference latency](https://www.ultralytics.com/glossary/inference-latency) is vital for real-time applications.

**Model Size and FLOPs:** YOLOv9 models achieve better performance with a comparable or even smaller number of parameters (params) and Floating Point Operations (FLOPs). This efficiency makes YOLOv9 models more resource-friendly for deployment on edge devices or in resource-constrained environments. Model [pruning](https://www.ultralytics.com/glossary/pruning) and [quantization](https://www.ultralytics.com/glossary/model-quantization) can further optimize model size and speed.

## Strengths and Weaknesses

**YOLOv8 Strengths:**

- **Versatility:** YOLOv8's architecture is adaptable to a wide range of tasks beyond object detection, including [image segmentation](https://docs.ultralytics.com/tasks/segment/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), and [image classification](https://docs.ultralytics.com/tasks/classify/).
- **Speed:** YOLOv8 offers excellent inference speed, particularly in its smaller variants, making it suitable for real-time applications like [security alarm systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8).
- **Ease of Use:** Ultralytics YOLOv8 is known for its user-friendly interface and straightforward implementation, making it accessible for both beginners and experts. You can get started quickly with pre-trained models and fine-tune them on your [custom datasets](https://www.ultralytics.com/blog/training-custom-datasets-with-ultralytics-yolov8-in-google-colab).

**YOLOv8 Weaknesses:**

- **Accuracy Gap:** While highly accurate, YOLOv8 may be slightly less precise than YOLOv9 in scenarios demanding the highest possible mAP.
- **Computational Cost for Larger Models:** Larger YOLOv8 models like YOLOv8x can be computationally intensive compared to the more efficient YOLOv9 counterparts.

**YOLOv9 Strengths:**

- **Superior Accuracy:** YOLOv9 excels in object detection accuracy, achieving higher mAP scores, which is crucial for applications requiring precise object recognition, such as in [medical image analysis](https://www.ultralytics.com/glossary/medical-image-analysis) or [satellite image analysis](https://www.ultralytics.com/blog/using-computer-vision-to-analyse-satellite-imagery).
- **Efficiency:** YOLOv9 achieves state-of-the-art performance with fewer parameters and FLOPs, making it more efficient in terms of computational resources. This is beneficial for deployment on [edge devices like NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/) or [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/).
- **Advanced Architecture:** Innovations like PGI and GELAN contribute to YOLOv9's enhanced learning capability and feature extraction, leading to better overall performance.

**YOLOv9 Weaknesses:**

- **Newer Model:** As a more recent model, YOLOv9 might have a smaller community and fewer readily available resources compared to the more established YOLOv8.
- **Inference Speed in Smaller Models:** While competitive, the CPU ONNX speed in smaller YOLOv9 models isn't as fast as YOLOv8's smaller counterparts.

## Use Cases

**YOLOv8 Use Cases:**

- **Real-time Object Detection:** Ideal for applications requiring rapid detection, such as [real-time object detection with webcam](https://www.ultralytics.com/blog/object-detection-with-a-pre-trained-ultralytics-yolov8-model), [autonomous vehicles](https://www.ultralytics.com/solutions/ai-in-self-driving), and [robotics](https://www.ultralytics.com/glossary/robotics).
- **Versatile Vision AI Applications:** Suitable for projects needing a balance of speed and accuracy across various tasks like segmentation and pose estimation, such as [pose estimation with Ultralytics YOLOv8](https://www.ultralytics.com/blog/pose-estimation-with-ultralytics-yolov8).
- **Resource-Constrained Environments:** Smaller YOLOv8 models are efficient for deployment on edge devices or mobile platforms where computational resources are limited.

**YOLOv9 Use Cases:**

- **High-Accuracy Object Detection:** Best suited for applications where accuracy is paramount, such as [quality inspection in manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing), [medical diagnostics](https://www.ultralytics.com/solutions/ai-in-healthcare), and [precision agriculture](https://www.ultralytics.com/solutions/ai-in-agriculture).
- **Efficient Deployment on Edge:** Despite its high accuracy, YOLOv9's efficiency allows for effective deployment on edge devices, enabling advanced AI capabilities in resource-limited scenarios.
- **Demanding Datasets:** Excels in complex scenarios and datasets where subtle features require advanced feature extraction and learning.

## Conclusion

Choosing between YOLOv8 and YOLOv9 depends on your project priorities. If speed and versatility across different vision tasks are key, YOLOv8 is an excellent choice. For applications demanding the highest accuracy and efficiency, especially in resource-constrained environments, YOLOv9 offers a cutting-edge solution.

Beyond YOLOv8 and YOLOv9, Ultralytics offers a range of YOLO models, including [YOLOv5](https://docs.ultralytics.com/models/yolov5/), [YOLOv7](https://docs.ultralytics.com/models/yolov7/), [YOLOv10](https://docs.ultralytics.com/models/yolov10/), [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/), and [RT-DETR](https://docs.ultralytics.com/models/rtdetr/). Explore our [Ultralytics Docs](https://docs.ultralytics.com/) to find the perfect model for your Vision AI needs.

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv8n | 640                   | 37.3                 | 80.4                           | 1.47                                | 3.2                | 8.7               |
| YOLOv8s | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |
|         |                       |                      |                                |                                     |                    |                   |
| YOLOv9t | 640                   | 38.3                 | -                              | 2.3                                 | 2.0                | 7.7               |
| YOLOv9s | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e | 640                   | 55.6                 | -                              | 16.77                               | 57.3               | 189.0             |
