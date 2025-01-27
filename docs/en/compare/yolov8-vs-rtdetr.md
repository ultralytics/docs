---
comments: true
description: Compare YOLOv8 and RTDETRv2 for object detection. Explore their architectures, performance, use cases, and choose the right model for your needs.
keywords: YOLOv8, RTDETRv2, object detection, model comparison, AI models, computer vision, real-time detection, Vision Transformer, Ultralytics
---

# Model Comparison: YOLOv8 vs RTDETRv2 for Object Detection

When selecting a computer vision model for object detection tasks, understanding the nuances between different architectures is crucial. This page provides a detailed technical comparison between Ultralytics YOLOv8 and RTDETRv2, two state-of-the-art models in the field. We will delve into their architectural differences, performance metrics, ideal use cases, and discuss their respective strengths and weaknesses to guide you in choosing the right model for your project.

<script async src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv8", "RTDETRv2"]'></canvas>

## YOLOv8: Streamlined Efficiency and Versatility

Ultralytics YOLOv8 is the cutting-edge successor in the YOLO family, known for its exceptional blend of speed and accuracy in object detection. It's designed for ease of use and adaptability across various applications.

### Architecture and Key Features

YOLOv8 builds upon the foundational principles of previous YOLO versions but introduces several architectural enhancements. It leverages a streamlined single-stage detector design, focusing on efficiency without sacrificing accuracy. Key features include a flexible backbone, an anchor-free detection head, and an optimized loss function. This architecture allows for fast inference speeds, making it suitable for real-time applications. For a deeper dive into YOLO models, explore the [Comprehensive Tutorials to Ultralytics YOLO](https://docs.ultralytics.com/guides/).

### Performance Metrics

YOLOv8 achieves impressive performance across different model sizes. The model variants, from YOLOv8n (nano) to YOLOv8x (extra-large), offer a scalable range of performance and computational requirements. For detailed metrics and understanding of performance evaluation, refer to the [YOLO Performance Metrics guide](https://docs.ultralytics.com/guides/yolo-performance-metrics/).

### Use Cases

YOLOv8's versatility makes it applicable to a wide array of use cases, from [object detection in agriculture](https://www.ultralytics.com/solutions/ai-in-agriculture) to [enhancing security systems](https://www.ultralytics.com/blog/computer-vision-for-theft-prevention-enhancing-security). Its speed and accuracy are particularly beneficial in real-time scenarios such as robotics, autonomous vehicles ([AI in self-driving cars](https://www.ultralytics.com/solutions/ai-in-self-driving)), and industrial automation ([AI in manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing)). Explore real-world applications and use cases on the [Ultralytics Solutions](https://www.ultralytics.com/solutions) page.

### Strengths and Weaknesses

**Strengths:**

- **Speed:** YOLOv8 is optimized for fast inference, crucial for real-time applications.
- **Accuracy:** It maintains high accuracy in object detection, balancing speed and precision effectively.
- **Ease of Use:** Ultralytics provides excellent documentation and a user-friendly Python package, simplifying implementation.
- **Scalability:** Offers various model sizes (n, s, m, l, x) to suit different computational constraints.

**Weaknesses:**

- **Single-Stage Limitations:** As a single-stage detector, it might be less accurate than two-stage detectors in complex scenes with overlapping objects, although YOLOv8 significantly bridges this gap.
- **Hyperparameter Tuning:** Achieving optimal performance may require careful hyperparameter tuning, as detailed in the [Hyperparameter Tuning Guide](https://docs.ultralytics.com/guides/hyperparameter-tuning/).

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## RTDETRv2: Real-Time DEtection Transformer v2

RTDETRv2, standing for Real-Time Detection Transformer version 2, represents a different approach to object detection, leveraging the power of Vision Transformers (ViT) for real-time performance.

### Architecture and Key Features

Unlike YOLO's CNN-centric architecture, RTDETRv2 is built upon a Transformer-based structure. This allows the model to capture global context in images more effectively, potentially leading to better understanding of complex scenes. It uses a hybrid efficient encoder, transformer decoder layers, and is designed for low latency inference. To understand more about the underlying technology, see our explanation of [Vision Transformer (ViT)](https://www.ultralytics.com/glossary/vision-transformer-vit).

### Performance Metrics

RTDETRv2 models are designed to offer a strong balance between accuracy and latency. While detailed CPU ONNX speed metrics are not provided in the table, the TensorRT speeds indicate its real-time capability, especially when utilizing hardware acceleration.

### Use Cases

RTDETRv2 is well-suited for applications where understanding the broader context of an image is important for accurate detection. This can include scenarios like autonomous driving, advanced robotics, and complex scene analysis in security or surveillance. For instance, in [AI in aviation](https://www.ultralytics.com/blog/ai-in-aviation-a-runway-to-smarter-airports), RTDETRv2 could be used for enhanced airport monitoring.

### Strengths and Weaknesses

**Strengths:**

- **Global Context Understanding:** Transformer architecture enables better capture of global context, potentially improving accuracy in complex scenes.
- **Real-Time Performance:** Optimized for real-time inference, particularly with TensorRT acceleration.
- **Strong Accuracy:** Achieves competitive mAP scores, demonstrating high detection accuracy.

**Weaknesses:**

- **Computational Cost:** Transformers can be computationally intensive, potentially requiring more resources compared to some CNN-based models, although RTDETR aims to mitigate this.
- **Complexity:** Transformer-based models can be more complex to implement and train compared to simpler architectures.

[Learn more about RTDETRv2](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

## Model Comparison Table

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- | ---- |
| YOLOv8n    | 640                   | 37.3                 | 80.4                           | 1.47                                | 3.2                | 8.7               |
| YOLOv8s    | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m    | 640                   | 50.2                 | 234.7                          | 5.86                                | 5.86               | 25.9              | 78.9 |
| YOLOv8l    | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x    | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |
|            |                       |                      |                                |                                     |                    |                   |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | 54.3                 | -                              | 15.03                               | 76                 | 259               |

## Conclusion

Both YOLOv8 and RTDETRv2 are powerful object detection models, each with unique strengths. YOLOv8 excels in speed and ease of use, making it ideal for a wide range of real-time applications. RTDETRv2, with its Transformer architecture, offers enhanced contextual understanding and strong accuracy, suitable for complex scene analysis.

Your choice between YOLOv8 and RTDETRv2 will depend on the specific requirements of your project, including the importance of speed versus accuracy, computational resources, and the complexity of the scenes being analyzed. For users interested in exploring other models, Ultralytics also provides access to YOLOv5, YOLOv7, YOLOv9, and YOLO-NAS, each offering different trade-offs between performance and efficiency. Explore the full range of [Ultralytics Models](https://docs.ultralytics.com/models/).
