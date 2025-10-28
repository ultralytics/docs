---
comments: true
description: Compare RTDETRv2 and YOLOv8 for object detection. Explore architecture, performance, and use cases to select the best model for your needs.
keywords: RTDETRv2, YOLOv8, object detection, computer vision, model comparison, deep learning, transformer architecture, real-time AI, Ultralytics
---

# RTDETRv2 vs YOLOv8: A Technical Comparison

Choosing the right object detection model is a critical decision that balances accuracy, speed, and computational cost. This page provides a detailed technical comparison between two powerful architectures: RTDETRv2, a transformer-based model from Baidu, and [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/), a state-of-the-art convolutional neural network (CNN) model. We will delve into their architectural differences, performance metrics, and ideal use cases to help you select the best model for your project.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["RTDETRv2", "YOLOv8"]'></canvas>

## RTDETRv2: Real-Time Detection Transformer v2

RTDETRv2 (Real-Time Detection Transformer v2) is a state-of-the-art object detector that leverages the power of Vision Transformers to achieve high accuracy while maintaining real-time performance. It represents an evolution of the original DETR (DEtection TRansformer) architecture, optimized for speed.

**Authors:** Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu  
**Organization:** [Baidu](https://www.baidu.com/)  
**Date:** 2023-04-17 (Initial RT-DETR), 2024-07-24 (RT-DETRv2 improvements)  
**Arxiv:** <https://arxiv.org/abs/2304.08069>, <https://arxiv.org/abs/2407.17140>  
**GitHub:** <https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch>  
**Docs:** <https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme>

### Architecture

RTDETRv2 employs a hybrid architecture that combines a conventional CNN backbone with a [Transformer](https://www.ultralytics.com/glossary/transformer)-based encoder-decoder. The CNN backbone extracts initial feature maps, which are then fed into the transformer. The transformer's [self-attention mechanism](https://www.ultralytics.com/glossary/self-attention) allows the model to capture global relationships between different parts of an image. This global context understanding is a key differentiator from purely CNN-based models and enables RTDETRv2 to excel at detecting objects in complex and cluttered scenes.

### Strengths

- **High Accuracy:** The transformer architecture allows RTDETRv2 to achieve excellent [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) scores, particularly on datasets with dense or small objects where global context is beneficial.
- **Robust Feature Extraction:** By processing the entire image context at once, it can better handle occlusions and complex object relationships.
- **Real-Time on GPU:** When accelerated with tools like [NVIDIA TensorRT](https://docs.ultralytics.com/integrations/tensorrt/), RTDETRv2 can achieve real-time inference speeds on high-end GPUs.

### Weaknesses

- **High Computational Cost:** Transformer-based models are notoriously resource-intensive. RTDETRv2 has a higher parameter count and FLOPs compared to YOLOv8, demanding more powerful hardware.
- **Slow Training and High Memory Usage:** Training transformers is computationally expensive and slow. They often require significantly more CUDA memory than CNN-based models like YOLOv8, making them inaccessible for users with limited hardware resources.
- **Slower CPU Inference:** While fast on GPUs, its performance on CPUs is significantly lower than that of highly optimized CNNs like YOLOv8.
- **Limited Ecosystem:** RTDETRv2 lacks the extensive, unified ecosystem provided by Ultralytics. This includes fewer integrations, less comprehensive [documentation](https://docs.ultralytics.com/), and a smaller community for support.

[Learn more about RT-DETR](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

## Ultralytics YOLOv8: Speed, Versatility, and Ease of Use

[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) is the latest iteration in the highly successful YOLO (You Only Look Once) family. Developed by Ultralytics, it sets a new standard for speed, accuracy, and ease of use, making it a top choice for a wide range of computer vision tasks.

**Authors:** Glenn Jocher, Ayush Chaurasia, and Jing Qiu  
**Organization:** [Ultralytics](https://www.ultralytics.com/)  
**Date:** 2023-01-10  
**GitHub:** <https://github.com/ultralytics/ultralytics>  
**Docs:** <https://docs.ultralytics.com/models/yolov8/>

### Architecture

YOLOv8 features a state-of-the-art, [anchor-free](https://www.ultralytics.com/glossary/anchor-free-detectors), single-stage architecture. It utilizes a novel CSP (Cross Stage Partial) backbone and a decoupled head, which separates the classification and regression tasks for improved accuracy. The entire architecture is highly optimized for an exceptional balance between performance and efficiency, enabling it to run on a wide spectrum of hardware, from powerful cloud GPUs to resource-constrained [edge devices](https://www.ultralytics.com/glossary/edge-ai).

### Strengths

- **Performance Balance:** YOLOv8 offers an outstanding trade-off between speed and accuracy, making it suitable for diverse real-world applications where both metrics are critical.
- **Versatility:** Unlike RTDETRv2, which is primarily an object detector, YOLOv8 is a multi-task framework that natively supports [object detection](https://docs.ultralytics.com/tasks/detect/), [instance segmentation](https://docs.ultralytics.com/tasks/segment/), [image classification](https://docs.ultralytics.com/tasks/classify/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), and [oriented object detection (OBB)](https://docs.ultralytics.com/tasks/obb/).
- **Ease of Use:** YOLOv8 is designed for a streamlined user experience with a simple [Python API](https://docs.ultralytics.com/usage/python/) and powerful [CLI](https://docs.ultralytics.com/usage/cli/). The extensive documentation and active community make it easy for developers to get started.
- **Training Efficiency and Low Memory:** YOLOv8 trains significantly faster and requires much less CUDA memory than RTDETRv2. This makes it more accessible and cost-effective for custom training.
- **Well-Maintained Ecosystem:** Ultralytics provides a robust ecosystem with frequent updates, numerous [integrations](https://docs.ultralytics.com/integrations/), and tools like [Ultralytics HUB](https://www.ultralytics.com/hub) for seamless dataset management and training.

### Weaknesses

- **Global Context:** While highly effective, its CNN-based architecture may not capture global context as comprehensively as a transformer in certain niche scenarios with extremely complex object relationships. However, for most applications, its performance is more than sufficient.

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## Performance Head-to-Head: RTDETRv2 vs. YOLOv8

The performance comparison highlights the different design philosophies of the two models. RTDETRv2 pushes for maximum accuracy, while YOLOv8 is engineered for a superior balance of speed, accuracy, and efficiency across a range of hardware.

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | **54.3**             | -                              | 15.03                               | 76                 | 259               |
|            |                       |                      |                                |                                     |                    |                   |
| YOLOv8n    | 640                   | 37.3                 | **80.4**                       | **1.47**                            | **3.2**            | **8.7**           |
| YOLOv8s    | 640                   | 44.9                 | **128.4**                      | **2.66**                            | **11.2**           | **28.6**          |
| YOLOv8m    | 640                   | 50.2                 | **234.7**                      | 5.86                                | 25.9               | 78.9              |
| YOLOv8l    | 640                   | 52.9                 | **375.2**                      | 9.06                                | 43.7               | 165.2             |
| YOLOv8x    | 640                   | 53.9                 | **479.1**                      | **14.37**                           | **68.2**           | **257.8**         |

From the table, we can draw several conclusions:

- **Accuracy:** The largest RTDETRv2-x model slightly edges out YOLOv8x in mAP. However, across the board, YOLOv8 models provide highly competitive accuracy for their size. For example, YOLOv8l nearly matches RTDETRv2-l in accuracy with fewer FLOPs.
- **GPU Speed:** YOLOv8 is significantly faster, especially its smaller variants. YOLOv8n is over 3x faster than the smallest RTDETRv2 model, making it ideal for high-framerate applications. Even the largest YOLOv8x model is faster than its RTDETRv2-x counterpart.
- **CPU Speed:** YOLOv8 demonstrates a massive advantage in CPU inference, a critical factor for deployment on many edge devices and standard servers without dedicated GPUs.
- **Efficiency:** YOLOv8 models are far more efficient in terms of parameters and FLOPs. YOLOv8x achieves nearly the same accuracy as RTDETRv2-x with fewer parameters and FLOPs, showcasing superior architectural efficiency.

## Training and Deployment

When it comes to training, the difference is stark. Training RTDETRv2 is a resource-intensive process that demands high-end GPUs with large amounts of VRAM and can take a considerable amount of time.

In contrast, the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) framework is built for **training efficiency**. It allows for rapid custom training with lower memory requirements, making it accessible to a broader range of developers. The streamlined workflow, from data preparation to model training and validation, is a significant advantage.

For deployment, YOLOv8's versatility shines. It can be easily exported to numerous formats like [ONNX](https://docs.ultralytics.com/integrations/onnx/), [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/), [CoreML](https://docs.ultralytics.com/integrations/coreml/), and [OpenVINO](https://docs.ultralytics.com/integrations/openvino/), ensuring optimized performance on virtually any platform, from cloud servers to mobile phones and embedded systems like [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/).

## Conclusion: Which Model Should You Choose?

RTDETRv2 is a powerful model for researchers and teams with significant computational resources who need to squeeze out the last fraction of a percentage in accuracy for complex [object detection](https://www.ultralytics.com/glossary/object-detection) tasks, such as in [autonomous driving](https://www.ultralytics.com/solutions/ai-in-automotive) or [satellite image analysis](https://www.ultralytics.com/blog/using-computer-vision-to-analyze-satellite-imagery).

However, for the vast majority of developers, researchers, and businesses, **Ultralytics YOLOv8 is the clear winner**. It offers a far more practical and effective solution by delivering an exceptional balance of speed and accuracy. Its key advantages—**versatility across multiple tasks, ease of use, superior training efficiency, lower resource requirements, and a comprehensive, well-supported ecosystem**—make it the ideal choice for building robust, real-world computer vision applications quickly and efficiently. Whether you are deploying on a high-end server or a low-power edge device, YOLOv8 provides a scalable, high-performance, and user-friendly solution.

## Explore Other Models

If you're interested in exploring other models, check out these additional comparisons in our [model comparison series](https://docs.ultralytics.com/compare/):

- [RT-DETR vs. YOLOv9](https://docs.ultralytics.com/compare/rtdetr-vs-yolov9/)
- [YOLOv8 vs. YOLOv9](https://docs.ultralytics.com/compare/yolov8-vs-yolov9/)
- [YOLOv8 vs. YOLOv5](https://docs.ultralytics.com/compare/yolov5-vs-yolov8/)
- [EfficientDet vs. YOLOv8](https://docs.ultralytics.com/compare/efficientdet-vs-yolov8/)
- [YOLOv8 vs. YOLOv7](https://docs.ultralytics.com/compare/yolov7-vs-yolov8/)
- [Explore the latest models like YOLOv10 and YOLO11](https://docs.ultralytics.com/models/yolo11/)
