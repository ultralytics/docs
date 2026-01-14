---
comments: true
description: Compare YOLO11 and YOLOv8 architectures, performance, use cases, and benchmarks. Discover which YOLO model fits your object detection needs.
keywords: YOLO11, YOLOv8, object detection, model comparison, performance benchmarks, YOLO series, computer vision, Ultralytics YOLO, YOLO architecture
---

# Ultralytics YOLO11 vs YOLOv8: The Evolution of Real-Time Object Detection

In the rapidly advancing world of computer vision, selecting the right model is critical for balancing accuracy, speed, and resource efficiency. Two of the most prominent names in this space are [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/), a revolutionary anchor-free model released in early 2023, and **Ultralytics YOLO11**, the refined successor introduced in late 2024.

While YOLOv8 established a new standard for ease of use and versatility, YOLO11 builds upon that foundation with architectural enhancements designed to squeeze more performance out of fewer parameters. This comparison explores the technical nuances, performance metrics, and ideal applications for both models to help you decide which is best for your [computer vision projects](https://www.ultralytics.com/blog/all-you-need-to-know-about-ultralytics-yolo11-and-its-applications).

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO11", "YOLOv8"]'></canvas>

### Performance Comparison

The following table highlights the performance differences between the two model families. Notably, YOLO11 offers higher accuracy (mAP) often at reduced parameter counts, translating to greater computational efficiency.

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLO11n | 640                   | **39.5**             | **56.1**                       | 1.5                                 | **2.6**            | **6.5**           |
| YOLO11s | 640                   | **47.0**             | **90.0**                       | **2.5**                             | **9.4**            | **21.5**          |
| YOLO11m | 640                   | **51.5**             | **183.2**                      | **4.7**                             | **20.1**           | **68.0**          |
| YOLO11l | 640                   | **53.4**             | **238.6**                      | **6.2**                             | **25.3**           | **86.9**          |
| YOLO11x | 640                   | **54.7**             | **462.8**                      | **11.3**                            | **56.9**           | **194.9**         |
|         |                       |                      |                                |                                     |                    |                   |
| YOLOv8n | 640                   | 37.3                 | 80.4                           | 1.47                                | 3.2                | 8.7               |
| YOLOv8s | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |

## Architectural Evolution

The transition from YOLOv8 to YOLO11 represents a shift towards deeper feature refinement and parameter efficiency.

**YOLOv8** was a landmark release that popularized the **anchor-free detection** head, eliminating the need for manual anchor box calculations and making the model more robust to variations in object shape. It introduced the **C2f module** (Cross-Stage Partial bottleneck with two convolutions), which combined high-level features with contextual information to improve gradient flow.

**YOLO11** retains the successful anchor-free design but introduces significant architectural upgrades:

1.  **C3k2 Block:** An evolution of the C2f module, the C3k2 block utilizes selectable kernel sizes. This allows the model to adapt its receptive field more dynamically, capturing intricate details in smaller objects while maintaining context for larger ones.
2.  **C2PSA (Cross-Stage Partial Spatial Attention):** YOLO11 integrates spatial attention mechanisms directly into the processing pipeline. This helps the model focus on relevant areas of the image, suppressing background noise and improving performance in cluttered environments.
3.  **Efficiency Improvements:** By optimizing the backbone and neck, YOLO11m achieves a higher mAP than YOLOv8m on the [COCO dataset](https://cocodataset.org/) while using **22% fewer parameters**.

!!! tip "Efficiency Matters"

    Fewer parameters in YOLO11 don't just mean a smaller file size; it significantly reduces the memory bandwidth required during inference. This makes YOLO11 particularly potent for [deployment on edge devices](https://docs.ultralytics.com/guides/model-deployment-practices/) like Raspberry Pis or NVIDIA Jetson modules where RAM is limited.

## Versatility and Supported Tasks

A hallmark of the Ultralytics ecosystem is versatility. Both YOLOv8 and YOLO11 are not limited to bounding boxes; they natively support a wide array of [computer vision tasks](https://docs.ultralytics.com/tasks/):

- **[Object Detection](https://docs.ultralytics.com/tasks/detect/):** Standard bounding box localization.
- **[Instance Segmentation](https://docs.ultralytics.com/tasks/segment/):** Pixel-level masking of objects, useful for precise boundary analysis.
- **[Pose Estimation](https://docs.ultralytics.com/tasks/pose/):** Mapping skeletal keypoints, essential for activity recognition and sports analytics.
- **[Oriented Bounding Boxes (OBB)](https://docs.ultralytics.com/tasks/obb/):** Detecting rotated objects, critical for aerial imagery and satellite analysis.
- **[Image Classification](https://docs.ultralytics.com/tasks/classify/):** Assigning a global class label to an entire image.

While both models support these tasks, YOLO11's improved feature extraction capabilities generally result in sharper masks for segmentation and more accurate keypoint placement in pose estimation tasks.

## Training and Ease of Use

Both models share the same streamlined [Ultralytics Python API](https://docs.ultralytics.com/usage/python/) and Command Line Interface (CLI). This unified experience ensures that developers can switch between versions with virtually zero code changes.

### Efficient Training

Ultralytics models are renowned for their **training efficiency**. Unlike large transformer-based models that require massive GPU clusters and extended training times, YOLOv8 and YOLO11 can be fine-tuned on custom datasets using standard consumer GPUs in a matter of hours. This accessibility is bolstered by [Transfer Learning](https://www.ultralytics.com/glossary/transfer-learning), where models pre-trained on massive datasets (like ImageNet or COCO) are adapted to specific niche tasks.

Here is how simple it is to train either model in Python:

```python
from ultralytics import YOLO

# Load a model (YOLO11n or yolov8n.pt)
model = YOLO("yolo11n.pt")

# Train the model on your custom data
results = model.train(
    data="coco8.yaml",  # path to dataset configuration
    epochs=100,  # training duration
    imgsz=640,  # input image size
    device=0,  # run on GPU
)
```

## Ideal Use Cases

Choosing between the two often comes down to the specific constraints of your deployment environment.

**When to choose YOLOv8:**
YOLOv8 remains a powerhouse with a massive, established community. It is an excellent choice for legacy projects where the pipeline is already optimized for v8 weights. Its stability and extensive third-party tutorial support make it a safe bet for researchers who need to benchmark against a well-known standard.

**When to choose YOLO11:**
YOLO11 is the recommended choice for **new development**. Its superior parameter efficiency makes it ideal for:

- **Edge AI:** Running on mobile phones, drones, or IoT cameras where every megabyte of RAM counts.
- **Real-Time Analytics:** Applications requiring high-FPS processing, such as [traffic monitoring](https://www.ultralytics.com/blog/optimizingtraffic-management-with-ultralytics-yolo11) or manufacturing quality control.
- **Complex Scenarios:** The added attention mechanisms help in scenarios with occlusion or complex backgrounds.

## The Ultralytics Advantage

Whether you choose YOLO11 or YOLOv8, leveraging the Ultralytics ecosystem provides distinct advantages over competing frameworks:

1.  **Well-Maintained Ecosystem:** Ultralytics provides frequent updates, ensuring compatibility with the latest versions of PyTorch, CUDA, and Python.
2.  **Performance Balance:** These models strike the industry's best balance between latency and accuracy (the "Pareto frontier"), making them suitable for real-world production, not just academic papers.
3.  **Low Memory Footprint:** The optimized architecture minimizes CUDA memory usage during training, allowing for larger [batch sizes](https://www.ultralytics.com/glossary/batch-size) on modest hardware.
4.  **Exportability:** Both models seamlessly [export](https://docs.ultralytics.com/modes/export/) to formats like **ONNX, TensorRT, CoreML, and TFLite**, ensuring your model can run anywhere from a server farm to an iPhone.

## Model Details

### YOLO11

YOLO11 represents the refined cutting edge of the YOLO family, optimized for efficiency and diverse deployment targets.

- **Authors:** Glenn Jocher and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com/)
- **Date:** 2024-09-27
- **GitHub:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Docs:** [https://docs.ultralytics.com/models/yolo11/](https://docs.ultralytics.com/models/yolo11/)

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

### YOLOv8

YOLOv8 remains one of the most popular and widely deployed object detection models in the world, known for its reliability and robust performance.

- **Authors:** Glenn Jocher, Ayush Chaurasia, and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com/)
- **Date:** 2023-01-10
- **GitHub:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Docs:** [https://docs.ultralytics.com/models/yolov8/](https://docs.ultralytics.com/models/yolov8/)

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## Other Models

While YOLO11 and YOLOv8 are excellent choices, the field of AI is always moving forward. Developers interested in the absolute latest innovations should also investigate **[YOLO26](https://docs.ultralytics.com/models/yolo26/)**, the newest state-of-the-art model from Ultralytics. YOLO26 introduces an end-to-end NMS-free design and MuSGD optimization, pushing the boundaries of speed and simplicity even further. Conversely, for those maintaining older systems, the legendary [YOLOv5](https://docs.ultralytics.com/models/yolov5/) remains a reliable workhorse in the industry.
