---
comments: true
description: Compare YOLOv8 and YOLO11 for object detection. Explore their performance, architecture, and best-use cases to find the right model for your needs.
keywords: YOLOv8, YOLO11, object detection, Ultralytics, YOLO comparison, machine learning, computer vision, inference speed, model accuracy
---

# YOLOv8 vs YOLO11: Evolution of State-of-the-Art Object Detection

The YOLO (You Only Look Once) family of models has consistently defined the cutting edge of real-time [object detection](https://www.ultralytics.com/glossary/object-detection) and computer vision. From the groundbreaking release of YOLOv8 in early 2023 to the refined advancements of YOLO11 in late 2024, Ultralytics has continually pushed the boundaries of speed, accuracy, and efficiency. This comparison explores the architectural shifts, performance metrics, and ideal use cases for both models, helping developers choose the right tool for their specific [computer vision tasks](https://docs.ultralytics.com/tasks/).

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv8", "YOLO11"]'></canvas>

## Ultralytics YOLOv8: The Industry Standard

Released in January 2023, YOLOv8 marked a significant milestone in the history of computer vision. It introduced a unified framework for training models across detection, [instance segmentation](https://docs.ultralytics.com/tasks/segment/), pose estimation, and classification, consolidating these tasks into a single, easy-to-use API. Its anchor-free detection head and mosaic augmentation strategies set a new standard for [accuracy](https://www.ultralytics.com/glossary/accuracy) and training speed.

### Key Features of YOLOv8

- **Unified Framework:** Seamlessly supports multiple vision tasks within the same codebase.
- **Anchor-Free Design:** Reduces the number of box predictions, speeding up Non-Maximum Suppression (NMS).
- **Mosaic Augmentation:** Enhances the model's ability to detect objects in complex scenes by combining four training images.
- **Extensive Community Support:** As a mature model, it boasts a massive ecosystem of tutorials, integrations, and third-party tools.

**Metadata:**

- **Authors:** Glenn Jocher, Ayush Chaurasia, and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com)
- **Date:** 2023-01-10
- **GitHub:** [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Docs:** [YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8/)

[Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/){ .md-button }

## Ultralytics YOLO11: Refined Efficiency and Power

Building on the solid foundation of its predecessor, YOLO11 (released September 2024) introduces targeted architectural improvements designed to boost efficiency and performance. By refining the [backbone](https://www.ultralytics.com/glossary/backbone) and neck architecture, YOLO11 achieves higher [feature extraction](https://www.ultralytics.com/glossary/feature-extraction) capabilities while reducing the parameter count. This results in a model that is not only more accurate but also computationally lighter, making it particularly effective for edge deployment.

### Key Improvements in YOLO11

- **Enhanced Feature Extraction:** A redesigned C3k2 block improves the flow of gradient information, leading to better convergence.
- **Parameter Efficiency:** YOLO11m uses 22% fewer parameters than YOLOv8m while achieving higher [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map).
- **Faster Inference:** Optimized specifically for modern hardware, offering quicker processing times on CPU and GPU.
- **Broad Adaptability:** Excellent support for diverse deployment environments, from cloud servers to [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/) edge devices.

**Metadata:**

- **Authors:** Glenn Jocher and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com)
- **Date:** 2024-09-27
- **GitHub:** [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Docs:** [YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/)

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

## Performance Comparison

The following metrics highlight the performance differences between the two models on the [COCO dataset](https://docs.ultralytics.com/datasets/detect/coco/). While YOLOv8 remains a powerful contender, YOLO11 demonstrates clear advantages in the trade-off between model size and accuracy.

| Model       | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ----------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv8n     | 640                   | 37.3                 | 80.4                           | 1.47                                | 3.2                | 8.7               |
| YOLOv8s     | 640                   | 44.9                 | 128.4                          | 2.66                                | 11.2               | 28.6              |
| YOLOv8m     | 640                   | 50.2                 | 234.7                          | 5.86                                | 25.9               | 78.9              |
| YOLOv8l     | 640                   | 52.9                 | 375.2                          | 9.06                                | 43.7               | 165.2             |
| YOLOv8x     | 640                   | 53.9                 | 479.1                          | 14.37                               | 68.2               | 257.8             |
|             |                       |                      |                                |                                     |                    |                   |
| **YOLO11n** | 640                   | **39.5**             | **56.1**                       | **1.5**                             | **2.6**            | **6.5**           |
| **YOLO11s** | 640                   | **47.0**             | **90.0**                       | **2.5**                             | **9.4**            | **21.5**          |
| **YOLO11m** | 640                   | **51.5**             | **183.2**                      | **4.7**                             | **20.1**           | **68.0**          |
| **YOLO11l** | 640                   | **53.4**             | **238.6**                      | **6.2**                             | **25.3**           | **86.9**          |
| **YOLO11x** | 640                   | **54.7**             | **462.8**                      | **11.3**                            | **56.9**           | **194.9**         |

### Critical Analysis

- **Efficiency:** YOLO11 consistently outperforms YOLOv8 in terms of [FLOPs](https://www.ultralytics.com/glossary/flops) (Floating Point Operations) and parameter count. For example, the **YOLO11n** model provides a significantly higher mAP (39.5 vs 37.3) while being nearly 30% faster on CPU.
- **Speed:** In real-time applications where every millisecond counts, such as [autonomous driving](https://www.ultralytics.com/blog/ultralytics-yolov8-for-speed-estimation-in-computer-vision-projects), YOLO11 provides a tangible advantage. The reduced latency on T4 GPUs makes it an attractive option for high-throughput video analytics.
- **Accuracy:** Across all scales, from Nano to X-Large, YOLO11 achieves higher validation accuracy. This is crucial for difficult tasks like [small object detection](https://www.ultralytics.com/blog/exploring-small-object-detection-with-ultralytics-yolo11), where previous models might struggle.

!!! tip "Choosing the Right Version"

    If you are starting a new project, **YOLO11** is generally the recommended choice due to its superior efficiency-to-accuracy ratio. However, if you have an existing production pipeline built heavily around YOLOv8 and do not require the absolute latest performance margins, YOLOv8 remains a fully supported and reliable option.

## Architecture Deep Dive

While both models share the core "backbone-neck-head" structure typical of YOLO detectors, the internal components have evolved significantly.

### Backbone and Neck

YOLOv8 utilizes the C2f module, which improved upon the C3 module from [YOLOv5](https://docs.ultralytics.com/models/yolov5/) by adding more gradient flow branches. YOLO11 iterates on this further with the **C3k2** block. This new block allows for more flexible kernel sizes and feature aggregation, enhancing the network's ability to capture spatial hierarchies in images. This is particularly beneficial for tasks like [instance segmentation](https://docs.ultralytics.com/tasks/segment/) where precise boundaries are key.

### Training Methodology

Both models benefit from the Ultralytics "Smart Training" system, which automatically adjusts hyperparameters during training. However, YOLO11 incorporates refined data augmentation strategies and optimized loss functions that stabilize training, especially for the smaller model variants. This results in faster convergence, meaning users use less GPU time to reach peak accuracy.

## Use Cases and Real-World Applications

The versatility of both models allows them to excel in a wide variety of industries.

### Where YOLOv8 Excels

- **Established Pipelines:** For companies with locked-in software versions, YOLOv8's stability and massive documentation base make it a safe, low-risk choice.
- **Community Projects:** Due to its longer time in the market, there are countless open-source projects, from [robotic arms](https://www.ultralytics.com/blog/from-algorithms-to-automation-ais-role-in-robotics) to DIY home security, built specifically for v8.

### Where YOLO11 Excels

- **Edge Computing:** With its lower parameter count, YOLO11 is the superior choice for deployment on resource-constrained devices like Raspberry Pis or mobile phones using [iOS CoreML](https://www.ultralytics.com/blog/bringing-ultralytics-yolo11-to-apple-devices-via-coreml) or Android TFLite.
- **High-Precision Tasks:** Applications requiring minute detail, such as [medical imaging](https://www.ultralytics.com/blog/using-yolo11-for-tumor-detection-in-medical-imaging) (e.g., tumor detection) or quality control in [manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing), benefit from YOLO11's improved feature extraction.
- **Real-Time Analytics:** For video feeds processing [traffic management](https://www.ultralytics.com/blog/optimizingtraffic-management-with-ultralytics-yolo11) data, the speed improvements in YOLO11 allow for processing more streams per GPU.

## Code Example: Training and Inference

The beauty of the Ultralytics ecosystem is the unified API. Switching from YOLOv8 to YOLO11 often requires changing only a single character in your code string.

```python
from ultralytics import YOLO

# Load a model (switch 'yolov8n.pt' to 'yolo11n.pt' to upgrade)
model = YOLO("yolo11n.pt")

# Train the model on the COCO8 dataset
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference on an image
# Returns a list of Results objects
results = model(["path/to/image.jpg"])

# Process results
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks
    result.show()  # Display to screen
    result.save(filename="result.jpg")  # Save to disk
```

## Conclusion

Both YOLOv8 and YOLO11 represent the pinnacle of open-source computer vision. YOLOv8 solidified the user-friendly, multi-task framework that developers love, while YOLO11 refined that vision with state-of-the-art architectural optimizations.

For developers seeking the best possible balance of speed, accuracy, and efficiency, **YOLO11** is the clear winner. Its ability to deliver higher performance with fewer parameters makes it a future-proof choice for modern AI applications. However, the robust ecosystem of YOLOv8 ensures it remains a viable and powerful tool for many existing workflows.

For those looking to the absolute cutting edge, keep an eye on the newly released [YOLO26](https://docs.ultralytics.com/models/yolo26/), which introduces end-to-end NMS-free detection for even faster deployment.

!!! note "Explore Further"

    Interested in other models? Check out [YOLOv9](https://docs.ultralytics.com/models/yolov9/) for programmable gradient information or [YOLOv10](https://docs.ultralytics.com/models/yolov10/) for earlier experiments in NMS-free architectures.
