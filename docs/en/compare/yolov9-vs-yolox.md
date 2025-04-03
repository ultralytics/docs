---
comments: true
description: Discover a detailed comparison of YOLOv9 and YOLOX, covering architectures, benchmarks, and use cases to help you choose the best object detection model.
keywords: YOLOv9, YOLOX, object detection, model comparison, computer vision, YOLO models, architecture, benchmarks, deep learning
---

# YOLOv9 vs. YOLOX: A Technical Comparison

Selecting the optimal object detection model is crucial for achieving desired outcomes in computer vision projects. Models differ significantly in architecture, performance, and resource requirements. This page provides a detailed technical comparison between [YOLOv9](https://docs.ultralytics.com/models/yolov9/) and YOLOX, analyzing their key features to help you choose the best fit for your needs.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv9", "YOLOX"]'></canvas>

## YOLOv9: Advancing Real-Time Object Detection

**Authors:** Chien-Yao Wang, Hong-Yuan Mark Liao  
**Organization:** Institute of Information Science, Academia Sinica, Taiwan  
**Date:** 2024-02-21  
**Arxiv:** [arXiv:2402.13616](https://arxiv.org/abs/2402.13616)  
**GitHub:** [github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)  
**Docs:** [docs.ultralytics.com/models/yolov9/](https://docs.ultralytics.com/models/yolov9/)

[YOLOv9](https://docs.ultralytics.com/models/yolov9/) represents a significant leap in object detection, introducing innovative techniques like **Programmable Gradient Information (PGI)** and the **Generalized Efficient Layer Aggregation Network (GELAN)**. Developed by Chien-Yao Wang and Hong-Yuan Mark Liao, YOLOv9 tackles information loss in deep neural networks, enhancing both accuracy and efficiency. Integrated into the Ultralytics ecosystem, YOLOv9 benefits from a streamlined user experience, comprehensive [documentation](https://docs.ultralytics.com/models/yolov9/), and robust community support.

### Architecture and Key Features

YOLOv9's architecture is designed to preserve crucial information flow through deep layers using PGI. GELAN optimizes the network structure for better parameter utilization and computational efficiency. This results in state-of-the-art performance with remarkable efficiency, particularly evident in the performance table below. The Ultralytics implementation ensures **ease of use** with a simple API and efficient training processes, leveraging readily available pre-trained weights.

### Strengths

- **State-of-the-Art Accuracy:** Achieves leading mAP scores on benchmarks like [COCO](https://docs.ultralytics.com/datasets/detect/coco/).
- **High Efficiency:** Outperforms previous models by delivering high accuracy with fewer parameters and FLOPs, making it suitable for [edge AI](https://www.ultralytics.com/glossary/edge-ai) deployment.
- **Information Preservation:** PGI effectively mitigates information loss, improving model learning.
- **Ultralytics Ecosystem:** Benefits from active development, extensive resources, [Ultralytics HUB](https://hub.ultralytics.com/) integration for MLOps, and lower memory requirements during training compared to many alternatives.
- **Versatility:** While the original paper focuses on detection, the architecture shows potential for tasks like [instance segmentation](https://docs.ultralytics.com/tasks/segment/) and potentially more, aligning with the multi-task capabilities often found in Ultralytics models like [YOLOv8](https://docs.ultralytics.com/models/yolov8/).

### Weaknesses

- **Novelty:** As a newer model, the range of community-driven deployment examples might still be growing compared to long-established models, although integration within Ultralytics accelerates adoption.

### Ideal Use Cases

YOLOv9 excels in applications demanding the highest accuracy and real-time performance:

- **Advanced Driver-Assistance Systems (ADAS):** Critical for [autonomous vehicles](https://www.ultralytics.com/solutions/ai-in-automotive).
- **High-Resolution Analysis:** Suitable for detailed inspection tasks in [manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing).
- **Resource-Constrained Deployment:** Smaller variants (YOLOv9t, YOLOv9s) offer excellent performance on devices with limited compute power.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

## YOLOX: Anchor-Free High Performance

**Authors:** Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, Jian Sun  
**Organization:** Megvii  
**Date:** 2021-07-18  
**Arxiv:** [arXiv:2107.08430](https://arxiv.org/abs/2107.08430)  
**GitHub:** [github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)  
**Docs:** [yolox.readthedocs.io/en/latest/](https://yolox.readthedocs.io/en/latest/)

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) is an anchor-free object detection model developed by Megvii, aiming for simplicity and high performance. Introduced in 2021, it simplifies the detection pipeline by removing anchor boxes and incorporating features like a decoupled head and the SimOTA label assignment strategy.

### Architecture and Key Features

YOLOX's **anchor-free design** reduces model complexity and the number of hyperparameters compared to anchor-based predecessors. It uses a **decoupled head** for classification and localization tasks and employs advanced training techniques like **SimOTA** and strong data augmentation (e.g., MixUp, Mosaic) to boost performance.

### Strengths

- **Good Accuracy/Speed Balance:** Achieves competitive performance, especially for its time.
- **Anchor-Free Simplicity:** Reduces design parameters and potentially improves generalization.
- **Scalability:** Offers various model sizes (Nano to X) for different resource constraints.

### Weaknesses

- **Outperformed by Newer Models:** While strong, YOLOX is generally surpassed in accuracy and efficiency by newer models like YOLOv9 (see table below).
- **Ecosystem:** Lacks the integrated ecosystem, extensive tooling ([Ultralytics HUB](https://hub.ultralytics.com/)), and unified API provided by Ultralytics YOLO models.
- **Hyperparameter Sensitivity:** Performance can be sensitive to tuning, as noted in some comparisons.

### Ideal Use Cases

YOLOX remains suitable for applications where a solid anchor-free model is needed, though newer alternatives often provide better performance:

- **Real-time Detection:** Where a balance between speed and accuracy is sufficient.
- **Research Baseline:** As a well-established anchor-free model for comparison.
- **Edge Deployment:** Smaller variants like YOLOX-Nano/Tiny are designed for low-resource devices.

[Learn more about YOLOX](https://yolox.readthedocs.io/en/latest/){ .md-button }

## Performance Comparison

The table below compares various YOLOv9 and YOLOX model variants based on their performance on the COCO dataset.

| Model     | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| --------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOv9t   | 640                   | 38.3                 | -                              | **2.3**                             | **2.0**            | **7.7**           |
| YOLOv9s   | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m   | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c   | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e   | 640                   | **55.6**             | -                              | 16.77                               | 57.3               | 189.0             |
|           |                       |                      |                                |                                     |                    |                   |
| YOLOXnano | 416                   | 25.8                 | -                              | -                                   | **0.91**           | **1.08**          |
| YOLOXtiny | 416                   | 32.8                 | -                              | -                                   | 5.06               | 6.45              |
| YOLOXs    | 640                   | 40.5                 | -                              | 2.56                                | 9.0                | 26.8              |
| YOLOXm    | 640                   | 46.9                 | -                              | 5.43                                | 25.3               | 73.8              |
| YOLOXl    | 640                   | 49.7                 | -                              | 9.04                                | 54.2               | 155.6             |
| YOLOXx    | 640                   | 51.1                 | -                              | 16.1                                | 99.1               | 281.9             |

**Analysis:** YOLOv9 models consistently demonstrate superior mAP compared to YOLOX models of similar or even larger sizes. For instance, YOLOv9c achieves 53.0% mAP with 25.3M parameters, surpassing YOLOXl (49.7% mAP, 54.2M params) and YOLOXx (51.1% mAP, 99.1M params). Furthermore, YOLOv9 models often exhibit competitive or better inference speeds, highlighting their enhanced efficiency thanks to architectural innovations like PGI and GELAN.

## Conclusion

YOLOv9 stands out as a superior choice compared to YOLOX, offering state-of-the-art accuracy and efficiency. Its innovative architecture addresses key challenges in deep learning, resulting in significant performance gains. When integrated within the Ultralytics framework, YOLOv9 provides an exceptional user experience characterized by **ease of use**, a **well-maintained ecosystem**, efficient training, and excellent performance balance. For developers and researchers seeking the best combination of accuracy, speed, and usability for modern [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) tasks, YOLOv9 is the recommended model.

Explore other models in the Ultralytics documentation, such as [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/), [YOLOv10](https://docs.ultralytics.com/models/yolov10/), and [RT-DETR](https://docs.ultralytics.com/models/rtdetr/), for a broader view of available state-of-the-art options.
