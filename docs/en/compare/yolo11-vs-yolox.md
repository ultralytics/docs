---
comments: true
description: Explore YOLO11 and YOLOX, two leading object detection models. Compare architecture, performance, and use cases to select the best model for your needs.
keywords: YOLO11, YOLOX, object detection, machine learning, computer vision, model comparison, deep learning, Ultralytics, real-time detection, anchor-free models
---

# YOLO11 vs YOLOX: Technical Comparison

Choosing the right object detection model is crucial for achieving optimal performance in computer vision applications. This page offers a detailed technical comparison between Ultralytics YOLO11 and YOLOX, two advanced models designed for object detection tasks. We will explore their architectural nuances, performance benchmarks, and suitability for different use cases to guide you in making an informed decision.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO11", "YOLOX"]'></canvas>

## Ultralytics YOLO11

[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) is the latest state-of-the-art (SOTA) model in the YOLO series from Ultralytics, released in September 2024. Developed by Glenn Jocher and Jing Qiu, it builds upon the success of previous versions like [YOLOv8](https://docs.ultralytics.com/models/yolov8/), focusing on enhanced speed, accuracy, and efficiency.

**Technical Details:**

- **Authors:** Glenn Jocher, Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com/)
- **Date:** 2024-09-27
- **Arxiv Link:** None
- **GitHub Link:** <https://github.com/ultralytics/ultralytics>
- **Docs Link:** <https://docs.ultralytics.com/models/yolo11/>

### Architecture and Key Features

YOLO11 employs an anchor-free architecture, similar to its predecessor YOLOv8, but with significant refinements in the backbone and neck structures. These optimizations aim to improve feature extraction and fusion while reducing computational cost and parameters. This results in faster inference speeds, particularly on CPUs, and lower memory requirements during training and inference compared to many competitors.

### Strengths

- **Ease of Use:** YOLO11 benefits from the streamlined Ultralytics ecosystem, featuring a simple [Python API](https://docs.ultralytics.com/usage/python/) and [CLI](https://docs.ultralytics.com/usage/cli/), extensive [documentation](https://docs.ultralytics.com/), and readily available tutorials.
- **Well-Maintained Ecosystem:** Ultralytics provides active development, frequent updates, strong community support via [GitHub](https://github.com/ultralytics/ultralytics/issues), [Discord](https://discord.com/invite/ultralytics), and integration with [Ultralytics HUB](https://hub.ultralytics.com/) for seamless MLOps workflows.
- **Performance Balance:** YOLO11 offers an excellent trade-off between speed and accuracy across various model sizes (from 'n' nano to 'x' extra-large), making it suitable for diverse deployment scenarios from edge devices to cloud servers.
- **Memory Efficiency:** Designed for efficiency, YOLO11 typically requires less CUDA memory for training and inference compared to larger architectures like transformers.
- **Versatility:** YOLO11 is a multi-task model supporting [object detection](https://docs.ultralytics.com/tasks/detect/), [instance segmentation](https://docs.ultralytics.com/tasks/segment/), [image classification](https://docs.ultralytics.com/tasks/classify/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), and [oriented bounding boxes (OBB)](https://docs.ultralytics.com/tasks/obb/).
- **Training Efficiency:** Offers efficient training processes with readily available pre-trained weights on datasets like [COCO](https://docs.ultralytics.com/datasets/detect/coco/) and [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/).

### Weaknesses

- Larger models (YOLO11l, YOLO11x) naturally require more computational resources, a common characteristic of high-accuracy models.

### Ideal Use Cases

YOLO11 excels in real-time applications requiring high accuracy and speed, such as autonomous systems ([robotics](https://www.ultralytics.com/glossary/robotics), [automotive](https://www.ultralytics.com/solutions/ai-in-automotive)), security and surveillance, [industrial automation](https://www.ultralytics.com/solutions/ai-in-manufacturing), and retail analytics.

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

## YOLOX

YOLOX was introduced by Megvii in 2021 as an anchor-free version of YOLO, aiming to simplify the architecture while boosting performance through techniques like a decoupled head and advanced label assignment (SimOTA).

**Technical Details:**

- **Authors:** Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, Jian Sun
- **Organization:** Megvii
- **Date:** 2021-07-18
- **Arxiv Link:** <https://arxiv.org/abs/2107.08430>
- **GitHub Link:** <https://github.com/Megvii-BaseDetection/YOLOX>
- **Docs Link:** <https://yolox.readthedocs.io/en/latest/>

### Architecture and Key Features

YOLOX's main contributions include its anchor-free design, a decoupled head separating classification and regression tasks, and the SimOTA label assignment strategy. It uses a modified CSPNet backbone similar to other YOLO models.

### Strengths

- **High Accuracy:** Achieves competitive mAP scores, particularly with larger model variants (YOLOX-l, YOLOX-x).
- **Anchor-Free Design:** Simplifies the detection pipeline compared to earlier anchor-based YOLO models.
- **Advanced Training Techniques:** Incorporates effective strategies like SimOTA and strong data augmentation.

### Weaknesses

- **Performance Trade-offs:** While accurate, larger YOLOX models can be slower and more resource-intensive (higher FLOPs and parameters) than comparable YOLO11 models. CPU inference speeds are often not reported or benchmarked.
- **Ecosystem and Usability:** May require more effort to integrate and use compared to the streamlined Ultralytics ecosystem. Documentation and community support might be less extensive.
- **Task Versatility:** Primarily focused on object detection, lacking the built-in multi-task capabilities (segmentation, pose, classification) offered by YOLO11 within the same framework.

### Ideal Use Cases

YOLOX is suitable for applications where high accuracy is paramount and real-time constraints are less critical, or where its specific architectural features (like the decoupled head) are beneficial. It serves as a strong baseline for object detection research.

[Learn more about YOLOX](https://github.com/Megvii-BaseDetection/YOLOX){ .md-button }

## Performance Comparison

The following table compares various sizes of YOLO11 and YOLOX models based on performance metrics on the COCO dataset.

| Model     | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| --------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLO11n   | 640                   | 39.5                 | **56.1**                       | **1.5**                             | **2.6**            | **6.5**           |
| YOLO11s   | 640                   | 47.0                 | **90.0**                       | **2.5**                             | 9.4                | 21.5              |
| YOLO11m   | 640                   | **51.5**             | **183.2**                      | **4.7**                             | 20.1               | **68.0**          |
| YOLO11l   | 640                   | **53.4**             | **238.6**                      | **6.2**                             | 25.3               | 86.9              |
| YOLO11x   | 640                   | **54.7**             | **462.8**                      | **11.3**                            | 56.9               | 194.9             |
|           |                       |                      |                                |                                     |                    |                   |
| YOLOXnano | 416                   | 25.8                 | -                              | -                                   | **0.91**           | **1.08**          |
| YOLOXtiny | 416                   | 32.8                 | -                              | -                                   | 5.06               | 6.45              |
| YOLOXs    | 640                   | 40.5                 | -                              | 2.56                                | 9.0                | 26.8              |
| YOLOXm    | 640                   | 46.9                 | -                              | 5.43                                | 25.3               | 73.8              |
| YOLOXl    | 640                   | 49.7                 | -                              | 9.04                                | 54.2               | 155.6             |
| YOLOXx    | 640                   | 51.1                 | -                              | 16.1                                | 99.1               | 281.9             |

Analysis of the table reveals that Ultralytics YOLO11 consistently outperforms YOLOX across various metrics. YOLO11 models achieve higher mAP scores with significantly fewer parameters and FLOPs compared to YOLOX models of similar scales (e.g., YOLO11m vs YOLOXm, YOLO11x vs YOLOXx). Furthermore, YOLO11 demonstrates substantially faster inference speeds, especially on CPU, highlighting its superior efficiency and optimization for real-world deployment. The availability of multiple tasks within the same architecture further enhances YOLO11's value proposition.

## Conclusion

While YOLOX was a significant contribution with its anchor-free design and training strategies, Ultralytics YOLO11 represents a more advanced and practical choice for most users. YOLO11 offers superior performance balance (higher accuracy with better speed and efficiency), greater versatility through multi-task support, and benefits from the user-friendly, well-maintained Ultralytics ecosystem. For developers and researchers seeking a state-of-the-art, easy-to-use, and highly efficient object detection model, Ultralytics YOLO11 is the recommended option.

For further exploration, consider comparing these models with others in the Ultralytics ecosystem, such as [YOLOv9](https://docs.ultralytics.com/models/yolov9/), [YOLOv10](https://docs.ultralytics.com/models/yolov10/), or other architectures like [RT-DETR](https://docs.ultralytics.com/models/rtdetr/). You can find more comparisons on the [Ultralytics Compare page](https://docs.ultralytics.com/compare/).
