---
comments: true
description: Compare YOLOv10 and YOLOX for object detection. Explore architecture, benchmarks, and use cases to choose the best real-time detection model for your needs.
keywords: YOLOv10, YOLOX, object detection, Ultralytics, real-time, model comparison, benchmark, computer vision, deep learning, AI
---

# Comparison of YOLOX vs YOLOv10

The evolution of real-time object detection has seen rapid advancements, with each iteration of the YOLO (You Only Look Once) family pushing the boundaries of speed and accuracy. Two significant milestones in this lineage are **YOLOX**, which reintroduced anchor-free mechanisms to the mainstream, and **YOLOv10**, which pioneered end-to-end NMS-free detection.

This comprehensive analysis compares YOLOX and YOLOv10, examining their architectural innovations, performance metrics, and suitability for modern computer vision applications. While YOLOX represented a major shift in 2021, YOLOv10 (2024) and the subsequent [YOLO26](https://docs.ultralytics.com/models/yolo26/) build upon these foundations to deliver state-of-the-art results.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOX", "YOLOv10"]'></canvas>

## YOLOX: Returning to Anchor-Free Roots

Released in July 2021 by researchers at [Megvii](https://www.megvii.com/), YOLOX marked a pivot away from the anchor-based approaches that dominated YOLOv4 and YOLOv5. By adopting an anchor-free design, YOLOX simplified the training process and improved generalization, particularly for diverse object shapes.

- **Authors:** Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun
- **Organization:** Megvii
- **Date:** 2021-07-18
- **Links:** [Arxiv](https://arxiv.org/abs/2107.08430), [GitHub](https://github.com/Megvii-BaseDetection/YOLOX), [Docs](https://yolox.readthedocs.io/en/latest/)

### Key Architectural Features

YOLOX introduced several "bag-of-freebies" that modernized the detector:

1.  **Anchor-Free Mechanism:** Unlike [YOLOv5](https://docs.ultralytics.com/models/yolov5/), which uses predefined anchor boxes, YOLOX predicts bounding boxes directly. This reduces the number of design parameters and eliminates the need for clustering analysis to determine optimal anchor shapes for custom datasets.
2.  **Decoupled Head:** The classification and regression tasks are separated into different branches. This separation resolves the conflict between classification confidence and localization accuracy, leading to faster convergence.
3.  **SimOTA:** A simplified Optimal Transport Assignment strategy handles label assignment dynamically, treating it as an optimal transport problem to match ground truths with predictions effectively.

!!! tip "Anchor-Free Legacy"

    YOLOX demonstrated that anchor-free detectors could match or exceed the performance of anchor-based counterparts, influencing subsequent designs like [YOLOv8](https://docs.ultralytics.com/models/yolov8/) and YOLOv10.

[Learn more about YOLOX](https://github.com/Megvii-BaseDetection/YOLOX){ .md-button }

## YOLOv10: The End-to-End Revolution

Released in May 2024 by [Tsinghua University](https://www.tsinghua.edu.cn/en/), YOLOv10 introduced a paradigm shift by eliminating the need for Non-Maximum Suppression (NMS) during inference. This end-to-end capability significantly lowers latency and simplifies deployment pipelines.

- **Authors:** Ao Wang, Hui Chen, Lihao Liu, et al.
- **Organization:** Tsinghua University
- **Date:** 2024-05-23
- **Links:** [Arxiv](https://arxiv.org/abs/2405.14458), [GitHub](https://github.com/THU-MIG/yolov10), [Docs](https://docs.ultralytics.com/models/yolov10/)

### Key Architectural Features

YOLOv10 focuses on both efficiency and accuracy through a holistic design strategy:

1.  **NMS-Free Training:** Utilizing [Consistent Dual Assignments](https://docs.ultralytics.com/models/yolov10/#consistent-dual-assignments-for-nms-free-training), YOLOv10 trains with both one-to-many and one-to-one heads. During inference, only the one-to-one head is used, removing the computational bottleneck of NMS post-processing.
2.  **Holistic Efficiency-Accuracy Design:** The architecture features lightweight classification heads and spatial-channel decoupled downsampling to reduce computational redundancy.
3.  **Large-Kernel Convolutions:** By selectively increasing the receptive field, the model captures broader context without the heavy cost of full attention mechanisms used in transformers.

[Learn more about YOLOv10](https://docs.ultralytics.com/models/yolov10/){ .md-button }

## Performance Comparison

When comparing these two models, the difference in generation becomes apparent. YOLOv10 generally offers superior parameter efficiency and inference speed, largely due to the removal of NMS. While YOLOX was highly competitive in 2021, YOLOv10 utilizes modern optimization techniques to achieve higher mAP<sup>val</sup> with fewer FLOPs.

The table below highlights the performance metrics on the [COCO dataset](https://docs.ultralytics.com/datasets/detect/coco/). Note the significant latency advantage of YOLOv10, particularly in the TensorRT environment.

| Model     | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| --------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLOXnano | 416                   | 25.8                 | -                              | -                                   | 0.91               | 1.08              |
| YOLOXtiny | 416                   | 32.8                 | -                              | -                                   | 5.06               | 6.45              |
| YOLOXs    | 640                   | 40.5                 | -                              | 2.56                                | 9.0                | 26.8              |
| YOLOXm    | 640                   | 46.9                 | -                              | 5.43                                | 25.3               | 73.8              |
| YOLOXl    | 640                   | 49.7                 | -                              | 9.04                                | 54.2               | 155.6             |
| YOLOXx    | 640                   | 51.1                 | -                              | 16.1                                | 99.1               | 281.9             |
|           |                       |                      |                                |                                     |                    |                   |
| YOLOv10n  | 640                   | 39.5                 | -                              | **1.56**                            | **2.3**            | **6.7**           |
| YOLOv10s  | 640                   | 46.7                 | -                              | 2.66                                | 7.2                | 21.6              |
| YOLOv10m  | 640                   | **51.3**             | -                              | 5.48                                | 15.4               | 59.1              |
| YOLOv10b  | 640                   | 52.7                 | -                              | 6.54                                | 24.4               | 92.0              |
| YOLOv10l  | 640                   | 53.3                 | -                              | 8.33                                | 29.5               | 120.3             |
| YOLOv10x  | 640                   | **54.4**             | -                              | **12.2**                            | **56.9**           | **160.4**         |

## Use Cases and Applications

Choosing between these models depends on specific project constraints, though newer models generally offer broader versatility.

### Ideal Scenarios for YOLOX

YOLOX remains a strong candidate for research involving legacy systems or specific academic comparisons. Its decoupled head architecture makes it useful for scenarios where classification and regression tasks need to be analyzed or modified independently. Additionally, developers deeply integrated into the Megvii ecosystem may find continuity in using YOLOX.

### Ideal Scenarios for YOLOv10

YOLOv10 is optimized for real-time applications where every millisecond counts.

- **Edge Computing:** The elimination of NMS reduces the CPU load on [edge AI devices](https://www.ultralytics.com/blog/edge-ai-and-edge-computing-powering-real-time-intelligence), making it perfect for Raspberry Pi or Jetson deployments.
- **High-Speed Manufacturing:** For [quality inspection](https://www.ultralytics.com/blog/quality-inspection-in-manufacturing-traditional-vs-deep-learning-methods) on fast-moving conveyor belts, the low latency ensures no defects are missed.
- **Crowded Scenes:** The end-to-end logic handles object occlusions better than traditional NMS, which sometimes suppresses valid detections in dense crowds.

## The Ultralytics Advantage

While YOLOX offers a robust standalone codebase, adopting Ultralytics models like YOLOv10 (and the newer YOLO26) provides distinct ecosystem benefits.

### Ease of Use and Ecosystem

Ultralytics prioritizes developer experience. Models are integrated into a unified Python package, allowing users to switch between [YOLOv8](https://docs.ultralytics.com/models/yolov8/), YOLOv10, and [YOLO11](https://docs.ultralytics.com/models/yolo11/) with a single line of code. This contrasts with YOLOX, which often requires a specific environment setup.

- **Training Efficiency:** Ultralytics models utilize advanced data augmentation and efficient training loops, often requiring less CUDA memory than older architectures.
- **Versatility:** Unlike YOLOX, which is primarily an object detector, the Ultralytics framework supports [instance segmentation](https://docs.ultralytics.com/tasks/segment/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), [OBB](https://docs.ultralytics.com/tasks/obb/), and classification.

### Moving Forward: YOLO26

For developers looking for the absolute cutting edge, **YOLO26** represents the pinnacle of this technology. It inherits the NMS-free design pioneered by YOLOv10 but refines it further. YOLO26 removes Distribution Focal Loss (DFL) for easier export to NPU devices and utilizes the MuSGD optimizer for faster convergence. With up to 43% faster CPU inference, YOLO26 is the recommended choice for new projects, offering improvements in small-object recognition via ProgLoss and STAL functions.

!!! example "Code Example: Running YOLOv10 with Ultralytics"

    Running inference with YOLOv10 is straightforward using the Ultralytics API.

    ```python
    from ultralytics import YOLO

    # Load a pre-trained YOLOv10n model
    model = YOLO("yolov10n.pt")

    # Run inference on a local image
    results = model("path/to/image.jpg")

    # Show results
    results[0].show()
    ```

## Conclusion

Both YOLOX and YOLOv10 are significant achievements in computer vision. YOLOX successfully challenged the anchor-based status quo, paving the way for modern detectors. However, YOLOv10's introduction of end-to-end, NMS-free detection represents a more recent leap in efficiency.

For developers seeking the best balance of speed, accuracy, and ease of deployment, YOLOv10—and the newer **YOLO26**—are the superior choices. Their integration into the [Ultralytics ecosystem](https://www.ultralytics.com/) ensures long-term support, extensive documentation, and seamless deployment across diverse platforms.

**Looking for other models?**
Explore the capabilities of [YOLO11](https://docs.ultralytics.com/models/yolo11/) for general-purpose tasks or [RT-DETR](https://docs.ultralytics.com/models/rtdetr/) for transformer-based detection.
