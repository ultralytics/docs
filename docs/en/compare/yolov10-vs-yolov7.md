---
title: YOLOv10 vs YOLOv7 Comparison
comments: true
description: Compare YOLOv10 and YOLOv7 for real-time object detection, including NMS-free design, E-ELAN, COCO mAP, parameter efficiency, Ultralytics support, and use cases.
keywords: YOLOv10, YOLOv7, object detection, model comparison, AI, deep learning, computer vision, performance metrics, architecture, edge AI, robotics, autonomous systems
---

# YOLOv10 vs YOLOv7

The rapid progression of computer vision over the last few years has yielded increasingly efficient architectures for real-time applications. Comparing [YOLOv10](https://docs.ultralytics.com/models/yolov10) and [YOLOv7](https://docs.ultralytics.com/models/yolov7) highlights a crucial transition period in this evolution. While YOLOv7 introduced highly effective training strategies and architectural scaling, YOLOv10 simplified deployment by removing the longstanding reliance on [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms).

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv10", "YOLOv7"]'></canvas>

Both models pushed the boundaries of [object detection](https://docs.ultralytics.com/tasks/detect) upon their respective releases, while the Ultralytics ecosystem and newer models like YOLO26 now offer simpler training and deployment workflows.

## Model Profiles and Origins

Understanding the origins of these models provides valuable context regarding their architectural design choices and the academic research driving them.

### YOLOv10 Details

- **Authors:** Ao Wang, Hui Chen, Lihao Liu, et al.
- **Organization:** [Tsinghua University](https://www.tsinghua.edu.cn/en/)
- **Date:** 2024-05-23
- **Arxiv:** [YOLOv10: Real-Time End-to-End Object Detection](https://arxiv.org/abs/2405.14458)
- **GitHub:** [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)
- **Docs:** [Ultralytics YOLOv10 Documentation](https://docs.ultralytics.com/models/yolov10)

[Learn more about YOLOv10](https://docs.ultralytics.com/models/yolov10){ .md-button }

### YOLOv7 Details

- **Authors:** Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao
- **Organization:** [Institute of Information Science, Academia Sinica, Taiwan](https://www.iis.sinica.edu.tw/en/index.html)
- **Date:** 2022-07-06
- **Arxiv:** [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art](https://arxiv.org/abs/2207.02696)
- **GitHub:** [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- **Docs:** [Ultralytics YOLOv7 Documentation](https://docs.ultralytics.com/models/yolov7)

[Learn more about YOLOv7](https://docs.ultralytics.com/models/yolov7){ .md-button }

## Architectural Innovations

### The YOLOv7 Approach

Released in 2022, YOLOv7 focused heavily on optimizing gradient pathways. It introduced the Extended Efficient Layer Aggregation Network (E-ELAN), which allowed the model to learn more diverse features without destroying the original gradient path. Furthermore, the authors implemented a "trainable bag-of-freebies" methodology, utilizing re-parameterization techniques during training that could be fused away during inference to maintain fast execution speeds. Despite these optimizations, YOLOv7 still relied on NMS for post-processing, creating variable latency during dense scene analysis.

### The YOLOv10 Breakthrough

YOLOv10 addressed the NMS bottleneck directly. By introducing consistent dual assignments during training, the Tsinghua University team enabled NMS-free end-to-end detection. This dual-head approach uses one branch with one-to-many assignments for rich supervisory signals during training, and another branch with one-to-one assignments for NMS-free inference. This architectural shift provides consistent, low [inference latency](https://www.ultralytics.com/glossary/inference-latency) suitable for high-speed video analytics. Furthermore, YOLOv10 employs a holistic efficiency-accuracy driven model design, stripping away computational redundancy found in earlier generations.

!!! tip "Post-Processing Impact"

    Removing NMS post-processing speeds up inference and simplifies deployment on edge AI hardware, such as AI accelerators and NPUs where custom NMS operations can be difficult to compile.

## Performance Comparison

When comparing raw metrics on the [MS COCO dataset](https://docs.ultralytics.com/datasets/detect/coco), the generational gap becomes evident. YOLOv10 achieves a much more favorable trade-off between parameters, computational requirements, and accuracy.

| Model    | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| -------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| YOLOv10n | 640                         | 38.5                       | -                                    | **1.84**                                  | **2.3**                  | **6.7**                 |
| YOLOv10s | 640                         | 46.3                       | -                                    | 2.49                                      | 7.2                      | 21.6                    |
| YOLOv10m | 640                         | 51.1                       | -                                    | 4.74                                      | 15.4                     | 59.1                    |
| YOLOv10b | 640                         | 52.5                       | -                                    | 5.74                                      | 19.1                     | 92.0                    |
| YOLOv10l | 640                         | 53.2                       | -                                    | 7.28                                      | 24.4                     | 120.3                   |
| YOLOv10x | 640                         | **54.4**                   | -                                    | 10.70                                     | 29.5                     | 160.4                   |
|          |                             |                            |                                      |                                           |                          |                         |
| YOLOv7l  | 640                         | 51.4                       | -                                    | 6.84                                      | 36.9                     | 104.7                   |
| YOLOv7x  | 640                         | 53.1                       | -                                    | 11.57                                     | 71.3                     | 189.9                   |

As seen above, YOLOv10x delivers higher mAP than YOLOv7x (54.4% vs 53.1%) while using less than half the parameters (29.5M vs 71.3M). The lightweight YOLOv10 Nano and Small models also offer fast [TensorRT deployment](https://docs.ultralytics.com/integrations/tensorrt) speeds (1.84 ms and 2.49 ms on a T4), making them attractive for edge deployment.

## The Ultralytics Ecosystem Advantage

While studying architectural papers is insightful, modern computer vision development relies on well-maintained frameworks. Selecting an Ultralytics-supported model is a major advantage for developers looking to move from prototype to production quickly.

### Streamlined Development

YOLOv10 is natively supported by the standard Ultralytics Python package, while YOLOv7 checkpoints trained upstream can be run in Ultralytics only after [export to ONNX or TensorRT](https://docs.ultralytics.com/models/yolov7). Native support provides **Ease of Use**, replacing boilerplate code with a simple, intuitive API. Ultralytics YOLO models also need less CUDA memory during training than heavy transformer architectures, enabling larger batch sizes on consumer-grade hardware.

### Unmatched Versatility

While older repositories often focus strictly on bounding box detection, the Ultralytics framework supports a wide range of tasks through models like YOLO26 and YOLO11. Whether you are performing [Instance Segmentation](https://docs.ultralytics.com/tasks/segment), [Pose Estimation](https://docs.ultralytics.com/tasks/pose), or [Oriented Bounding Box (OBB)](https://docs.ultralytics.com/tasks/obb) detection, the workflow remains the same.

### Code Example: Consistent Training Workflows

The following code snippet demonstrates the training process, which automatically handles [data augmentation](https://docs.ultralytics.com/guides/yolo-data-augmentation) and learning rate scheduling:

```python
from ultralytics import YOLO

# Load the desired model (e.g. YOLOv10 or the recommended YOLO26)
model = YOLO("yolo26n.pt")

# Train the model on your dataset
results = model.train(data="coco8.yaml", epochs=100, imgsz=640, batch=16, device=0)

# Export to ONNX format for rapid deployment
model.export(format="onnx")
```

## Use Cases and Recommendations

Choosing between YOLOv10 and YOLOv7 depends on your specific project requirements, deployment constraints, and ecosystem preferences.

### When to Choose YOLOv10

YOLOv10 is a strong choice for:

- **NMS-Free Real-Time Detection:** Applications that benefit from end-to-end detection without Non-Maximum Suppression, reducing deployment complexity.
- **Balanced Speed-Accuracy Tradeoffs:** Projects requiring a strong balance between inference speed and detection accuracy across various model scales.
- **Consistent-Latency Applications:** Deployment scenarios where predictable inference times are critical, such as [robotics](https://www.ultralytics.com/glossary/robotics) or autonomous systems.

### When to Choose YOLOv7

YOLOv7 is recommended for:

- **Academic Benchmarking:** Reproducing 2022-era state-of-the-art results or studying the effects of E-ELAN and trainable bag-of-freebies techniques.
- **Reparameterization Research:** Investigating planned reparameterized convolutions and compound model scaling strategies.
- **Existing Custom Pipelines:** Projects with heavily customized pipelines built around YOLOv7's specific architecture that cannot easily be refactored.

### When to Choose Ultralytics (YOLO26)

For most new projects, [Ultralytics YOLO26](https://platform.ultralytics.com/ultralytics/yolo26) offers the best combination of performance and developer experience:

- **NMS-Free Edge Deployment:** Applications requiring consistent, low-latency inference without the complexity of Non-Maximum Suppression post-processing.
- **CPU-Only Environments:** Devices without dedicated GPU acceleration, where YOLO26n runs up to 43% faster on CPU than YOLO11n.
- **Small Object Detection:** Challenging scenarios like [aerial drone imagery](https://docs.ultralytics.com/datasets/detect/visdrone) or IoT sensor analysis where ProgLoss and STAL improve accuracy on tiny objects.

## The New Standard: Introducing YOLO26

While YOLOv10 was a major step forward in 2024, the field moves quickly. For new development, Ultralytics recommends the latest generation model, **Ultralytics YOLO26**. Released in January 2026, it supersedes both YOLOv7 and YOLOv10 for real-time vision AI.

[Learn more about YOLO26](https://platform.ultralytics.com/ultralytics/yolo26){ .md-button }

YOLO26 brings several improvements designed for modern deployment environments:

- **End-to-End NMS-Free Design:** Building on the foundation laid by YOLOv10, YOLO26's optional one-to-one head (`nms=False`) skips NMS post-processing for simpler deployment pipelines and consistent inference speed.
- **Up to 43% Faster CPU Inference:** YOLO26n runs up to 43% faster than YOLO11n on CPU, which suits edge computing and devices without dedicated GPUs and can lower hardware costs.
- **DFL Removal:** Removing Distribution Focal Loss (DFL) simplifies export logic and improves compatibility with low-power edge devices.
- **MuSGD Optimizer:** Inspired by Moonshot AI's Kimi K2, this hybrid of SGD and Muon brings [Large Language Model (LLM)](https://www.ultralytics.com/glossary/large-language-model-llm) training techniques into computer vision for more stable training and faster convergence.
- **ProgLoss + STAL:** These loss functions improve small-object recognition, a historically challenging area that is critical for drones, robotics, and [smart city monitoring](https://www.ultralytics.com/blog/computer-vision-ai-in-smart-cities).
- **Task-Specific Improvements:** YOLO26 is not just a detector. It adds a semantic segmentation loss that improves instance segmentation convergence, Residual Log-Likelihood Estimation (RLE) for more accurate pose estimation, and a specialized angle loss that resolves OBB boundary issues.

!!! note "Managing Datasets and Training"

    To manage datasets, train YOLO26, and deploy models to the cloud, explore the [Ultralytics Platform](https://platform.ultralytics.com). Its no-code interface complements the Python package.

## Real-World Use Cases

Selecting the right architecture depends heavily on your hardware and application constraints.

### When to use YOLOv7

YOLOv7 remains a reliable choice for maintaining legacy pipelines that are already deeply integrated with its specific tensor structures or when replicating academic benchmarks from 2022 and 2023. It performs well on high-end server GPUs.

### When to use YOLOv10

YOLOv10 suits scenarios requiring consistent latency. Because its NMS-free head avoids post-processing that grows with the number of detections, it works well for high-density crowd counting or [manufacturing defect detection](https://www.ultralytics.com/blog/how-vision-ai-enhances-defect-detection-on-production-lines) where the number of objects fluctuates but the processing time per frame must stay stable.

### When to use YOLO26

YOLO26 is the recommended choice for new projects. From deploying [security alarm systems](https://docs.ultralytics.com/guides/security-alarm-system) on a Raspberry Pi to running large cloud-based video analytics, its faster CPU inference and improved small-object detection give it a clear advantage over older generations.

For developers interested in alternative architectures, Ultralytics also supports transformer-based detectors like [RT-DETR](https://docs.ultralytics.com/models/rtdetr) and earlier models like [Ultralytics YOLO11](https://platform.ultralytics.com/ultralytics/yolo11).
