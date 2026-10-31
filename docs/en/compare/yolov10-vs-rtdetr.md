---
title: YOLOv10 vs RTDETRv2 Comparison
comments: true
description: Compare YOLOv10 and RTDETRv2, two NMS-free real-time object detectors, covering CNN vs transformer design, COCO mAP, T4 TensorRT latency, parameters, and FLOPs.
keywords: YOLOv10, RTDETRv2, object detection, model comparison, AI, computer vision, Ultralytics, real-time detection, transformer-based models, YOLO series
---

# YOLOv10 vs RTDETRv2

[Computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) moves quickly, and new architectures regularly raise the state of the art in real-time [object detection](https://www.ultralytics.com/glossary/object-detection). Two notable milestones are YOLOv10 and RTDETRv2. Both models remove a bottleneck in traditional detection pipelines by eliminating the need for [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) post-processing, but they approach this challenge with entirely different architectures.

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv10", "RTDETRv2"]'></canvas>

This technical comparison analyzes their architectures, training methodologies, and ideal deployment scenarios to help developers and researchers choose the right tool for their next vision AI project.

## YOLOv10: The NMS-Free Pioneer

Developed by researchers at Tsinghua University, YOLOv10 focuses on architectural efficiency and the removal of post-processing bottlenecks. By introducing consistent dual assignments for NMS-free training, it achieves competitive accuracy with lower inference latency.

### Technical Specifications

- **Authors:** Ao Wang, Hui Chen, Lihao Liu, et al.
- **Organization:** [Tsinghua University](https://www.tsinghua.edu.cn/en/)
- **Date:** 2024-05-23
- **ArXiv:** [YOLOv10 Paper](https://arxiv.org/abs/2405.14458)
- **GitHub:** [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)
- **Docs:** [YOLOv10 Documentation](https://docs.ultralytics.com/models/yolov10)

### Architecture and Methodologies

YOLOv10's primary contribution is its holistic efficiency-accuracy driven model design. It optimizes various components from both perspectives, reducing computational overhead. The consistent dual assignments strategy allows the model to train without relying on NMS, which translates to a streamlined, end-to-end deployment pipeline. This is particularly beneficial when exporting models to edge formats like [ONNX](https://docs.ultralytics.com/integrations/onnx) or [TensorRT](https://docs.ultralytics.com/integrations/tensorrt), where post-processing operations can introduce unexpected latency.

### Strengths and Weaknesses

The model offers strong speed-accuracy trade-offs, especially in the smaller variants (N and S). Its low latency makes it well suited to high-speed edge environments. However, while YOLOv10 excels at raw detection speed, it remains a specialized detection-only model. Teams requiring [instance segmentation](https://docs.ultralytics.com/tasks/segment) or [pose estimation](https://docs.ultralytics.com/tasks/pose) will need to look towards more versatile frameworks.

[Learn more about YOLOv10](https://docs.ultralytics.com/models/yolov10){ .md-button }

## RTDETRv2: Refining the Detection Transformer

Building upon the original Real-Time Detection Transformer, RTDETRv2 incorporates a "bag of freebies" to improve upon its baseline, showcasing that transformers can compete with CNNs in real-time scenarios.

### Technical Specifications

- **Authors:** Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu
- **Organization:** [Baidu](https://www.baidu.com/)
- **Date:** 2024-07-24
- **ArXiv:** [RTDETRv2 Paper](https://arxiv.org/abs/2407.17140)
- **GitHub:** [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)
- **Docs:** [RTDETRv2 README](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme)

### Architecture and Methodologies

RTDETRv2 uses a hybrid architecture, combining a convolutional neural network (CNN) backbone for visual feature extraction with a transformer encoder-decoder for scene-level understanding. The transformer's self-attention mechanism allows the model to view the image globally, making it effective at handling complex scenes, overlapping objects, and dense crowds.

### Strengths and Weaknesses

The transformer architecture provides excellent accuracy, particularly on larger parameter scales, and natively outputs final detections without NMS. However, this comes at a cost. Transformer models traditionally require more CUDA memory during training and can be slower to converge compared to pure CNN architectures. While RTDETRv2 has improved inference speeds, it generally consumes more memory than lightweight YOLO variants.

[Learn more about RTDETRv2](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch){ .md-button }

## Performance Comparison

Evaluating the performance metrics provides a clearer picture of where each model excels. The following table highlights their capabilities on the [COCO dataset](https://docs.ultralytics.com/datasets/detect/coco):

| Model      | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ---------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| YOLOv10n   | 640                         | 38.5                       | -                                    | **1.84**                                  | **2.3**                  | **6.7**                 |
| YOLOv10s   | 640                         | 46.3                       | -                                    | 2.49                                      | 7.2                      | 21.6                    |
| YOLOv10m   | 640                         | 51.1                       | -                                    | 4.74                                      | 15.4                     | 59.1                    |
| YOLOv10b   | 640                         | 52.5                       | -                                    | 5.74                                      | 19.1                     | 92.0                    |
| YOLOv10l   | 640                         | 53.2                       | -                                    | 7.28                                      | 24.4                     | 120.3                   |
| YOLOv10x   | 640                         | **54.4**                   | -                                    | 10.70                                     | 29.5                     | 160.4                   |
|            |                             |                            |                                      |                                           |                          |                         |
| RTDETRv2-s | 640                         | 48.1                       | -                                    | 5.03                                      | 20                       | 60                      |
| RTDETRv2-m | 640                         | 51.9                       | -                                    | 7.51                                      | 36                       | 100                     |
| RTDETRv2-l | 640                         | 53.4                       | -                                    | 9.76                                      | 42                       | 136                     |
| RTDETRv2-x | 640                         | 54.3                       | -                                    | 15.03                                     | 76                       | 259                     |

YOLOv10 holds a clear advantage in parameter efficiency and TensorRT inference speed across comparable sizes, while RTDETRv2 is slightly more accurate at the s, m, and l scales (for example, 48.1 vs 46.3 mAP for the s models, at 5.03 ms vs 2.49 ms). RTDETRv2-x matches YOLOv10x in accuracy (54.3 vs 54.4 mAP) but requires more than 2.5x the parameters (76M vs 29.5M) and higher FLOPs (259B vs 160.4B).

## Use Cases and Recommendations

Choosing between YOLOv10 and RT-DETR depends on your specific project requirements, deployment constraints, and ecosystem preferences.

### When to Choose YOLOv10

YOLOv10 is a strong choice for:

- **NMS-Free Real-Time Detection:** Applications that benefit from end-to-end detection without Non-Maximum Suppression, reducing deployment complexity.
- **Balanced Speed-Accuracy Tradeoffs:** Projects requiring a strong balance between inference speed and detection accuracy across various model scales.
- **Consistent-Latency Applications:** Deployment scenarios where predictable inference times are critical, such as [robotics](https://www.ultralytics.com/glossary/robotics) or autonomous systems.

### When to Choose RT-DETR

RT-DETR is recommended for:

- **Transformer-Based Detection Research:** Projects exploring attention mechanisms and transformer architectures for end-to-end object detection without NMS.
- **High-Accuracy Scenarios with Flexible Latency:** Applications where detection accuracy is the top priority and slightly higher inference latency is acceptable.
- **Large Object Detection:** Scenes with primarily medium-to-large objects where the global attention mechanism of transformers provides a natural advantage.

### When to Choose Ultralytics (YOLO26)

For most new projects, [Ultralytics YOLO26](https://platform.ultralytics.com/ultralytics/yolo26) offers the best combination of performance and developer experience:

- **NMS-Free Edge Deployment:** Applications requiring consistent, low-latency inference without the complexity of Non-Maximum Suppression post-processing.
- **CPU-Only Environments:** Devices without dedicated GPU acceleration, where YOLO26's up to 43% faster CPU inference provides a decisive advantage.
- **Small Object Detection:** Challenging scenarios like [aerial drone imagery](https://docs.ultralytics.com/datasets/detect/visdrone) or IoT sensor analysis where ProgLoss and STAL improve accuracy on small objects.

## The Ultralytics Advantage: Ecosystem and Innovation

YOLOv10 and RTDETRv2 are both capable detectors, but choosing a model is often about the surrounding software ecosystem. [Ultralytics Platform](https://platform.ultralytics.com) provides a unified interface that hides much of the complexity of deep learning workflows.

### The New Standard: Ultralytics YOLO26

For developers seeking the best performance, Ultralytics YOLO26 brings together recent architectural advances. Released in January 2026, YOLO26 inherits the end-to-end NMS-free design pioneered by YOLOv10 as an optional head (`nms=False`), eliminating NMS post-processing for faster, simpler deployment.

!!! tip "Why Choose YOLO26?"

    YOLO26 brings large language model (LLM) training methods to computer vision via the **MuSGD Optimizer** (a hybrid of SGD and Muon), resulting in more stable training and faster convergence. YOLO26n also runs up to **43% faster on CPU** than YOLO11n, making YOLO26 a strong choice for edge computing.

Furthermore, YOLO26 introduces **ProgLoss + STAL** for notable improvements in small-object recognition, and unlike the detection-only YOLOv10, it supports [object detection](https://docs.ultralytics.com/tasks/detect), instance segmentation, semantic segmentation, depth estimation, image classification, pose estimation, and [oriented bounding boxes (OBB)](https://docs.ultralytics.com/tasks/obb), with task-specific improvements like a semantic segmentation loss and Residual Log-Likelihood Estimation (RLE) for pose. The removal of Distribution Focal Loss (DFL) also simplifies export and improves low-power device compatibility.

[Learn more about YOLO26](https://platform.ultralytics.com/ultralytics/yolo26){ .md-button }

### Ease of Use and Training Efficiency

Whether you are working with earlier models like [Ultralytics YOLO11](https://platform.ultralytics.com/ultralytics/yolo11) or the newer YOLO26, the Python API keeps training memory usage low and workflows short. The same API also runs YOLOv10 and RT-DETR:

```python
from ultralytics import RTDETR, YOLO

# Train the end-to-end YOLOv10 model
model_yolo = YOLO("yolov10n.pt")
model_yolo.train(data="coco8.yaml", epochs=100, imgsz=640)

# Alternatively, evaluate RTDETR within the same API
model_rtdetr = RTDETR("rtdetr-l.pt")
results = model_rtdetr.predict("https://ultralytics.com/images/bus.jpg")
```

The well-maintained ecosystem provides tools for easy [hyperparameter tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning) and integrates with experiment tracking tools and [model deployment options](https://docs.ultralytics.com/guides/model-deployment-options).

## Conclusion

Both YOLOv10 and RTDETRv2 are important milestones in NMS-free object detection. RTDETRv2 proves that transformers can achieve real-time latency with excellent global context comprehension, albeit with higher memory requirements. YOLOv10 provides an efficient, fast CNN alternative tailored for resource-constrained detection tasks.

For balanced performance, multi-task versatility, and a mature ecosystem, we recommend **Ultralytics YOLO26**. It combines the architectural ideas of its predecessors with user-friendly tooling that simplifies deploying vision AI.
