---
title: RTDETRv2 vs PP-YOLOE+ Comparison
comments: true
description: Compare RTDETRv2 and PP-YOLOE+ for real-time object detection, covering transformer vs CNN design, COCO mAP, TensorRT speed, model size, and use cases.
keywords: RTDETRv2, PP-YOLOE+, object detection, model comparison, Vision Transformer, YOLO, real-time detection, AI, Ultralytics, deep learning
---

# RTDETRv2 vs PP-YOLOE+

Computer vision research has produced several architectural approaches to [real-time object detection](https://docs.ultralytics.com/tasks/detect). **RTDETRv2** and **PP-YOLOE+** take fundamentally different routes: one is a detection transformer, the other a convolutional detector. Both aim for high-performance detection, but their mechanics, training approaches, and ideal deployment scenarios differ.

This guide compares their architectures, performance metrics, and ecosystem support to help developers and researchers choose the right model for their deployment needs.

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["RTDETRv2", "PP-YOLOE+"]'></canvas>

## Model Overviews

Before analyzing the performance data, it is important to understand the origins and architectural goals of each model. Both originate from research teams at [Baidu](https://www.baidu.com/), yet they represent different branches of the object detection family tree.

### RTDETRv2

RTDETRv2 builds on the original Real-Time Detection Transformer and pairs a CNN backbone with an efficient hybrid encoder and a transformer decoder. Its defining characteristic is end-to-end prediction, which removes the need for [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) during post-processing.

- **Author:** Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu
- **Organization:** [Baidu](https://www.baidu.com/)
- **Date:** 2024-07-24
- **Arxiv:** [2407.17140](https://arxiv.org/abs/2407.17140)
- **GitHub:** [RT-DETR Repository](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)

[Learn more about RTDETRv2](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch){ .md-button }

### PP-YOLOE+

PP-YOLOE+ is an improved version of PP-YOLOE from the PaddleDetection team, optimized for industrial applications. It uses a scalable CNN architecture with an anchor-free detection head. Designed for strong speed-accuracy trade-offs, it introduces techniques such as the Efficient Task-aligned Head (ET-head) and distribution focal loss to improve [small object detection](https://www.ultralytics.com/blog/exploring-small-object-detection-with-ultralytics-yolo11).

- **Author:** PaddlePaddle Authors
- **Organization:** Baidu
- **Date:** 2022-04-02
- **Arxiv:** [2203.16250](https://arxiv.org/abs/2203.16250)
- **GitHub:** [PaddleDetection Repository](https://github.com/PaddlePaddle/PaddleDetection/)

[Learn more about PP-YOLOE+](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.8.1/configs/ppyoloe/README.md){ .md-button }

!!! tip "Ecosystem Integration"

    While both models have their standalone research repositories, you can experiment with the original RT-DETR directly in the [Ultralytics Python package](https://docs.ultralytics.com/usage/python), which provides a unified API and built-in export options.

## Architectural Differences

The fundamental difference between these two models lies in how they process visual context and generate predictions.

PP-YOLOE+ uses a traditional but highly optimized convolutional neural network (CNN) backbone. It relies on local receptive fields to extract features, which makes it fast and efficient for standard deployment. However, it still requires NMS post-processing to filter overlapping bounding boxes, which can add latency in dense scenes.

Conversely, RTDETRv2 employs a Hybrid Encoder and a Transformer Decoder. This allows the model to capture global context across the entire image. Its attention mechanism models relationships between objects, so the model outputs final bounding boxes directly without NMS. This end-to-end approach keeps inference latency stable regardless of the number of objects detected.

## Performance Metrics and Comparison

When evaluating [YOLO performance metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics), balance accuracy (mAP) against computational cost (FLOPs) and inference speed. The table below shows the performance of both models across their sizes.

| Model      | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ---------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| RTDETRv2-s | 640                         | 48.1                       | -                                    | 5.03                                      | 20                       | 60                      |
| RTDETRv2-m | 640                         | 51.9                       | -                                    | 7.51                                      | 36                       | 100                     |
| RTDETRv2-l | 640                         | 53.4                       | -                                    | 9.76                                      | 42                       | 136                     |
| RTDETRv2-x | 640                         | 54.3                       | -                                    | 15.03                                     | 76                       | 259                     |
|            |                             |                            |                                      |                                           |                          |                         |
| PP-YOLOE+t | 640                         | 39.9                       | -                                    | 2.84                                      | **4.85**                 | 19.15                   |
| PP-YOLOE+s | 640                         | 43.7                       | -                                    | **2.62**                                  | 7.93                     | **17.36**               |
| PP-YOLOE+m | 640                         | 49.8                       | -                                    | 5.56                                      | 23.43                    | 49.91                   |
| PP-YOLOE+l | 640                         | 52.9                       | -                                    | 8.36                                      | 52.2                     | 110.07                  |
| PP-YOLOE+x | 640                         | **54.7**                   | -                                    | 14.3                                      | 98.42                    | 206.59                  |

PP-YOLOE+x reaches the highest mAP<sup>val</sup> on the [COCO dataset](https://cocodataset.org/) at 54.7%, 0.4 points above RTDETRv2-x, with slightly lower TensorRT latency (14.3 ms vs 15.03 ms) but more parameters (98.42M vs 76M). At the large scale, RTDETRv2-l is more accurate than PP-YOLOE+l (53.4% vs 52.9%) with fewer parameters, while PP-YOLOE+l is faster (8.36 ms vs 9.76 ms). PP-YOLOE+ also offers tiny and small variants under 8M parameters, a size tier RTDETRv2 does not cover, which suits edge deployments.

## The Ultralytics Advantage: Enter YOLO26

RTDETRv2 and PP-YOLOE+ are both capable detectors, but the field has continued to evolve. For developers who want a balance of speed, accuracy, and ecosystem support, [Ultralytics YOLO26](https://platform.ultralytics.com/ultralytics/yolo26) is the recommended model for new projects.

YOLO26 brings the end-to-end inference of detection transformers to a CNN. It offers an optional **End-to-End NMS-Free** inference mode (`nms=False`) that skips NMS post-processing when enabled. It also introduces the **MuSGD Optimizer**, a hybrid of SGD and Muon inspired by large language model (LLM) training, for more stable training and faster convergence.

!!! note "Optimized for the Edge"

    Unlike heavy transformer models that demand substantial CUDA memory, YOLO26 features **DFL Removal** (Distribution Focal Loss) and is optimized for edge computing, delivering up to **43% faster CPU inference** for YOLO26n compared with YOLO11n.

YOLO26 also goes beyond object detection, supporting [instance segmentation](https://docs.ultralytics.com/tasks/segment), semantic segmentation, depth estimation, image classification, [pose estimation](https://docs.ultralytics.com/tasks/pose), and [oriented bounding boxes (OBB)](https://docs.ultralytics.com/tasks/obb) out of the box, whereas PP-YOLOE+ focuses on bounding box detection.

[Learn more about YOLO26](https://platform.ultralytics.com/ultralytics/yolo26){ .md-button }

## Training Methodologies and Ecosystem

Training efficiency and ease of use are where the [Ultralytics ecosystem](https://docs.ultralytics.com/platform) differs most from standalone research repositories. PP-YOLOE+ relies on the PaddlePaddle framework and RTDETRv2 often requires complex environment setup, while Ultralytics models share one installable package and API.

The Ultralytics API provides automated dataset handling and built-in hyperparameter tuning, and YOLO models train with lower memory requirements than transformer detectors. Exporting models to production formats like [ONNX](https://onnxruntime.ai/) or [TensorRT](https://developer.nvidia.com/tensorrt) can be accomplished with a single command.

### Code Example: Streamlined Inference

The example below runs RT-DETR alongside the recommended YOLO26 model using the Ultralytics Python package:

```python
from ultralytics import RTDETR, YOLO

# Initialize an RT-DETR model (Ultralytics supports the original RT-DETR-L and RT-DETR-X)
model_rtdetr = RTDETR("rtdetr-l.pt")

# Perform NMS-free inference on a test image
results_rtdetr = model_rtdetr("https://ultralytics.com/images/bus.jpg")
results_rtdetr[0].show()

# For superior speed and versatility, initialize the latest YOLO26 model
model_yolo26 = YOLO("yolo26n.pt")

# Train the YOLO26 model effortlessly with optimized memory usage
model_yolo26.train(data="coco8.yaml", epochs=50, imgsz=640)

# Export to TensorRT for edge deployment
model_yolo26.export(format="engine")
```

## Real-world Applications and Use Cases

Choosing between these architectures often depends on the specific hardware and application requirements.

- **RTDETRv2** excels in server-side environments and complex scene understanding. Its global attention makes it effective for [crowd management](https://www.ultralytics.com/blog/vision-ai-in-crowd-management) and dense [medical image analysis](https://www.ultralytics.com/glossary/medical-image-analysis), where heavily overlapping objects can cause NMS to suppress valid detections.
- **PP-YOLOE+** suits high-speed industrial inspection and teams already invested in the PaddlePaddle ecosystem. Its low parameter count at the smaller scales makes it viable for certain robotics applications.
- **Ultralytics YOLO26** is the recommended choice for most commercial deployments. Progressive Loss and Small-Target-Aware Label Assignment (STAL) improve small-object recognition, which matters for [aerial drone operations](https://www.ultralytics.com/blog/computer-vision-applications-ai-drone-uav-operations) and [smart city traffic monitoring](https://www.ultralytics.com/blog/computer-vision-ai-in-smart-cities).

## Use Cases and Recommendations

Choosing between RT-DETR and PP-YOLOE+ depends on your specific project requirements, deployment constraints, and ecosystem preferences.

### When to Choose RT-DETR

RT-DETR is a strong choice for:

- **Transformer-Based Detection Research:** Projects exploring attention mechanisms and transformer architectures for end-to-end object detection without NMS.
- **High-Accuracy Scenarios with Flexible Latency:** Applications where detection accuracy is the top priority and slightly higher inference latency is acceptable.
- **Large Object Detection:** Scenes with primarily medium-to-large objects where the global attention mechanism of transformers provides a natural advantage.

### When to Choose PP-YOLOE+

PP-YOLOE+ is recommended for:

- **PaddlePaddle Ecosystem Integration:** Organizations with existing infrastructure built on [Baidu's PaddlePaddle](https://www.paddlepaddle.org.cn/) framework and tooling.
- **Paddle Lite Edge Deployment:** Deploying to hardware with highly optimized inference kernels specifically for the Paddle Lite or Paddle inference engine.
- **High-Accuracy Server-Side Detection:** Scenarios prioritizing maximum detection accuracy on powerful GPU servers where framework dependency is not a concern.

### When to Choose Ultralytics (YOLO26)

For most new projects, Ultralytics YOLO26 offers the best combination of performance and developer experience:

- **NMS-Free Edge Deployment:** Applications requiring consistent, low-latency inference without the complexity of Non-Maximum Suppression post-processing.
- **CPU-Only Environments:** Devices without dedicated GPU acceleration, where YOLO26's up to 43% faster CPU inference (YOLO26n vs YOLO11n) provides a decisive advantage.
- **Small Object Detection:** Challenging scenarios like [aerial drone imagery](https://docs.ultralytics.com/datasets/detect/visdrone) or IoT sensor analysis where ProgLoss and STAL improve accuracy on small objects.

## Conclusion

RTDETRv2 and PP-YOLOE+ show that both transformer and optimized CNN architectures are viable for real-time detection. However, deploying models from separate research codebases can slow production timelines.

The [Ultralytics Platform](https://platform.ultralytics.com) brings annotation, training, and deployment into one workflow. By moving to integrated models like [YOLO11](https://platform.ultralytics.com/ultralytics/yolo11) or the newer YOLO26, teams get strong accuracy-speed trade-offs with lower memory requirements and less development overhead.
