---
title: YOLOv8 vs PP-YOLOE+ Comparison
comments: true
description: Compare Ultralytics YOLOv8 and Baidu PP-YOLOE+ for object detection, covering architectures, COCO mAP, TensorRT speed, model size, and framework workflows.
keywords: YOLOv8, PP-YOLOE+, object detection, computer vision, model comparison, YOLO models, Ultralytics, PaddlePaddle, deep learning
---

# YOLOv8 vs PP-YOLOE+

In the rapidly evolving field of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv), selecting the right model for [object detection](https://www.ultralytics.com/glossary/object-detection) is critical for achieving a balance between inference speed and accuracy. Two prominent models that have significantly impacted the industry are [Ultralytics YOLOv8](https://platform.ultralytics.com/ultralytics/yolov8) and [PP-YOLOE+](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.8.1/configs/ppyoloe/README.md). This guide provides a technical comparison to help developers and machine learning engineers understand the nuances of their architectures, performance metrics, and ideal deployment scenarios.

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv8", "PP-YOLOE+"]'></canvas>

## Ultralytics YOLOv8: The Versatile Ecosystem Standard

Introduced by Ultralytics, YOLOv8 quickly became widely used in production vision applications, building on years of YOLO research to deliver strong performance across multiple tasks.

- **Authors:** Glenn Jocher, Ayush Chaurasia, and Jing Qiu
- **Organization:** [Ultralytics](https://www.ultralytics.com)
- **Date:** 2023-01-10
- **GitHub:** [Ultralytics Repository](https://github.com/ultralytics/ultralytics)
- **Docs:** [YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8)

[Learn more about YOLOv8](https://platform.ultralytics.com/ultralytics/yolov8){ .md-button }

### Architectural Innovations and Versatility

YOLOv8 features an [anchor-free](https://www.ultralytics.com/glossary/anchor-free-detectors) design and incorporates a decoupled head to independently process classification and regression tasks. This structural refinement leads to better feature representation and faster convergence during training.

Unlike many specialized models, YOLOv8 supports multiple tasks. Beyond bounding box detection, the same unified architecture and API natively support [instance segmentation](https://docs.ultralytics.com/tasks/segment), [image classification](https://docs.ultralytics.com/tasks/classify), [pose estimation](https://docs.ultralytics.com/tasks/pose), and [oriented bounding boxes (OBB)](https://docs.ultralytics.com/tasks/obb).

!!! tip "Streamlined Development"

    The unified Ultralytics ecosystem allows developers to switch between detection, segmentation, and pose estimation tasks by changing the model weights, reducing technical debt.

## PP-YOLOE+: The PaddlePaddle Powerhouse

PP-YOLOE+ is an evolutionary step from previous PP-YOLO iterations, designed to run efficiently on Baidu's [PaddlePaddle](https://github.com/PaddlePaddle) framework.

- **Authors:** PaddlePaddle Authors
- **Organization:** [Baidu](https://www.baidu.com/)
- **Date:** 2022-04-02
- **Arxiv:** [PP-YOLOE Paper](https://arxiv.org/abs/2203.16250)
- **GitHub:** [PaddleDetection Repository](https://github.com/PaddlePaddle/PaddleDetection/)
- **Docs:** [PP-YOLOE+ Configuration](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.8.1/configs/ppyoloe/README.md)

[Learn more about PP-YOLOE+](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.8.1/configs/ppyoloe/README.md){ .md-button }

### Architectural Focus

PP-YOLOE+ introduced the CSPRepResNet backbone and implemented the Efficient Task-aligned Head (ET-head) to improve detection accuracy. It relies heavily on the [PaddlePaddle](https://www.paddlepaddle.org.cn/) deep learning framework. While it achieves high accuracy on standard benchmark datasets like the [COCO dataset](https://cocodataset.org/), its implementation is tied to the PaddlePaddle ecosystem, which can make it challenging to integrate into standard [PyTorch](https://pytorch.org/) or [TensorFlow](https://www.tensorflow.org/) pipelines popular in the broader AI community.

## Performance and Metrics Comparison

When deploying models to edge devices or cloud servers, the balance of accuracy ([mAP](https://www.ultralytics.com/glossary/mean-average-precision-map)), speed, and parameter count is crucial. Ultralytics models are known for low training memory requirements and fast inference.

Below is a detailed comparison table of the models evaluated on COCO val2017.

| Model                                                                  | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ---------------------------------------------------------------------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| [YOLOv8n](https://platform.ultralytics.com/ultralytics/yolov8/yolov8n) | 640                         | 37.3                       | **80.4**                             | **1.47**                                  | **3.2**                  | **8.7**                 |
| [YOLOv8s](https://platform.ultralytics.com/ultralytics/yolov8/yolov8s) | 640                         | 44.9                       | 128.4                                | 2.66                                      | 11.2                     | 28.6                    |
| [YOLOv8m](https://platform.ultralytics.com/ultralytics/yolov8/yolov8m) | 640                         | 50.2                       | 234.7                                | 5.86                                      | 25.9                     | 78.9                    |
| [YOLOv8l](https://platform.ultralytics.com/ultralytics/yolov8/yolov8l) | 640                         | 52.9                       | 375.2                                | 9.06                                      | 43.7                     | 165.1                   |
| [YOLOv8x](https://platform.ultralytics.com/ultralytics/yolov8/yolov8x) | 640                         | 53.9                       | 479.1                                | 14.37                                     | 68.2                     | 257.8                   |
|                                                                        |                             |                            |                                      |                                           |                          |                         |
| PP-YOLOE+t                                                             | 640                         | 39.9                       | -                                    | 2.84                                      | 4.85                     | 19.15                   |
| PP-YOLOE+s                                                             | 640                         | 43.7                       | -                                    | 2.62                                      | 7.93                     | 17.36                   |
| PP-YOLOE+m                                                             | 640                         | 49.8                       | -                                    | 5.56                                      | 23.43                    | 49.91                   |
| PP-YOLOE+l                                                             | 640                         | 52.9                       | -                                    | 8.36                                      | 52.2                     | 110.07                  |
| PP-YOLOE+x                                                             | 640                         | **54.7**                   | -                                    | 14.3                                      | 98.42                    | 206.59                  |

### Analyzing the Trade-offs

While PP-YOLOE+x edges out YOLOv8x in mAP (54.7 vs 53.9), it needs over 30 million additional parameters. In the small and medium sizes the two families are close, with PP-YOLOE+ using fewer FLOPs and only YOLOv8 reporting CPU ONNX speeds in this table. The lightweight YOLOv8n requires only 3.2M parameters and 8.7B FLOPs, making it the smallest model in this comparison, although PP-YOLOE+t reaches higher mAP (39.9 vs 37.3) with 4.85M parameters.

YOLO models also typically use less memory during training than large transformer-based architectures. Models with high CUDA memory footprints often require expensive hardware, whereas YOLOv8 trains efficiently on consumer-grade GPUs.

## Ecosystem, Ease of Use, and Deployment

User experience is a major difference between these architectures.

The **[Ultralytics Platform](https://platform.ultralytics.com)** and Python package form a well-maintained ecosystem that reduces the friction of [machine learning operations](https://www.ultralytics.com/glossary/machine-learning-operations-mlops). They provide a simple API, extensive documentation, and native tools for data logging, hyperparameter tuning, and cross-platform export. Whether you need to deploy via [ONNX](https://docs.ultralytics.com/integrations/onnx), [TensorRT](https://docs.ultralytics.com/integrations/tensorrt), or [CoreML](https://developer.apple.com/documentation/coreml), Ultralytics handles the conversion.

Conversely, PP-YOLOE+ often requires deep knowledge of the PaddlePaddle framework. Converting these models to run efficiently on standard [NVIDIA GPUs](https://www.nvidia.com/en-us/data-center/tesla-t4/) or edge devices outside the Paddle ecosystem can be a multi-step process without the export automation found in Ultralytics tools.

### Training Efficiency with Ultralytics

Training an Ultralytics model requires very little boilerplate code. The following example trains a YOLOv8 model in Python and exports it for TensorRT:

```python
from ultralytics import YOLO

# Load a pre-trained YOLOv8 small model
model = YOLO("yolov8s.pt")

# Train the model on the COCO8 example dataset
results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=0)

# Quickly export the trained model for TensorRT deployment
model.export(format="engine", device=0)
```

## Use Cases and Recommendations

Choosing between YOLOv8 and PP-YOLOE+ depends on your specific project requirements, deployment constraints, and ecosystem preferences.

### When to Choose YOLOv8

YOLOv8 is a strong choice for:

- **Versatile Multi-Task Deployment:** Projects requiring a proven model for [detection](https://docs.ultralytics.com/tasks/detect), segmentation, classification, and pose estimation within the Ultralytics ecosystem.
- **Established Production Systems:** Existing production environments already built on the YOLOv8 architecture with stable, well-tested deployment pipelines.
- **Broad Community and Ecosystem Support:** Applications benefiting from YOLOv8's extensive tutorials, third-party integrations, and active community resources.

### When to Choose PP-YOLOE+

PP-YOLOE+ is recommended for:

- **PaddlePaddle Ecosystem Integration:** Organizations with existing infrastructure built on Baidu's PaddlePaddle framework and tooling.
- **Paddle Lite Edge Deployment:** Deploying to hardware with highly optimized inference kernels specifically for the Paddle Lite or Paddle inference engine.
- **High-Accuracy Server-Side Detection:** Scenarios prioritizing maximum detection accuracy on powerful GPU servers where framework dependency is not a concern.

### When to Choose Ultralytics (YOLO26)

For most new projects, [Ultralytics YOLO26](https://platform.ultralytics.com/ultralytics/yolo26) offers the best combination of performance and developer experience:

- **NMS-Free Edge Deployment:** Applications requiring consistent, low-latency inference without the complexity of [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) post-processing.
- **CPU-Only Environments:** Devices without dedicated GPU acceleration, where YOLO26's up to 43% faster CPU inference (YOLO26n vs YOLO11n) provides a decisive advantage.
- **Small Object Detection:** Challenging scenarios like [aerial drone imagery](https://docs.ultralytics.com/datasets/detect/visdrone) or IoT sensor analysis where ProgLoss and STAL improve accuracy on tiny objects.

## Looking Forward: The YOLO26 Advantage

For new applications, **Ultralytics YOLO26**, released in January 2026, supersedes both YOLOv8 and [YOLO11](https://platform.ultralytics.com/ultralytics/yolo11) with the following features:

- **End-to-End NMS-Free Design:** YOLO26's optional one-to-one head (`nms=False`) [removes the need for NMS post-processing](https://www.ultralytics.com/blog/why-ultralytics-yolo26-removes-nms-and-how-that-changes-deployment), reducing latency variability and simplifying deployment logic.
- **MuSGD Optimizer:** Bringing LLM training techniques to vision AI, this hybrid of SGD and [Muon](https://github.com/KellerJordan/Muon) provides more stable training and faster convergence.
- **Up to 43% Faster CPU Inference:** With Distribution Focal Loss (DFL) removed and a lighter detection head, YOLO26n runs up to 43% faster than YOLO11n on CPU, making it well suited to IoT and mobile applications.
- **ProgLoss + STAL:** Progressive Loss and Small-Target-Aware Label Assignment (STAL) improve small-object detection, a critical requirement for drone analytics and aerial imagery.

!!! note "Upgrade Recommendation"

    While YOLOv8 remains a robust and well-supported option, **YOLO26** is the recommended architecture for new enterprise and research projects, offering higher accuracy, faster CPU inference, and native end-to-end processing.

## Conclusion

Both YOLOv8 and PP-YOLOE+ advanced real-time detection. For most developers and researchers, however, **Ultralytics YOLOv8** and its successor **YOLO26** are the more practical choice: an intuitive API, an active open-source community, lower training memory requirements, and a unified multi-task framework simplify the path from dataset creation to production deployment.
