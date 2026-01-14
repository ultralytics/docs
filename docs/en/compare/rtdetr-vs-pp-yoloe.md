---
comments: true
description: Explore the key differences between RTDETRv2 and PP-YOLOE+, two leading object detection models. Compare architectures, performance, and use cases.
keywords: RTDETRv2,PP-YOLOE+,object detection,model comparison,Vision Transformer,YOLO,real-time detection,AI,Ultralytics,deep learning
---

# RT-DETRv2 vs. PP-YOLOE+: Advancing Real-Time Object Detection

The evolution of real-time object detection has been marked by a fierce rivalry between Convolutional Neural Networks (CNNs) and Transformer-based architectures. This comparison explores two significant milestones in this journey: **RT-DETRv2**, a vision transformer designed to eliminate post-processing bottlenecks, and **PP-YOLOE+**, a highly optimized CNN-based detector from the PaddlePaddle ecosystem.

Both models aim to solve the classic trade-off between inference speed and detection accuracy, but they approach the problem with fundamentally different architectural philosophies. Whether you are building [autonomous vehicle systems](https://www.ultralytics.com/solutions/ai-in-automotive) or deploying [smart retail analytics](https://www.ultralytics.com/solutions/ai-in-retail), understanding these differences is crucial for selecting the right tool.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["RTDETRv2", "PP-YOLOE+"]'></canvas>

## Performance Metrics Comparison

The table below provides a direct comparison of key performance indicators. While PP-YOLOE+ demonstrates impressive efficiency in smaller models, RT-DETRv2 often shines in complex scenarios requiring global context.

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| RTDETRv2-s | 640                   | **48.1**             | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | **51.9**             | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | **53.4**             | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | 54.3                 | -                              | 15.03                               | **76**             | 259               |
|            |                       |                      |                                |                                     |                    |                   |
| PP-YOLOE+t | 640                   | 39.9                 | -                              | 2.84                                | **4.85**           | **19.15**         |
| PP-YOLOE+s | 640                   | 43.7                 | -                              | **2.62**                            | 7.93               | 17.36             |
| PP-YOLOE+m | 640                   | 49.8                 | -                              | **5.56**                            | 23.43              | **49.91**         |
| PP-YOLOE+l | 640                   | 52.9                 | -                              | **8.36**                            | 52.2               | **110.07**        |
| PP-YOLOE+x | 640                   | **54.7**             | -                              | **14.3**                            | 98.42              | **206.59**        |

## RT-DETRv2: The Transformer Evolution

RT-DETRv2 (Real-Time Detection Transformer v2) builds upon the success of the original RT-DETR, which was the first transformer-based detector to genuinely rival YOLO models in real-time applications. Developed by Baidu, it addresses the high computational cost typically associated with transformers while retaining their ability to capture long-range dependencies in an image.

### Architecture and Innovation

The core innovation of RT-DETRv2 is its **efficient hybrid encoder** and the elimination of Non-Maximum Suppression (NMS). Unlike CNNs that process local features, the transformer architecture allows the model to "look" at the entire image at once. This global receptive field is particularly beneficial for detecting objects in [crowded scenes](https://www.ultralytics.com/blog/vision-ai-in-crowd-management) or when objects are occluded.

Key architectural features include:

- **End-to-End Prediction:** It directly predicts a set of bounding boxes without the need for NMS post-processing, simplifying deployment pipelines.
- **IoU-aware Query Selection:** This mechanism improves initialization by selecting the most relevant image features as object queries.
- **Flexible Decoder:** The inference speed can be adjusted by modifying the number of decoder layers without retraining, offering versatility for different hardware constraints.

[Learn more about RT-DETR](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

### Strengths and Weaknesses

The primary advantage of RT-DETRv2 is its accuracy in complex environments and its NMS-free design, which ensures stable inference latency. However, transformers generally require more GPU memory during training compared to CNNs, and their training convergence can be slower.

!!! tip "Ultralytics Integration"

    Ultralytics provides full support for RT-DETR models. You can easily train, validate, and deploy them using the same simple API used for YOLO models, leveraging pre-trained weights for faster convergence.

    ```python
    from ultralytics import RTDETR

    # Load a COCO-pretrained RT-DETR-l model
    model = RTDETR("rtdetr-l.pt")

    # Train the model on your custom dataset
    results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
    ```

## PP-YOLOE+: The Refined CNN Powerhouse

PP-YOLOE+ is an evolved version of PP-YOLOE, developed by the PaddlePaddle team at Baidu. It represents the pinnacle of "anchor-free" CNN object detection, refining the classic YOLO architecture with modern "Bag-of-Freebies" to squeeze out maximum performance.

### Architecture and Innovation

PP-YOLOE+ utilizes a CSPResNet backbone and a distinctive head architecture. It focuses on optimizing the trade-off between parameters and accuracy through:

- **Task-Aligned Learning (TAL):** A dynamic label assignment strategy that aligns classification and localization quality, ensuring high-confidence detections are also spatially accurate.
- **Object365 Pre-training:** The "+" in PP-YOLOE+ often refers to models initialized with weights pre-trained on the massive [Objects365 dataset](https://docs.ultralytics.com/datasets/detect/objects365/), which provides a robust feature representation capability superior to standard COCO pre-training.
- **Efficient Scaling:** The model scales effectively from tiny edge devices (PP-YOLOE+ t) to high-performance servers (PP-YOLOE+ x).

### Strengths and Weaknesses

PP-YOLOE+ is exceptionally fast on standard GPU hardware and offers very competitive accuracy. Its CNN nature makes it memory-efficient during training. However, like most CNN-based detectors (excluding the newest generations like [YOLO26](https://docs.ultralytics.com/models/yolo26/)), it typically relies on NMS, which can introduce latency variability in scenes with a high density of objects.

## Technical Summary

Both models are excellent choices, but they serve slightly different niches.

**RT-DETRv2** is ideal when:

- **Stability is key:** You need consistent inference times regardless of the number of objects (due to NMS-free design).
- **Complex Scenes:** Your application involves heavy occlusion or requires understanding relationships between distant parts of an image.
- **Versatility:** You want to adjust speed/accuracy trade-offs at runtime by changing decoder layers.

**PP-YOLOE+** is ideal when:

- **Edge Deployment:** You are targeting smaller edge devices where every millisecond of compute and megabyte of memory counts.
- **Legacy Compatibility:** Your infrastructure is heavily optimized for standard CNN operations.
- **Training Resources:** You have limited GPU VRAM available for training, as CNNs are generally less memory-hungry than transformers.

### Authors and Citations

**RT-DETRv2**

- **Authors:** Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu
- **Organization:** [Baidu](https://www.baidu.com/)
- **Date:** April 17, 2023 (Original RT-DETR), July 2024 (v2 update)
- **Arxiv:** [RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer](https://arxiv.org/abs/2304.08069)
- **GitHub:** [RT-DETR Repository](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)

**PP-YOLOE+**

- **Authors:** PaddlePaddle Authors
- **Organization:** [Baidu](https://www.baidu.com/)
- **Date:** April 2, 2022
- **Arxiv:** [PP-YOLOE: An Evolved Version of YOLO](https://arxiv.org/abs/2203.16250)
- **GitHub:** [PaddleDetection Repository](https://github.com/PaddlePaddle/PaddleDetection/)

## Why Choose Ultralytics Models?

While RT-DETRv2 and PP-YOLOE+ are formidable, the [Ultralytics ecosystem](https://www.ultralytics.com/) offers distinct advantages for developers looking for a unified, user-friendly experience.

- **Ease of Use:** Whether you are using [YOLO11](https://docs.ultralytics.com/models/yolo11/) or the cutting-edge **YOLO26**, the Ultralytics Python API allows you to switch between models with a single line of code. You can train a detection model, switch to segmentation, or try pose estimation without learning a new framework.
- **Performance Balance:** Ultralytics models are engineered for the optimal sweet spot between speed and accuracy. The new **YOLO26** introduces an end-to-end NMS-free design similar to RT-DETR but with the efficiency of a CNN, offering the "best of both worlds."
- **Training Efficiency:** Ultralytics models are renowned for their "Train, Val, Predict" simplicity. We provide highly optimized [pre-trained weights](https://docs.ultralytics.com/models/) that allow for transfer learning in a fraction of the time it takes to train other architectures from scratch.
- **Versatility:** Unlike many specialized repositories, Ultralytics supports a vast array of tasks including [Object Detection](https://docs.ultralytics.com/tasks/detect/), [Instance Segmentation](https://docs.ultralytics.com/tasks/segment/), [Pose Estimation](https://docs.ultralytics.com/tasks/pose/), [Oriented Bounding Boxes (OBB)](https://docs.ultralytics.com/tasks/obb/), and [Classification](https://docs.ultralytics.com/tasks/classify/) all under one roof.

[Learn more about YOLO26](https://docs.ultralytics.com/models/yolo26/){ .md-button }

### Discover More

For those interested in exploring further, check out our guides on [training custom datasets](https://docs.ultralytics.com/modes/train/) or learn how to [export models](https://docs.ultralytics.com/modes/export/) to formats like ONNX, TensorRT, and CoreML for production deployment.
