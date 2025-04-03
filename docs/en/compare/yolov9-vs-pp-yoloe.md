---
comments: true
description: Compare YOLOv9 and PP-YOLOE+ models in architecture, performance, and use cases. Find the best object detection model for your needs.
keywords: YOLOv9,PP-YOLOE+,object detection,model comparison,computer vision,AI,deep learning,YOLO,PP-YOLOE,performance comparison
---

# YOLOv9 vs PP-YOLOE+: A Technical Comparison

Choosing the right object detection model involves balancing accuracy, speed, and resource requirements. This page provides a detailed technical comparison between Ultralytics YOLOv9 and Baidu's PP-YOLOE+, analyzing their architectures, performance metrics, and ideal use cases to guide your selection.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv9", "PP-YOLOE+"]'></canvas>

## YOLOv9: Programmable Gradient Information for Enhanced Learning

**Authors:** Chien-Yao Wang and Hong-Yuan Mark Liao  
**Organization:** Institute of Information Science, Academia Sinica, Taiwan  
**Date:** 2024-02-21  
**Arxiv:** [https://arxiv.org/abs/2402.13616](https://arxiv.org/abs/2402.13616)  
**GitHub:** [https://github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)  
**Documentation:** [https://docs.ultralytics.com/models/yolov9/](https://docs.ultralytics.com/models/yolov9/)

Ultralytics [YOLOv9](https://docs.ultralytics.com/models/yolov9/) represents a significant advancement in real-time object detection. It introduces innovative concepts like Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN). PGI tackles the challenge of information loss in deep networks, ensuring reliable gradient information for updates. GELAN optimizes the network architecture for better parameter utilization and computational efficiency. This combination allows YOLOv9 to achieve superior accuracy while maintaining efficiency.

YOLOv9, integrated within the Ultralytics ecosystem, benefits from a streamlined user experience, a simple API, and comprehensive [documentation](https://docs.ultralytics.com/models/yolov9/). The ecosystem provides active development, strong community support via platforms like [GitHub Discussions](https://github.com/ultralytics/ultralytics/discussions) and [Discord](https://discord.com/invite/ultralytics), frequent updates, and readily available pre-trained weights, facilitating efficient training and deployment. Furthermore, YOLOv9 demonstrates versatility, supporting tasks like [object detection](https://docs.ultralytics.com/tasks/detect/), [instance segmentation](https://docs.ultralytics.com/tasks/segment/), and potentially more as seen in the original repository's teaser for panoptic segmentation and image captioning.

**Strengths:**

- **High Accuracy:** Achieves state-of-the-art mAP scores on datasets like COCO, particularly with larger variants like YOLOv9-E.
- **Efficient Architecture:** GELAN and PGI lead to excellent performance with comparatively fewer parameters and FLOPs.
- **Information Preservation:** PGI effectively mitigates information loss in deep networks.
- **Ultralytics Ecosystem:** Benefits from ease of use, extensive documentation, active maintenance, and community support.
- **Versatility:** Supports multiple computer vision tasks beyond just detection.

**Weaknesses:**

- **Newer Model:** As a recent release, the breadth of community examples and third-party integrations might still be growing compared to older models.
- **Training Resources:** Larger YOLOv9 variants may require significant computational resources for training, although efficiency gains are notable relative to performance.

**Ideal Use Cases:**

YOLOv9 excels in applications demanding high accuracy and efficiency, such as [autonomous driving](https://www.ultralytics.com/solutions/ai-in-automotive), advanced [security systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8), and complex [robotic tasks](https://www.ultralytics.com/glossary/robotics). Its efficient design also makes smaller variants suitable for deployment in resource-constrained environments.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

## PP-YOLOE+: Enhanced Anchor-Free Detection

**Authors:** PaddlePaddle Authors  
**Organization:** Baidu  
**Date:** 2022-04-02  
**Arxiv:** [https://arxiv.org/abs/2203.16250](https://arxiv.org/abs/2203.16250)  
**GitHub:** [https://github.com/PaddlePaddle/PaddleDetection/](https://github.com/PaddlePaddle/PaddleDetection/)  
**Documentation:** [https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.8.1/configs/ppyoloe/README.md](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.8.1/configs/ppyoloe/README.md)

PP-YOLOE+, developed within the [PaddlePaddle](https://github.com/PaddlePaddle/PaddleDetection/) framework, is an evolution of the PP-YOLOE series. It employs an [anchor-free](https://www.ultralytics.com/glossary/anchor-free-detectors) approach, simplifying the detection head and reducing anchor-related hyperparameters. Key features often include a ResNet-based backbone, a Path Aggregation Network (PAN) neck, decoupled heads for classification and localization, and potentially VariFocal Loss or Task Alignment Learning (TAL) to improve accuracy.

**Strengths:**

- **Anchor-Free Design:** Simplifies model architecture and potentially reduces tuning effort related to anchor boxes.
- **Balanced Performance:** Offers a competitive trade-off between detection accuracy (mAP) and inference speed.
- **PaddlePaddle Ecosystem:** Well-integrated and documented within the Baidu PaddlePaddle deep learning framework.

**Weaknesses:**

- **Ecosystem Specificity:** Primarily designed for and optimized within the PaddlePaddle framework, which might be a limitation for users preferring other ecosystems like PyTorch used by Ultralytics.
- **Community and Support:** While supported by Baidu, the broader community and readily available resources might be less extensive compared to the large, active Ultralytics YOLO community.
- **Versatility:** Primarily focused on object detection, potentially lacking the built-in support for multiple tasks (segmentation, pose, classification) found in models like YOLOv9 within the Ultralytics framework.

**Ideal Use Cases:**

PP-YOLOE+ is well-suited for industrial applications like [quality inspection](https://www.ultralytics.com/solutions/ai-in-manufacturing), [smart retail](https://www.ultralytics.com/blog/achieving-retail-efficiency-with-ai), and environmental monitoring where a robust and efficient anchor-free detector is beneficial, particularly for developers already invested in the PaddlePaddle ecosystem.

[Learn more about PP-YOLOE+](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.8.1/configs/ppyoloe/README.md){ .md-button }

## Performance Comparison

The following table compares various sizes of YOLOv9 and PP-YOLOE+ models based on their performance on the COCO dataset.

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| :--------- | :-------------------- | :------------------- | :----------------------------- | :---------------------------------- | :----------------- | :---------------- |
| YOLOv9t    | 640                   | 38.3                 | -                              | **2.3**                             | **2.0**            | **7.7**           |
| YOLOv9s    | 640                   | 46.8                 | -                              | 3.54                                | **7.1**            | 26.4              |
| YOLOv9m    | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c    | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e    | 640                   | **55.6**             | -                              | 16.77                               | 57.3               | 189.0             |
|            |                       |                      |                                |                                     |                    |                   |
| PP-YOLOE+t | 640                   | 39.9                 | -                              | 2.84                                | 4.85               | 19.15             |
| PP-YOLOE+s | 640                   | 43.7                 | -                              | 2.62                                | 7.93               | 17.36             |
| PP-YOLOE+m | 640                   | 49.8                 | -                              | 5.56                                | 23.43              | 49.91             |
| PP-YOLOE+l | 640                   | 52.9                 | -                              | 8.36                                | 52.2               | 110.07            |
| PP-YOLOE+x | 640                   | 54.7                 | -                              | 14.3                                | 98.42              | 206.59            |

YOLOv9 models generally demonstrate superior accuracy (mAP) across comparable model sizes (e.g., YOLOv9c vs PP-YOLOE+l, YOLOv9e vs PP-YOLOE+x). Notably, YOLOv9 achieves this high accuracy with significantly fewer parameters and FLOPs in many cases (e.g., YOLOv9c vs PP-YOLOE+l), indicating higher computational efficiency. While PP-YOLOE+ models offer competitive speeds, particularly the smaller variants on TensorRT, YOLOv9 often provides a better balance, especially when considering the accuracy achieved per parameter or FLOP.

## Conclusion

Both YOLOv9 and PP-YOLOE+ are powerful object detection models. PP-YOLOE+ offers a solid anchor-free alternative, particularly within the PaddlePaddle ecosystem. However, Ultralytics YOLOv9 stands out with its state-of-the-art accuracy, innovative architecture (PGI and GELAN) leading to high efficiency, and the significant advantages of the Ultralytics ecosystem. For developers seeking top performance, ease of use, versatility across tasks, and robust support within a PyTorch-native environment, YOLOv9 presents a compelling choice.

## Other Models

Users interested in exploring further options within the Ultralytics ecosystem might consider:

- [**YOLOv8**](https://docs.ultralytics.com/models/yolov8/): A highly versatile and widely adopted model known for its balance of speed and accuracy across various tasks.
- [**YOLOv5**](https://docs.ultralytics.com/models/yolov5/): A mature and reliable model, still popular for its speed and ease of deployment.
- [**YOLOv10**](https://docs.ultralytics.com/models/yolov10/): Another recent advancement focusing on end-to-end NMS-free detection for improved latency.
- [**YOLO11**](https://docs.ultralytics.com/models/yolo11/): The latest cutting-edge model from Ultralytics, designed for peak performance.
