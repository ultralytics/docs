---
description: Discover the key differences between YOLOv5 and RTDETRv2, from architecture to accuracy, and find the best object detection model for your project.
keywords: YOLOv5, RTDETRv2, object detection comparison, YOLOv5 vs RTDETRv2, Ultralytics models, model performance, computer vision, object detection, RTDETR, YOLOv5 features, transformer architecture
---

# YOLOv5 vs RTDETRv2: Detailed Technical Comparison

Choosing the optimal object detection model is a critical decision for computer vision projects. Ultralytics offers a diverse range of models to address various project needs. This page delivers a technical comparison between [Ultralytics YOLOv5](https://docs.ultralytics.com/models/yolov5/) and [RTDETRv2](https://docs.ultralytics.com/models/rtdetr/), emphasizing their architectural distinctions, performance benchmarks, and suitability for different applications.

<script async src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["RTDETRv2", "YOLOv5"]'></canvas>

## YOLOv5: Optimized for Speed and Efficiency

[Ultralytics YOLOv5](https://docs.ultralytics.com/models/yolov5/) is a widely-adopted one-stage object detector celebrated for its rapid inference speed and operational efficiency. YOLOv5's architecture is composed of:

- **Backbone:** CSPDarknet53, responsible for feature extraction.
- **Neck:** PANet, utilized for feature fusion.
- **Head:** YOLOv5 head, designed for detection tasks.

YOLOv5 is available in multiple sizes (n, s, m, l, x), providing users with options to balance speed and accuracy based on their specific requirements.

**Strengths:**

- **Inference Speed:** YOLOv5 excels in speed, making it an excellent choice for real-time applications such as [security alarm systems](https://docs.ultralytics.com/guides/security-alarm-system/).
- **Efficiency:** YOLOv5 models are compact, demanding fewer computational resources, suitable for edge deployment like [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/) and [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/).
- **Versatility:** Adaptable to various hardware environments, including resource-constrained devices.
- **User-Friendliness:** Well-documented and straightforward to implement using the Ultralytics [Python package](https://pypi.org/project/ultralytics/) and [Ultralytics HUB](https://www.ultralytics.com/hub).

**Weaknesses:**

- **Accuracy Trade-off:** While achieving high accuracy, larger models like RTDETRv2 may offer superior mAP, particularly in complex scenarios.

**Ideal Use Cases:**

- Real-time object detection scenarios including video surveillance and [AI in traffic management](https://www.ultralytics.com/blog/optimizingtraffic-management-with-ultralytics-yolo11).
- Edge computing and mobile deployments.
- Applications requiring rapid processing, such as [robotics](https://www.ultralytics.com/blog/from-algorithms-to-automation-ais-role-in-robotics) ([ROS Quickstart](https://docs.ultralytics.com/guides/ros-quickstart/)) and [autonomous vehicles](https://www.ultralytics.com/solutions/ai-in-self-driving).

[Learn more about YOLOv5](https://docs.ultralytics.com/models/yolov5/){ .md-button }

## RTDETRv2: High-Accuracy Real-Time Detection Transformer

**RTDETRv2** ([Real-Time Detection Transformer v2](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#readme)) is a state-of-the-art object detection model prioritizing high accuracy and real-time performance. It was introduced in a paper titled "[RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer](https://arxiv.org/abs/2407.17140)" on 2023-04-17 by authors Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu from Baidu. Built on a Vision Transformer (ViT) architecture, RTDETRv2 excels in applications demanding precise object localization and classification.

**Architecture and Key Features:**

RTDETRv2 leverages a transformer-based architecture, enabling it to capture global context within images through self-attention mechanisms. This approach allows the model to weigh the importance of different image regions, leading to enhanced feature extraction and improved accuracy, especially in complex scenes.

**Strengths:**

- **Superior Accuracy:** Transformer architecture provides enhanced object detection accuracy, particularly in complex environments as demonstrated in scenarios like [vision-ai-in-crowd-management](https://www.ultralytics.com/blog/vision-ai-in-crowd-management).
- **Real-Time Capability:** Achieves competitive inference speeds, particularly when using hardware acceleration like NVIDIA T4 GPUs.
- **Robust Feature Extraction:** Vision Transformers effectively capture global context and intricate details, beneficial in applications such as [using computer vision to analyse satellite imagery](https://www.ultralytics.com/blog/using-computer-vision-to-analyse-satellite-imagery).

**Weaknesses:**

- **Larger Model Size:** RTDETRv2 models, especially larger variants, have a higher parameter count and FLOPs than YOLOv5, necessitating more computational resources.
- **Inference Speed:** While real-time capable, inference speed may be lower compared to the fastest YOLOv5 models, especially on less powerful devices.

**Ideal Use Cases:**

RTDETRv2 is optimally suited for applications where accuracy is paramount and computational resources are sufficient. These include:

- **Autonomous Driving:** For reliable and precise environmental perception in [AI in self-driving cars](https://www.ultralytics.com/solutions/ai-in-self-driving).
- **Robotics:** Enabling robots to accurately interact with their surroundings, essential for tasks discussed in "[From Algorithms to Automation: AI's Role in Robotics](https://www.ultralytics.com/blog/from-algorithms-to-automation-ais-role-in-robotics)".
- **Medical Imaging:** For precise anomaly detection, aiding in diagnostics as highlighted in [AI in Healthcare](https://www.ultralytics.com/solutions/ai-in-healthcare) and potentially useful in areas like [using-yolo11-for-tumor-detection-in-medical-imaging](https://www.ultralytics.com/blog/using-yolo11-for-tumor-detection-in-medical-imaging).
- **High-Resolution Image Analysis:** Applications requiring detailed analysis of large images, like satellite imagery or industrial inspection, as seen in [improving-manufacturing-with-computer-vision](https://www.ultralytics.com/blog/improving-manufacturing-with-computer-vision).

[Learn more about RTDETRv2](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

## Model Comparison Table

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | 54.3                 | -                              | 15.03                               | 76                 | 259               |
|            |                       |                      |                                |                                     |                    |                   |
| YOLOv5n    | 640                   | 28.0                 | 73.6                           | 1.12                                | 2.6                | 7.7               |
| YOLOv5s    | 640                   | 37.4                 | 120.7                          | 1.92                                | 9.1                | 24.0              |
| YOLOv5m    | 640                   | 45.4                 | 233.9                          | 4.03                                | 25.1               | 64.2              |
| YOLOv5l    | 640                   | 49.0                 | 408.4                          | 6.61                                | 53.2               | 135.0             |
| YOLOv5x    | 640                   | 50.7                 | 763.2                          | 11.89                               | 97.2               | 246.4             |

## Conclusion

Both RTDETRv2 and YOLOv5 are robust object detection models, each designed for distinct priorities. RTDETRv2 is favored when accuracy is paramount and computational resources are available. Conversely, YOLOv5 excels in scenarios requiring real-time performance and efficiency, especially on resource-limited platforms.

For users exploring other models, Ultralytics offers a broad model zoo, including:

- **YOLOv8** and **YOLOv11**: Successors to YOLOv5, providing further advancements in performance and efficiency as highlighted in "[Ultralytics YOLOv8 Turns One: A Year of Breakthroughs and Innovations](https://www.ultralytics.com/blog/ultralytics-yolov8-turns-one-a-year-of-breakthroughs-and-innovations)" and "[Ultralytics YOLO11 Has Arrived: Redefine What's Possible in AI](https://www.ultralytics.com/blog/ultralytics-yolo11-has-arrived-redefine-whats-possible-in-ai)".
- **YOLO-NAS**: Models architected with Neural Architecture Search for optimized performance ([YOLO-NAS by Deci AI](https://docs.ultralytics.com/models/yolo-nas/)).
- **FastSAM** and **MobileSAM**: For real-time instance segmentation tasks ([FastSAM](https://docs.ultralytics.com/models/fast-sam/) and [MobileSAM](https://docs.ultralytics.com/models/mobile-sam/)).

Selecting between RTDETRv2, YOLOv5, or other Ultralytics models should be based on the specific demands of your computer vision project, carefully considering the balance between accuracy, speed, and resource availability. Consult the [Ultralytics Documentation](https://docs.ultralytics.com/models/) and [GitHub repository](https://github.com/ultralytics/ultralytics) for comprehensive details and implementation guides.