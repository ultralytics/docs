---
comments: true
description: Compare RTDETRv2 and YOLOv9 object detection models. Explore performance, strengths, weaknesses, and ideal use cases to make an informed decision.
keywords: RTDETRv2, YOLOv9, object detection, Ultralytics models, transformer vision, YOLO series, real-time object detection, model comparison, Vision Transformers, computer vision
---

# RTDETRv2 vs YOLOv9: Comparing Transformer and CNN Architectures for Real-Time Detection

In the rapidly evolving landscape of computer vision, choosing the right object detection architecture is critical for balancing speed, accuracy, and deployment feasibility. This comparison delves into **RTDETRv2**, a cutting-edge real-time transformer model by Baidu, and **YOLOv9**, a CNN-based architecture by Academia Sinica that introduces Programmable Gradient Information (PGI). Both models represent significant leaps in their respective paradigms, offering unique advantages for researchers and developers.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["RTDETRv2", "YOLOv9"]'></canvas>

## Performance Benchmarks

The following table provides a direct comparison of key performance metrics on the COCO dataset. It highlights the trade-offs between the transformer-based approach of RTDETRv2 and the efficient layer aggregation strategies of YOLOv9.

| Model      | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| RTDETRv2-s | 640                   | 48.1                 | -                              | 5.03                                | 20                 | 60                |
| RTDETRv2-m | 640                   | 51.9                 | -                              | 7.51                                | 36                 | 100               |
| RTDETRv2-l | 640                   | 53.4                 | -                              | 9.76                                | 42                 | 136               |
| RTDETRv2-x | 640                   | 54.3                 | -                              | 15.03                               | 76                 | 259               |
|            |                       |                      |                                |                                     |                    |                   |
| YOLOv9t    | 640                   | 38.3                 | -                              | **2.3**                             | **2.0**            | **7.7**           |
| YOLOv9s    | 640                   | 46.8                 | -                              | 3.54                                | 7.1                | 26.4              |
| YOLOv9m    | 640                   | 51.4                 | -                              | 6.43                                | 20.0               | 76.3              |
| YOLOv9c    | 640                   | 53.0                 | -                              | 7.16                                | 25.3               | 102.1             |
| YOLOv9e    | 640                   | **55.6**             | -                              | 16.77                               | 57.3               | 189.0             |

## RTDETRv2: Refining the Real-Time Transformer

**RTDETRv2** (Real-Time Detection Transformer v2) builds upon the foundation of the original RT-DETR, aiming to further bridge the gap between transformer accuracy and the speed requirements of real-time applications. Developed by researchers at [Baidu](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch), this model addresses the computational bottlenecks often associated with Vision Transformers (ViTs) while leveraging their superior global context understanding.

### Key Architectural Features

- **Efficient Hybrid Encoder:** Unlike traditional DETR models that rely heavily on computationally expensive attention mechanisms across all scales, RTDETRv2 utilizes a hybrid encoder. This component decouples intra-scale interactions and cross-scale fusion, significantly reducing the computational cost while maintaining robust feature representation.
- **IoU-Aware Query Selection:** To improve initialization, RTDETRv2 employs an Intersection-over-Union (IoU) aware query selection mechanism. This focuses the model's attention on the most relevant image regions early in the decoding process, leading to faster convergence and higher accuracy.
- **Adaptable Inference Speed:** A standout feature is the ability to adjust inference speed by modifying the number of decoder layers without retraining. This flexibility allows developers to fine-tune the model for different hardware constraints dynamically.

[Learn more about RTDETRv2](https://docs.ultralytics.com/models/rtdetr/){ .md-button }

### Ideal Use Cases for RTDETRv2

The transformer architecture excels in scenarios requiring complex global context understanding.

1.  **Crowded Scene Analysis:** In [crowd management](https://docs.ultralytics.com/guides/queue-management/) and surveillance, where occlusion is common, the global attention mechanism helps track individuals more effectively than local CNN kernels.
2.  **Autonomous Navigation:** For [self-driving cars](https://www.ultralytics.com/solutions/ai-in-automotive), understanding the relationship between distant objects (e.g., a traffic light and a car far down the road) is crucial.
3.  **Defect Detection in Manufacturing:** Identifying subtle [defects in manufacturing](https://www.ultralytics.com/blog/quality-inspection-in-manufacturing-traditional-vs-deep-learning-methods) often requires comparing local textures against the global structure of a component, a task where transformers often outperform CNNs.

!!! note "Deployment Consideration"

    While RTDETRv2 offers excellent accuracy, transformer models generally require more CUDA memory during training and inference compared to CNNs like YOLOv9. Ensure your [hardware resources](https://docs.ultralytics.com/guides/nvidia-jetson/) are sufficient for transformer deployments.

## YOLOv9: Programmable Gradient Information

**YOLOv9**, released by researchers at Academia Sinica, introduces a novel concept called Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN). This architecture aims to solve the "information bottleneck" problem inherent in deep neural networks, where essential data is lost as it passes through successive layers.

### Key Architectural Innovations

- **Programmable Gradient Information (PGI):** PGI is an auxiliary supervision framework that ensures reliable gradient generation for updating network weights. By preserving critical information throughout the deep network, PGI allows YOLOv9 to train deeper models without the degradation in convergence speed or accuracy typically seen in very deep CNNs.
- **GELAN Architecture:** The Generalized Efficient Layer Aggregation Network (GELAN) optimizes parameter utilization. It allows for the flexible integration of various computational blocks, making the model lightweight and fast while maximizing accuracy. This efficiency is evident in the benchmarks, where YOLOv9t achieves incredible speeds on T4 GPUs.
- **Reversible Functions:** Inspired by dynamic detachment, YOLOv9 employs reversible functions to mitigate information loss, ensuring that the model retains the necessary data to make accurate predictions even in lightweight configurations.

[Learn more about YOLOv9](https://docs.ultralytics.com/models/yolov9/){ .md-button }

### Ideal Use Cases for YOLOv9

YOLOv9's efficiency makes it a top contender for edge deployment and resource-constrained environments.

1.  **Edge AI on IoT Devices:** With the tiny (t) and small (s) variants offering high speed and low FLOPs, YOLOv9 is perfect for [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/) and other embedded systems.
2.  **Real-Time Sports Analytics:** The high frame rates achievable with YOLOv9 make it ideal for tracking fast-moving objects, such as [golf balls](https://docs.ultralytics.com/reference/solutions/ai_gym/) or players in sports broadcasts.
3.  **Aerial Imagery and Drone Surveillance:** The improved information retention helps in detecting small objects from high altitudes, useful for [agriculture monitoring](https://www.ultralytics.com/solutions/ai-in-agriculture) and search and rescue missions.

## Comparison and Conclusion

When choosing between RTDETRv2 and YOLOv9, the decision often comes down to the specific constraints of your deployment environment and the nature of your visual data.

- **Choose RTDETRv2 if:** You have access to powerful GPU hardware (like NVIDIA TensorRT-enabled devices) and your application benefits from the superior global context understanding of transformers. The adaptable inference speed is also a major plus for variable-load systems.
- **Choose YOLOv9 if:** You prioritize raw inference speed on CPU or edge devices, or if you need a model with a very small memory footprint (specifically the `t` and `s` variants). The CNN architecture is generally easier to train on consumer-grade hardware due to lower VRAM requirements.

Both models are fully supported within the Ultralytics ecosystem, allowing you to easily [train](https://docs.ultralytics.com/modes/train/), [validate](https://docs.ultralytics.com/modes/val/), and [export](https://docs.ultralytics.com/modes/export/) them using a unified API. This seamless integration ensures that you can experiment with both architectures to find the perfect fit for your project without rewriting your codebase.

For developers looking for the absolute latest in performance and ease of use, we also recommend exploring **[YOLO26](https://docs.ultralytics.com/models/yolo26/)**. Released in January 2026, YOLO26 introduces an end-to-end NMS-free design, MuSGD optimizer for stable training, and significant speedups for CPU inference, making it the new state-of-the-art recommendation for most computer vision tasks.

## Code Examples

### Training RTDETRv2 with Ultralytics

The Ultralytics Python API makes training complex transformer models straightforward.

```python
from ultralytics import RTDETR

# Load a COCO-pretrained RTDETR-l model
model = RTDETR("rtdetr-l.pt")

# Train the model on your custom dataset
# Ensure your hardware has sufficient VRAM for transformer training
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference on a test image
results = model("path/to/image.jpg")
```

### Training YOLOv9 with Ultralytics

Similarly, YOLOv9 can be trained and deployed with minimal code changes.

```python
from ultralytics import YOLO

# Load a YOLOv9c model from pretrained weights
model = YOLO("yolov9c.pt")

# Train the model on the COCO8 example dataset
# YOLO models typically require less VRAM than equivalent transformers
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Export the model to ONNX format for deployment
model.export(format="onnx")
```

## Citations

**RTDETRv2**
Authors: Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu  
Organization: [Baidu](https://github.com/lyuwenyu/RT-DETR)  
Date: 2024-07  
Arxiv: [https://arxiv.org/abs/2407.17140](https://arxiv.org/abs/2407.17140)

**YOLOv9**
Authors: Chien-Yao Wang, Hong-Yuan Mark Liao  
Organization: [Academia Sinica](https://www.sinica.edu.tw/en)  
Date: 2024-02-21  
Arxiv: [https://arxiv.org/abs/2402.13616](https://arxiv.org/abs/2402.13616)
