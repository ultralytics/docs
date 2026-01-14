# YOLO11 vs. YOLO26: Evolution of Real-Time Vision AI

The field of computer vision is advancing rapidly, and Ultralytics continues to lead the charge with state-of-the-art object detection models. This comparison explores the architectural evolution, performance metrics, and practical applications of **YOLO11**, released in late 2024, and the groundbreaking **YOLO26**, released in January 2026. While both models represent the pinnacle of vision AI at their respective launch times, YOLO26 introduces significant architectural shifts that redefine efficiency and speed for edge deployment.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO11", "YOLO26"]'></canvas>

## Model Overview

### YOLO11

**Authors:** Glenn Jocher and Jing Qiu  
**Organization:** [Ultralytics](https://www.ultralytics.com)  
**Date:** 2024-09-27  
**GitHub:** [Ultralytics Repository](https://github.com/ultralytics/ultralytics)  
**Docs:** [YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/)

YOLO11 marked a significant refinement in the YOLO series, offering a 22% reduction in parameters compared to [YOLOv8](https://docs.ultralytics.com/models/yolov8/) while improving detection accuracy. It introduced an enhanced architectural design that balanced speed and precision, making it a reliable choice for diverse computer vision tasks ranging from [object detection](https://docs.ultralytics.com/tasks/detect/) to [instance segmentation](https://docs.ultralytics.com/tasks/segment/).

[Learn more about YOLO11](https://docs.ultralytics.com/models/yolo11/){ .md-button }

### YOLO26

**Authors:** Glenn Jocher and Jing Qiu  
**Organization:** [Ultralytics](https://www.ultralytics.com)  
**Date:** 2026-01-14  
**GitHub:** [Ultralytics Repository](https://github.com/ultralytics/ultralytics)  
**Docs:** [YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26/)

YOLO26 represents a paradigm shift with its **natively end-to-end NMS-free design**, eliminating the need for Non-Maximum Suppression post-processing. This innovation, first pioneered in [YOLOv10](https://docs.ultralytics.com/models/yolov10/), significantly simplifies deployment pipelines and reduces latency. YOLO26 is specifically optimized for edge computing, delivering up to **43% faster CPU inference** and incorporating novel training techniques like the **MuSGD Optimizer**—a hybrid of SGD and Muon inspired by LLM training innovations.

[Learn more about YOLO26](https://docs.ultralytics.com/models/yolo26/){ .md-button }

!!! tip "End-to-End Latency Advantage"

    By removing the NMS step, YOLO26 provides consistent inference times regardless of the number of objects detected in a scene. This is crucial for real-time applications like [autonomous driving](https://www.ultralytics.com/solutions/ai-in-automotive), where post-processing spikes can cause dangerous delays.

## Performance Comparison

The table below highlights the performance improvements of YOLO26 over YOLO11. Note the substantial gains in CPU speed, making YOLO26 exceptionally capable for devices without dedicated GPUs, such as Raspberry Pis or mobile phones.

| Model       | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ----------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| YOLO11n     | 640                   | 39.5                 | 56.1                           | **1.5**                             | 2.6                | 6.5               |
| YOLO11s     | 640                   | 47.0                 | 90.0                           | **2.5**                             | 9.4                | 21.5              |
| YOLO11m     | 640                   | 51.5                 | 183.2                          | 4.7                                 | **20.1**           | **68.0**          |
| YOLO11l     | 640                   | 53.4                 | 238.6                          | 6.2                                 | 25.3               | 86.9              |
| YOLO11x     | 640                   | 54.7                 | 462.8                          | **11.3**                            | 56.9               | 194.9             |
|             |                       |                      |                                |                                     |                    |                   |
| **YOLO26n** | 640                   | **40.9**             | **38.9**                       | 1.7                                 | **2.4**            | **5.4**           |
| **YOLO26s** | 640                   | **48.6**             | **87.2**                       | 2.5                                 | 9.5                | **20.7**          |
| **YOLO26m** | 640                   | **53.1**             | 220.0                          | 4.7                                 | 20.4               | 68.2              |
| **YOLO26l** | 640                   | **55.0**             | 286.2                          | 6.2                                 | **24.8**           | **86.4**          |
| **YOLO26x** | 640                   | **57.5**             | 525.8                          | 11.8                                | **55.7**           | **193.9**         |

## Architectural Deep Dive

### YOLO11 Architecture

YOLO11 built upon the [CSPNet](https://github.com/WongKinYiu/CrossStagePartialNetworks) backbone concept, refining the feature extraction layers to capture more granular details. It utilized a standard anchor-free detection head and relied on Distribution Focal Loss (DFL) to refine bounding box regression. While highly effective, the reliance on NMS meant that inference speed could fluctuate based on scene density, a common bottleneck in [smart city surveillance](https://www.ultralytics.com/blog/smart-surveillance-ultralytics-yolo11).

### YOLO26 Architecture

YOLO26 introduces several radical changes designed for efficiency and stability:

1.  **NMS-Free End-to-End:** The model predicts a fixed set of bounding boxes with one-to-one matching during training, removing the heuristic NMS step during inference.
2.  **DFL Removal:** Distribution Focal Loss was removed to simplify the export process to formats like [ONNX](https://docs.ultralytics.com/integrations/onnx/) and [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/), enhancing compatibility with low-power edge devices.
3.  **MuSGD Optimizer:** Inspired by Moonshot AI's Kimi K2 and Large Language Model (LLM) training, this hybrid optimizer combines SGD and Muon to ensure faster convergence and more stable training runs, reducing the "loss spikes" often seen in large-scale vision training.
4.  **ProgLoss + STAL:** New loss functions (Progressive Loss and Soft-Target Assignment Loss) specifically target small-object recognition, providing a massive boost for [aerial imagery analysis](https://www.ultralytics.com/blog/build-ai-powered-drone-applications-with-ultralytics-yolo11) and IoT sensors.

## Task Versatility

Both models support a wide array of tasks within the Ultralytics ecosystem, ensuring developers can switch models without rewriting their pipelines.

- **Detection:** Standard bounding box detection.
- **Segmentation:** Pixel-level masks. YOLO26 adds a specific semantic segmentation loss and multi-scale proto for better mask quality.
- **Classification:** Whole-image categorization.
- **Pose Estimation:** Keypoint detection. YOLO26 utilizes **Residual Log-Likelihood Estimation (RLE)** for higher precision in complex poses, beneficial for [sports analytics](https://www.ultralytics.com/blog/using-pose-estimation-to-perfect-your-running-technique).
- **OBB (Oriented Bounding Box):** Rotated boxes for aerial or angled objects. YOLO26 features a specialized angle loss to resolve boundary discontinuity issues common in satellite imagery.

## Training and Usage

One of the hallmarks of the [Ultralytics ecosystem](https://github.com/ultralytics/ultralytics) is the unified API. Whether you are using YOLO11 or upgrading to YOLO26, the code remains virtually identical, minimizing technical debt.

### Python Example

Here is how you can train the new YOLO26 model using the same familiar interface used for YOLO11. This example demonstrates training on the [COCO8 dataset](https://docs.ultralytics.com/datasets/detect/coco8/), a small 8-image dataset perfect for testing.

```python
from ultralytics import YOLO

# Load the latest YOLO26 nano model
model = YOLO("yolo26n.pt")

# Train the model
# The MuSGD optimizer is handled automatically internally for YOLO26 models
results = model.train(
    data="coco8.yaml",
    epochs=100,
    imgsz=640,
    device="cpu",  # Use '0' for GPU
)

# Run inference
results = model("https://ultralytics.com/images/bus.jpg")
results[0].show()
```

### CLI Example

The command-line interface is equally streamlined, allowing for rapid experimentation and [model benchmarking](https://docs.ultralytics.com/modes/benchmark/).

```bash
# Train YOLO26n on the COCO8 dataset
yolo train model=yolo26n.pt data=coco8.yaml epochs=100 imgsz=640

# Export to ONNX for simplified edge deployment
yolo export model=yolo26n.pt format=onnx
```

## Ideal Use Cases

**Choose YOLO11 if:**

- You have an existing production pipeline highly tuned for YOLO11 and cannot afford validation time for a new architecture.
- Your deployment hardware has specific optimizations for the YOLO11 layer structure that haven't been updated for YOLO26 yet.

**Choose YOLO26 if:**

- **Edge Deployment is Critical:** The removal of NMS and DFL makes YOLO26 the superior choice for [Android/iOS apps](https://docs.ultralytics.com/guides/model-deployment-options/) and embedded systems where CPU cycles are precious.
- **Small Object Detection:** The ProgLoss and STAL functions make it significantly better for [identifying pests in agriculture](https://www.ultralytics.com/blog/leverage-ultralytics-yolo11-object-detection-for-pest-control) or distant objects in drone footage.
- **Training Stability:** If you are training on massive custom datasets and have experienced divergence issues, the MuSGD optimizer in YOLO26 offers a more stable training path.
- **Simplest Export:** The end-to-end architecture exports more cleanly to formats like CoreML and TensorRT without requiring complex external NMS plugins.

For developers interested in exploring other options within the Ultralytics family, models like [YOLOv10](https://docs.ultralytics.com/models/yolov10/) (the precursor to end-to-end YOLO) or [YOLO-World](https://docs.ultralytics.com/models/yolo-world/) (for open-vocabulary detection) are also fully supported.

## Conclusion

While YOLO11 remains a robust and highly capable model, **YOLO26** establishes a new baseline for what is possible in real-time computer vision. By integrating LLM-inspired training dynamics and simplifying the inference pipeline through an NMS-free design, Ultralytics has created a model that is not only more accurate but also significantly easier to deploy in the real world.

The Ultralytics ecosystem ensures that upgrading is seamless. With lower memory requirements during training and faster CPU speeds during inference, YOLO26 is the recommended starting point for all new projects in 2026.

[Get Started with Ultralytics](https://docs.ultralytics.com/quickstart/){ .md-button }
