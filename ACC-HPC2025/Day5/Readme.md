# Day 5: High-Performance Deep Learning Inference & OpenVINO™ Deployment
### ACC-HPC High Performance Computing & Edge AI Workshop

Welcome to Day 5 of the ACC-HPC series. This module transitions from training neural network models (Days 1–4) to deploying production-grade, hardware-optimized deep learning inference pipelines on high-performance compute architectures.

---

## 🎯 Learning Objectives
1. Understand the core paradigm shift: **Model Training vs. High-Performance Inference**.
2. Master the **Intel® OpenVINO™ Toolkit** architecture (`ov.Core`, Model Optimizer, OpenVINO IR, NNCF).
3. Convert deep learning models from PyTorch (`.pt`) and TensorFlow (`.keras`) into optimized OpenVINO Intermediate Representations (`.xml` + `.bin`).
4. Apply **Post-Training Quantization (PTQ)** using NNCF to compress models from FP32 to INT8 with $<1\%$ accuracy loss.
5. Engineer low-latency and high-throughput inference engines using `AsyncInferQueue` and multi-stream hardware execution.
6. Benchmark HPC performance metrics: **p50/p95/p99 Latency (ms)**, **Throughput (FPS)**, **Memory Footprint**, and **Energy Efficiency**.

---

## 📓 Core Master Notebook
The complete hands-on curriculum, code implementations, interactive visualizations, and diagnostics are located in the root repository notebook:
👉 **[`../../OpenVINO_Inference_Deployment.ipynb`](../../OpenVINO_Inference_Deployment.ipynb)**

---

## 🧪 Laboratory Exercises & Assignments

### Assignment 1: PyTorch to OpenVINO IR Conversion & In-Graph Preprocessing
1. Load the trained CIFAR-10 Convolutional Neural Network from Day 4 (`NeuralNetworks_PyTorch.ipynb`).
2. Use `openvino.convert_model()` to serialize the network into an OpenVINO Intermediate Representation (`model.xml` and `model.bin`).
3. Using OpenVINO's `PrePostProcessor` API:
   - Configure input shape to `[1, 32, 32, 3]` with `U8` precision and `NHWC` layout.
   - Embed RGB normalization (mean `[0.4914, 0.4822, 0.4465]`, std `[0.2470, 0.2435, 0.2616]`) directly into the execution graph.
   - Verify that your input pipeline accepts raw `uint8` NumPy image arrays without external tensor transformations.

### Assignment 2: Post-Training INT8 Quantization with NNCF
1. Instantiate an unlabelled calibration dataset of 300 random test samples from CIFAR-10.
2. Apply `nncf.quantize()` with `preset=nncf.QuantizationPreset.PERFORMANCE`.
3. Save the quantized model as `cifar10_int8.xml` and `cifar10_int8.bin`.
4. Calculate and compare:
   - File size reduction on disk (FP32 vs INT8).
   - Validation classification accuracy on the 10,000-sample CIFAR-10 test set (verify accuracy retention $\ge 98\%$).

### Assignment 3: Latency vs. Throughput Benchmarking Harness
1. Write a custom Python benchmarking function evaluating:
   - Baseline PyTorch CPU inference latency ($B=1$, 500 iterations).
   - OpenVINO FP32 latency on CPU ($B=1$).
   - OpenVINO INT8 latency on CPU ($B=1$).
2. Compute the 50th, 95th, and 99th percentile latencies (p50, p95, p99).
3. Evaluate throughput (Frames Per Second) across varying batch sizes: $B \in [1, 2, 4, 8, 16, 32, 64]$.
4. Plot a dual-axis visualization showing **Throughput (FPS)** vs **p95 Latency (ms)** across batch sizes.

### Assignment 4: Asynchronous Multi-Stream Inference Queue
1. Configure an `ov.AsyncInferQueue` with optimal stream concurrency for your local machine (`jobs=0` for auto-detected parallel streams).
2. Set up an asynchronous callback function that extracts top-1 predictions and updates a non-blocking queue.
3. Simulate streaming image input (e.g., simulating 1,000 video frames) and compare the processing speed of synchronous single-image inference vs. the asynchronous worker queue.

---

## 📚 Essential Reading & References
- Intel OpenVINO Documentation: [https://docs.openvino.ai/](https://docs.openvino.ai/)
- NNCF Quantization Guide: [https://github.com/openvinotoolkit/nncf](https://github.com/openvinotoolkit/nncf)
- OpenVINO Model Zoo: [https://github.com/openvinotoolkit/open_model_zoo](https://github.com/openvinotoolkit/open_model_zoo)
