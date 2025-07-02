# Pruning for Vision Representation

This repository provides code and tools for research on **pruning neural networks for vision representation tasks**, including classification, object detection, and segmentation. The project leverages PyTorch and TorchVision, and supports various models such as ResNet and ViT (Vision Transformers).

## Features

- **Pruning Methods**: Implementations for different pruning strategies to reduce model size and improve efficiency.
- **Support for Multiple Architectures**: Includes scripts for ResNet, ViT, MobileNet, VGG, and more.
- **Training & Evaluation**: End-to-end scripts for training pruned models and evaluating them on standard vision benchmarks (e.g., ImageNet, VOC, COCO).
- **Visualization & Analysis**: Utilities for comparing model vs. human performance, saving high-quality plots, and analyzing learned representations.
- **Explainability**: Some explainability techniques implemented in this repository are taken from the [Captum library](https://captum.ai/).
- **Model Export**: Convenient functions for saving and loading model weights, including hashing for reproducibility.

## News

- 📄 The corresponding paper has been **accepted at ICIAP 2025**.

## Getting Started

### Requirements

- Python 3.8+
- PyTorch (>=1.10)
- TorchVision
- numpy, matplotlib, opencv-python, Pillow, and other standard ML libraries

### Datasets

Prepare your datasets (e.g., ImageNet, VOC, COCO) and organize them as follows:

```
your_data_path/
    train/
    val/
```
Update dataset paths in your scripts as needed.

## Usage

### Training a Pruned Model

Example for ImageNet:

```bash
python train.py --model resnet18 --data-path /path/to/imagenet --pruning-method snip --target-sparsity 0.5 --epochs 90 --output-dir ./results
```

### Running Object Discovery (LOST)

```bash
python main_lost.py --arch vit_small --dataset VOC07 --set train --models-dir /path/to/models --data-path /path/to/data
```

### Visualization

Scripts such as `mvh_triple_comparison.py` and `mvh_performance_rn50_vs_rn18.py` generate high-quality performance comparison plots.

## Repository Structure

- `train.py` — Training loop with support for pruning and logging.
- `main_lost.py` — Object discovery with LOST.
- `explain.py` — Explanation and analysis tools (with techniques from [Captum](https://captum.ai/)).
- `utils.py` — Utilities for model export, reproducibility, and more.
- `datasets.py` — Dataset loading and handling.
- `cluster_for_OD.py`, `mvh_triple_comparison.py`, etc. — Additional experiments and analyses.

## Acknowledgements

- Some explainability techniques are taken from the [Captum library](https://captum.ai/).
- This code builds on top of PyTorch and TorchVision libraries. If you use this repository for your research, please consider citing the relevant papers and this repository.

## License

This project is for research purposes. See individual file headers for license information.

---

**Maintained by [EIDOSLAB](https://eidos.di.unito.it/).**  
For questions or contributions, please open an issue or pull request.
