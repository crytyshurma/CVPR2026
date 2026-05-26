# Coded-E2LF (CVPR 2026) — CPU/GPU Compatible Implementation

This repository contains an implementation and experimentation setup for the CVPR 2026 paper:

> **Tomoya Tsuchida, Keita Takahashi, Chihiro Tsutake, Toshiaki Fujii, Hajime Nagahara**  
> *"Coded-E2LF: Coded Aperture Light Field Imaging from Events"*  
> IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

The project reconstructs an 8×8 Light Field (LF) from a small number of coded event images using a reconstruction CNN.

This repository has been modified to support both:
- CPU execution
- GPU execution (CUDA)

---

# Project Overview

The pipeline reconstructs a full Light Field using:
- coded aperture imaging
- event camera simulation / event data
- neural reconstruction network

Two testing pipelines are included:

---

## 1. Light Field Simulation Pipeline (`test_from_LF.py`)

This pipeline:
1. Takes an original 8×8 Light Field
2. Simulates coded aperture acquisition
3. Generates event images
4. Reconstructs the Light Field
5. Saves reconstructed LF views

### Pipeline Flow

```text
Original 64 LF Views
        ↓
Coded Aperture Simulation
        ↓
3 Event Images Generated
        ↓
Reconstruction CNN
        ↓
Reconstructed 64 LF Views
```

---

## 2. Real Event Data Pipeline (`test_from_acquired_data.py`)

This pipeline:
1. Takes real captured event camera data
2. Runs the reconstruction CNN
3. Generates reconstructed LF views

### Pipeline Flow

```text
Real Event Camera Data (.npy)
        ↓
Load 3 Event Images
        ↓
Normalize / Preprocess
        ↓
Reconstruction CNN
        ↓
Reconstructed 64 LF Views
        ↓
Save Reconstructed Outputs
```

---

# Repository Structure

```text
├── Acquired_data/              # Real event camera data
│   ├── Animals/
│   └── Fish/
│
├── LF_data/                    # Ground-truth LF dataset
│   └── 8x8/BasicLFSR/Lego_Knights/
│
├── model_weights/              # Pretrained weights
│   ├── AcqNet.pth
│   └── RecNet.pth
│
├── utils/                      # Utility functions
│
├── Dst/                        # Output directory
│
├── model.py                    # Network architectures
├── option.py                   # Runtime arguments
├── test_from_LF.py             # LF simulation pipeline
├── test_from_acquired_data.py  # Real event reconstruction pipeline
├── requirements.txt
└── README.md
```

---

# Prerequisites

## Python

Recommended:
- Python 3.13+

---

## Install Dependencies

### Create Virtual Environment (Optional)

### Windows

```bash
python -m venv .venv_cpu
.\.venv_cpu\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv_cpu
source .venv_cpu/bin/activate
```

---

## Install PyTorch

### CPU Version

```bash
pip install torch torchvision torchaudio
```

### CUDA Version (GPU)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

## Install Remaining Requirements

```bash
pip install -r requirements.txt
```

---

# Device Configuration

The repository supports:
- CPU
- GPU (CUDA)
- Auto device selection

Inside `option.py`:

```python
parser.add_argument('--device', type=str, default='auto')
```

Possible values:
- `cpu`
- `cuda:0`
- `auto`

---

# Running the Project

---

## 1. Run Light Field Simulation Pipeline

```bash
python test_from_LF.py
```

### Outputs

```text
Dst/test/from_LF_data/
```

Contains:
- generated event images
- reconstructed LF views

---

## 2. Run Real Event Reconstruction Pipeline

```bash
python test_from_acquired_data.py
```

### Outputs

```text
Dst/test/from_acquired_data/
```

Contains:
- reconstructed LF views from real event data

---

# Example Outputs

## Generated Event Images

```text
event_image_00000.png
event_image_00001.png
event_image_00002.png
```

## Reconstructed Light Field Views

```text
00_00.png
00_01.png
...
07_07.png
```

Total:
- 64 reconstructed viewpoints

---

# Notes

- The provided scripts perform inference only.
- Accuracy evaluation metrics such as PSNR/SSIM are not included in the current repository.
- CPU execution is supported for easier experimentation.
- GPU acceleration is recommended for faster inference and future model training.

---

# Citation

If you use this repository for research purposes, please cite:

```bibtex
@inproceedings{codede2lf2026,
  title={Coded-E2LF: Coded Aperture Light Field Imaging from Events},
  author={Tsuchida, Tomoya and Takahashi, Keita and Tsutake, Chihiro and Fujii, Toshiaki and Nagahara, Hajime},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```

---

# License

This project follows the original MIT License provided by the authors.
