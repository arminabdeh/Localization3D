# LUENN: PyTorch Package for 3D Single Molecule Localization Microscopy  

> **LUENN (Localization3D)** is an open-source PyTorch package for **3D Single Molecule Localization Microscopy (SMLM)**.  
> It delivers **end-to-end pipelines** for synthetic data generation, deep learning model training, sub-pixel emitter localization, and 3D super-resolution image rendering.  
> Validated in peer-reviewed research, LUENN enables **fast, accurate, and robust live-cell imaging**, bridging the gap between computational microscopy and biomedical applications.  

📄 Results published in: [Applied Optics, 2024](https://doi.org/10.1364/AO.539076)  

---

## ✨ Key Features
- **Data Generation** → create synthetic microscopy datasets for training and benchmarking.  
- **Sampling Utilities** → extract training/validation subsets efficiently from large datasets.  
- **Custom Model Training** → PyTorch-based CNNs for SMLM with flexible architectures and loss functions.  
- **Post-Processing Functions** → refine localization results, suppress artifacts, and enhance reconstruction quality.  
- **3D Localization** → achieve sub-pixel emitter positioning in 3D, surpassing classical resolution limits.  
- **Rendering Tools** → generate and visualize high-resolution 3D reconstructions, including live-cell time series.  

🎥 Example: 3D reconstruction and rendering of a live-cell dataset  
![3D Reconstruction](https://user-images.githubusercontent.com/61014265/219693582-acd024b2-b547-496d-9136-95d91459288e.mp4)  

---

## 🚀 Performance
- Robust across 2D, 3D, and live-cell microscopy modalities.  
- Achieves **sub-10 nm localization accuracy** under challenging noise/light conditions.  
- Processes live-cell SMLM data in **~3 seconds**, enabling dynamic biological imaging.  

---

## 📦 Installation
1. Clone or download this repository.  
2. Create the environment (Linux example):  
   ```bash
   conda env create -f environment_linux.yml
   conda activate LUENN
3. Run training, inference, and reconstruction pipelines.

---

## Performance
Luenn has demonstrated exceptional accuracy across a broad spectrum of imaging conditions. Its ability to handle live-cell SMLM data with reduced light exposure in just 3 seconds makes it a valuable asset for dynamic imaging scenarios.

---

### System Requirements
- Linux (Ubuntu 20.04+) or Windows 10
- Python 3.8+
- NVIDIA GPU with CUDA support (tested on Titan Xp, 12 GB VRAM)
- 32 GB RAM recommended

## Contributers:

__Armin Abdehkakha__, _Email: arminabd@buffalo.edu_<br>
__Craig Snoeyink__, _Email: craigsno@buffalo.edu_
