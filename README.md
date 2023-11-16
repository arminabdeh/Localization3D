# Localization3D module (luenn): A PyTorch-Based Package for 3D Single Molecule Localization Microscopy

Luenn is a powerful Python package built on PyTorch, designed for 3D Single Molecule Localization Microscopy (SMLM). It offers a comprehensive set of functionalities for data generation, sampling, model training, post-processing, 3D localization, and rendering. Leveraging deep learning techniques, Luenn achieves high accuracy across various imaging modalities and conditions.

## Key Features

- **Data Generation:** Luenn provides tools for generating synthetic data, enabling users to simulate diverse imaging scenarios for training and evaluation.

- **Sampling:** The package includes sampling utilities to efficiently extract training and validation data from large datasets, optimizing the training process.

- **Model Training:** Luenn utilizes a DEep COntext DEpendent (DECODE) neural network to detect and localize emitters at sub-pixel resolution. Training is customizable, allowing users to adapt the model to specific experimental conditions.

- **Post-Processing Functions:** Luenn offers post-processing functions to enhance and refine localization results, ensuring superior super-resolution reconstructions.

- **3D Localization:** Luenn specializes in 3D localization, enabling precise positioning of emitters in three-dimensional space, a crucial aspect in single molecule localization microscopy.

- **Rendering:** The package facilitates the rendering of super-resolved images, providing visualization tools for the analyzed data.
example of 3D reconstruction and luenn rendering tool for a live cell time-series image set <br>

https://user-images.githubusercontent.com/61014265/219693582-acd024b2-b547-496d-9136-95d91459288e.mp4
## Performance

Luenn has demonstrated exceptional accuracy across a broad spectrum of imaging conditions. Its ability to handle live-cell SMLM data with reduced light exposure in just 3 seconds makes it a valuable asset for dynamic imaging scenarios.

## Getting Started
### System Requirements
The software was tested on a Linux system with Ubuntu version 7.0, and a Windows system with Windows 10 Home.
Training and evaluation were run on a standard workstation equipped with 32 GB of memory, an Intel(R) Core(TM) i7 − 8700, 3.20 GHz CPU, and a NVidia GeForce Titan Xp GPU with 12 GB of video memory.
 
### Installation
1. Download this repository as a zip file (or clone it using git). <br>
2. Go to the downloaded directory and unzip it. <br>
3. The conda environment for this project is given in environment_<os>.yml where <os> should be substituted with your operating system. For example, to replicate the environment on a linux system use the command: conda env create -f environment_linux.yml from within the downloaded directory. This should take a couple of minutes. <br>
4. After activation of the environment using: conda activate LUENN, you're set to go!

## Contributers:

__Armin Abdehkakha__, _Email: arminabd@buffalo.edu_<br>
__Craig Snoeyink__, _Email: craigsno@buffalo.edu_
