# Orthophoto Shadow Removal

This repository contains the implementation of a **U-Net–based deep learning model** for **shadow removal from orthophoto imagery**.  
The project addresses the lack of paired shadow–non-shadow training data by introducing a novel synthetic dataset and a complete pipeline for dataset generation, preprocessing, training, and evaluation.

## Overview

Shadows in aerial orthophotos can obscure ground details and reduce image usability in downstream tasks such as segmentation, object detection, and mapping.  
This project explores a data-driven approach to removing shadows using **convolutional neural networks (CNNs)**, specifically a **U-Net architecture** trained on synthetic shadow data generated in Unity.

## Main Components

- **U-Net shadow removal model** – implemented using PyTorch for pixel-wise image-to-image translation.  
- **Data preprocessing tools** – scripts to crop, tile, and organize orthophoto imagery for training and validation.  
- **Dataset generation utilities** – available in a companion repository [generate-scene](https://github.com/LGaljo/generate-scene), which creates realistic synthetic shadows on orthophotos using Unity’s HDRP (High Definition Render Pipeline).  
- **Training and evaluation pipeline** – supports multiple datasets and includes quantitative evaluation using metrics such as **RMSE**, **SSIM**, **PSNR**, and **LPIPS**.  

## Datasets

- **Unity Synthetic Orthophoto Shadow Dataset (USOS)**  
  A custom dataset of **300,000 synthetic shadowed and shadow-free image triplets** generated with Unity.  
  The dataset includes diverse terrain, vegetation, and illumination conditions and is publicly available on [Zenodo](https://zenodo.org/records/17009467).

- **Pretraining dataset**  
  Derived from Slovenian national orthophoto sources (ARSO *Atlas Okolja* and *eProstor*), consisting of 46,584 cropped tiles (256×256 px) used for model pretraining.

## Model Architecture

The implemented **U-Net** follows a symmetric encoder–decoder structure with skip connections for precise spatial reconstruction.  
It was trained and compared against the **ST-CGAN** model using several public shadow datasets (ISTD, SRD) and the custom USOS dataset.

## Results

- The model trained on **USOS** achieved competitive quantitative metrics compared to models trained on real-world datasets.
- The synthetic dataset proved effective for small and well-defined shadows, though realism in illumination simulation remains a key challenge.
- Evaluation confirmed the **importance of high-quality, domain-specific training data** for orthophoto applications.

## Citation

If you use this work, please cite:

> **Luka Galjot (2025).** *Odstranjevanje senc z ortofoto slik* (Master’s Thesis).  
> University of Ljubljana, Faculty of Computer and Information Science.  
> DOI: [10.5281/zenodo.17009467](https://zenodo.org/records/17009467)

## Related Projects

- **Dataset and scene generator** → [generate-scene](https://github.com/LGaljo/generate-scene)  
- **Synthetic dataset (Zenodo)** → [USOS dataset](https://zenodo.org/records/17009467)
