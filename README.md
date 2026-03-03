# Underwater Image Enhancement Based on Diffusion Probabilistic Model

## Overview

This project focuses on enhancing underwater images using a Diffusion Probabilistic Model.

Underwater images usually suffer from strong blue or green color tone, low contrast, poor visibility, and loss of details due to light absorption and scattering in water. The goal of this project is to improve the visual quality of such images and make them clearer and more natural.

The system is implemented as a clean and modular enhancement pipeline with result evaluation.

---

## How It Works

The model is based on a diffusion probabilistic approach.

In simple terms:
- During training, noise is gradually added to images.
- The model learns how to remove this noise step by step.
- During testing, it uses this learned reverse process to restore degraded underwater images.

After the main restoration step, additional color and contrast refinement is applied to improve the final output quality.

---

## Project Structure

Underwater-Image-Enhancement/
│
├── Diffusion_model/      # Core model files  
├── inputs/               # Input underwater images  
├── outputs/              # Enhanced results  
├── enhance.py            # Main enhancement script  
├── evaluate.py           # PSNR and SSIM evaluation  
├── README.md  
└── .gitignore  

---

## Installation

Install the required dependencies:

pip install -r Diffusion_model/requirements.txt

---

## How to Use

1. Add Input Images  

Place 3–5 underwater images inside the `inputs/` folder.

Example:

inputs/
- coral_01.jpg  
- fish_02.jpg  
- diver_03.jpg  

2. Run Enhancement  

Run the following command:

python enhance.py

The enhanced images will be saved inside the `outputs/` folder.

3. Run Evaluation  

To evaluate the results, run:

python evaluate.py

This will calculate:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)

These metrics help measure the improvement in image quality.

---

## Results

Example 1  
Input → inputs/coral_01.jpg  
Output → outputs/coral_01.jpg  

Example 2  
Input → inputs/fish_02.jpg  
Output → outputs/fish_02.jpg  

---

## Conclusion

This project demonstrates a structured implementation of underwater image enhancement using a diffusion probabilistic model. It focuses on clean design, modular implementation, and clear result evaluation.
