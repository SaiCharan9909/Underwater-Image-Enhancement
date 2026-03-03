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
├── Diffusion_model/       
├── inputs/            
├── outputs/             
├── enhance.py             
├── evaluate.py            
├── README.md  
└── .gitignore  




