import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np


class UWEnhancer(nn.Module):
    """
    Wrapper model for underwater image enhancement.
    Combines diffusion restoration + custom color refinement.
    """

    def __init__(self, diffusion_model, device="cuda"):
        super(UWEnhancer, self).__init__()
        self.diffusion_model = diffusion_model
        self.device = device

    def forward(self, img_tensor):
        # Step 1: Diffusion enhancement
        with torch.no_grad():
            enhanced = self.diffusion_model(img_tensor)

        # Step 2: Convert to numpy for post-processing
        enhanced_img = enhanced.squeeze().permute(1, 2, 0).cpu().numpy()
        enhanced_img = np.clip(enhanced_img, 0, 1)

        # Step 3: Apply custom refinement
        refined = self.color_refinement(enhanced_img)

        # Step 4: Convert back to tensor
        refined_tensor = torch.from_numpy(refined).permute(2, 0, 1).unsqueeze(0)
        return refined_tensor.float().to(self.device)

    def color_refinement(self, image):
        """
        Custom color correction + contrast enhancement
        """
        image = (image * 255).astype(np.uint8)

        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        merged = cv2.merge((l, a, b))
        corrected = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

        corrected = corrected.astype(np.float32) / 255.0
        return corrected