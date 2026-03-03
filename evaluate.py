import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def evaluate(input_folder, output_folder):
    psnr_values = []
    ssim_values = []

    input_images = sorted(os.listdir(input_folder))
    output_images = sorted(os.listdir(output_folder))

    if len(input_images) == 0 or len(output_images) == 0:
        print("Input or output folder is empty.")
        return

    for img_name in input_images:
        input_path = os.path.join(input_folder, img_name)
        output_path = os.path.join(output_folder, img_name)

        if not os.path.exists(output_path):
            continue

        input_img = cv2.imread(input_path)
        output_img = cv2.imread(output_path)

        if input_img is None or output_img is None:
            continue

        # Resize input if dimensions mismatch
        if input_img.shape != output_img.shape:
            input_img = cv2.resize(input_img, (output_img.shape[1], output_img.shape[0]))

        psnr = peak_signal_noise_ratio(input_img, output_img, data_range=255)
        ssim = structural_similarity(input_img, output_img, channel_axis=2, data_range=255)

        psnr_values.append(psnr)
        ssim_values.append(ssim)

    if len(psnr_values) == 0:
        print("No valid image pairs found.")
        return

    print("Evaluation Results")
    print("-------------------")
    print("Number of images:", len(psnr_values))
    print("Average PSNR:", round(np.mean(psnr_values), 4))
    print("Average SSIM:", round(np.mean(ssim_values), 4))


if __name__ == "__main__":
    input_folder = "sample_inputs"
    output_folder = "sample_outputs"
    evaluate(input_folder, output_folder)