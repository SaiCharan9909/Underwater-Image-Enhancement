import os
import cv2
import numpy as np


def color_balance(image):
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(result)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def gamma_correction(image, gamma=1.2):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def enhance_image(image):
    balanced = color_balance(image)
    enhanced = gamma_correction(balanced, gamma=1.1)
    return enhanced


def main():
    input_folder = "inputs"
    output_folder = "outputs"

    os.makedirs(output_folder, exist_ok=True)

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)

        image = cv2.imread(img_path)

        if image is None:
            continue

        enhanced = enhance_image(image)

        save_path = os.path.join(output_folder, img_name)
        cv2.imwrite(save_path, enhanced)

        print(f"Processed: {img_name}")


if __name__ == "__main__":
    main()