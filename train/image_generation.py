import os
import cv2
import numpy as np

input_image = cv2.imread(r"D:\Door_Detection\train\full\image12.png")

if input_image is None:
    print("Image could not be loaded!")
    exit()

output_folder = "augmented_images"
os.makedirs(output_folder, exist_ok=True)

# Count existing images
existing_files = [
    f for f in os.listdir(output_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

start_index = len(existing_files) + 1

for i in range(28):
    img = input_image.copy()

    # Random rotation
    angle = np.random.randint(-30, 30)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))

    # Random brightness
    brightness = np.random.uniform(0.7, 1.3)
    img = cv2.convertScaleAbs(img, alpha=brightness, beta=0)

    # Random horizontal flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)

    image_number = start_index + i
    cv2.imwrite(
        os.path.join(output_folder, f"image_{image_number}.jpg"),
        img
    )

print(f"40 images added successfully. Starting from image_{start_index}.jpg")