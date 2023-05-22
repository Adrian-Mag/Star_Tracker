import cv2
import numpy as np
import torch

def detect_stars(image_path, model_path):
    # Load the StarNet++ model
    model = torch.load(model_path)

    # Load the original image
    image = cv2.imread(image_path)

    # Preprocess the image for StarNet++
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = input_image.astype(np.float32) / 255.0
    input_image = np.transpose(input_image, (2, 0, 1))
    input_image = np.expand_dims(input_image, axis=0)
    input_tensor = torch.from_numpy(input_image)

    # Run StarNet++ inference to obtain the star mask
    with torch.no_grad():
        output = torch.onnx._run_symbolic_function(model, ["input"], {}, torch.from_numpy(input_image))
    star_mask = output[0].numpy()[0, 0]

    # Subtract the starless image from the original to get an image with only stars
    star_image = image.astype(np.float32) - np.stack([star_mask] * 3, axis=-1)
    star_image = np.clip(star_image, 0, 255).astype(np.uint8)

    # Convert the star image to grayscale
    gray_image = cv2.cvtColor(star_image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding algorithm to find stars
    _, threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find contours of the stars
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    result = image.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    # Save the resulting image
    cv2.imwrite("star_detection_result.jpg", result)

# Example usage
image_path = "path/to/your/image.jpg"
model_path = "path/to/your/starnet++-lite.onnx"
detect_stars(image_path, model_path)
