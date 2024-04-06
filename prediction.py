import numpy as np
from PIL import Image
import torch
import easyocr

# Load your custom PyTorch model (trained to detect 0 and 5)
# Replace with the actual path to your custom model
custom_model_path = 'persian_number_model0&5.21.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the entire model
custom_model = torch.load(custom_model_path, map_location=device)
custom_model = custom_model.to(device)

# Load the input image
image_path = "melli-pic/test-1.jpg"
image = Image.open(image_path)

# Initialize EasyOCR reader
reader = easyocr.Reader(['fa', 'ar'], gpu=True)  # You can adjust the languages as needed

# Read text from the image
result = reader.readtext(np.array(image))  # Convert image to a numpy array

def extract_image_patch(image, position):
    """
    Extracts the region of interest (ROI) around the detected text position.

    Args:
        image (PIL.Image.Image): The original input image.
        position (tuple): (x, y) coordinates of the detected text.

    Returns:
        PIL.Image.Image: The extracted image patch (ROI).
    """
    x, y = position
    patch_size = 50  # Adjust the patch size based on your requirements
    roi = image.crop((x, y, x + patch_size, y + patch_size))
    return roi

class_names = ['0', '5']
# Process each detected text
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Process each detected text
for detection in result:
    coordinates, detected_text, confidence_score = detection
    print(f"Detected: {detected_text}")
    corrected_text = ""
    for char in detected_text:
        if char in ["۰", "۵", "٥", "٠"]:
            # Extract the corresponding image patch (ROI) for prediction
            x, y = coordinates[0]  # Get the x, y coordinates of the detected text
            roi = extract_image_patch(image, (x, y))

            # Convert ROI to a PyTorch tensor
            roi_tensor = transform(roi).unsqueeze(0).to(device)

            # Predict using your custom model
            with torch.no_grad():
                output = custom_model(roi_tensor)
                _, predicted = torch.max(output, 1)
                corrected_label = class_names[predicted.item()]

            corrected_text += corrected_label
        elif char == " ":
            corrected_text += " "
        else:
            corrected_text += char

    print(f"Corrected: {corrected_text}")
