# import os
# import torch
# from torch import nn
# from torchvision import transforms, models
# from PIL import Image
#
# # Define the image transformations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
#
# # Load the pretrained model
# model = models.resnet18()
# num_classes = 2  # Number of classes (0 and 5)
# model.fc = nn.Linear(model.fc.in_features, num_classes)
# model.load_state_dict(torch.load('persian_number_model.pth'))
# model.eval()
#
# # Move the model to the GPU if available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)
#
# # Define the class names
# class_names = ['0', '5']
#
# # Load and predict the labels for the new images
# test_images_dir = 'test_image'  # Corrected the folder name
# test_images = os.listdir(test_images_dir)
#
# with torch.no_grad():
#     for test_image in test_images:
#         image_path = os.path.join(test_images_dir, test_image)
#         image = Image.open(image_path)
#         image = transform(image).unsqueeze(0).to(device)
#         output = model(image)
#         _, predicted = torch.max(output, 1)
#         predicted_class = class_names[predicted.item()]
#         print(f'Predicted label for {test_image}: {predicted_class}')




import os
import torch
from torchvision import transforms, models
from PIL import Image

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the pretrained model
model_path = 'persian_number_model3.6.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the entire model
model = torch.load(model_path)
model = model.to(device)
model.eval()

# Define the class names
class_names = ['0', '1','2','3','4','5','6','7','8','9']

# Load and predict the labels for the new images
test_images_dir = 'test_image'
test_images = os.listdir(test_images_dir)

with torch.no_grad():
    for test_image in test_images:
        image_path = os.path.join(test_images_dir, test_image)
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0).to(device)
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]
        print(f'Predicted label for {test_image}: {predicted_class}')
