import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from PIL import Image

# Enable cuDNN benchmarking for potential performance improvement
torch.backends.cudnn.benchmark = True

# Define the training parameters
batch_size = 88
epochs = 40
learning_rate = 0.0001

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the dataset
dataset = datasets.ImageFolder('persian_digit2', transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.80 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders for the training and validation sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Load a pre-trained model
model = models.resnet34(pretrained=True)

# Replace the last layer to match the number of classes
num_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move the model to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.5f}')




# Save the entire model
torch.save(model, 'persian_number_model3.5.pth')


print('Model trained and saved')

# Switch the model to evaluation mode
model.eval()

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
        print(f'Predicted label for {test_image}: {dataset.classes[predicted.item()]}')


#Epoch 12/12, Loss: 0.0031