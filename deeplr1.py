# import torch
# print(torch.cuda.is_available())
#
# from detecto import core, utils, visualize
# from detecto.visualize import show_labeled_image, plot_prediction_grid
# from torchvision import transforms
# import matplotlib.pyplot as plt
# import numpy as np
#
#
#
# def test_image(path, model, threshold):
#   image = utils.read_image(path)
#   predictions = model.predict(image)
#   labels, boxes, scores = predictions
#
#
#   filtered_indices=np.where(scores>threshold)
#   filtered_scores=scores[filtered_indices]
#   filtered_boxes=boxes[filtered_indices]
#   num_list = filtered_indices[0].tolist()
#   filtered_labels = [labels[i] for i in num_list]
#   show_labeled_image(image, filtered_boxes, filtered_labels)
#
#
# # Augmenting the data
#
# custom_transforms = transforms.Compose([
# transforms.ToPILImage(),
# transforms.Resize(900),
# transforms.RandomHorizontalFlip(0.5),
# transforms.ColorJitter(saturation=0.2),
# transforms.ToTensor(),
# utils.normalize_transform(),
# ])
#
#
# # Training the model
#
# Train_dataset=core.Dataset('persian_digit2/' ,transform=custom_transforms)
# # Test_dataset = core.Dataset('test_image')
# loader=core.DataLoader(Train_dataset, batch_size=1)
# model2 = core.Model(['N'])
# losses2 = model2.fit(loader, Test_dataset, epochs=20, lr_step_size=5, learning_rate=0.001, verbose=True)
#
#
# model2.save('persian_number_model_weights_2.pth')
# model2 = core.Model.load('persian_number_model_weights_2.pth', ['N'])
# test_image('licence_plates_/extras/index6.jpg', model2, 0.95)

from torch.utils.data import Dataset
import os
import torch
import numpy as np
from torchvision.io import read_image
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from torchvision import transforms

print(torch.cuda.is_available())

def test_image(path, model, threshold):
    image = utils.read_image(path)
    predictions = model.predict(image)
    labels, boxes, scores = predictions

    filtered_indices = np.where(scores > threshold)
    filtered_scores = scores[filtered_indices]
    filtered_boxes = boxes[filtered_indices]
    num_list = filtered_indices[0].tolist()
    filtered_labels = [labels[i] for i in num_list]
    show_labeled_image(image, filtered_boxes, filtered_labels)

# Augmenting the data
custom_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(900),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(saturation=0.2),
    transforms.ToTensor(),
    utils.normalize_transform(),
])

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.images = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            self.images += [(os.path.join(cls_dir, img), cls) for img in os.listdir(cls_dir)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = read_image(img_path)
        label = torch.tensor([self.classes.index(label)])  # Now label is a tensor
        return image, label

dataset = CustomDataset('persian_digit2')
train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.2)

# Create Subset objects using the indices
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# Create DataLoaders from the Subsets
train_loader = core.DataLoader(train_dataset, batch_size=1)
test_loader = core.DataLoader(test_dataset, batch_size=1)

# Training the model
model2 = core.Model(['N'])
losses2 = model2.fit(train_loader, test_loader, epochs=20, lr_step_size=5, learning_rate=0.001, verbose=True)

model2.save('persian_number_model_weights_2.pth')
model2 = core.Model.load('persian_number_model_weights_2.pth', ['N'])
test_image('test_image/0.jpg', model2, 0.95)

