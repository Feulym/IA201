import torchvision
import torchvision.transforms as transforms
import torch
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn as nn
import torch.optim as optim

#define GPU as device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Image transforms for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset using ImageFolder
train_data_dir = "./archive/train"
test_data_dir = "./archive/test"
val_data_dir = "./archive/valid"

train_dataset = torchvision.datasets.ImageFolder(root=train_data_dir, transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=test_data_dir, transform=transform)
val_dataset = torchvision.datasets.ImageFolder(root=val_data_dir, transform=transform)

NUM_CLASSES = 38

def get_model(pretrained=True, freeze_backbone=False):
    model = resnet18(pretrained=pretrained)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    if freeze_backbone:
        for param in model.fc.parameters():
            param.requires_grad = True
    return model.to(device)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,num_workers=4)

for batch_idx, (data, target) in enumerate(train_loader):
    if batch_idx == 0:
        break

NUM_IMAGES_TO_DISPLAY = 5

fig, axes = plt.subplots(1, NUM_IMAGES_TO_DISPLAY, figsize=(15, 5))

for i in range(NUM_IMAGES_TO_DISPLAY):
  image = data[i]
  image = image.permute(1, 2, 0) # Reshape for matplotlib
  image = image.numpy()
  # Normalize image to range 0-1
  image = (image - image.min()) / (image.max() - image.min())
  axes[i].imshow(image)
  axes[i].set_title(f"Label: {target[i]}")
  axes[i].axis("off")

plt.show()

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False,num_workers=4)

model = get_model(freeze_backbone=False)

LEARNING_RATE = 0.001
NUM_EPOCHS = 5

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

total = 0
correct = 0
# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()  # Set model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
          print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # Validation loop (optional)
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # Calculate validation metrics (e.g., accuracy) here
            # Compute Accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Validation Accuracy : {accuracy:.3f}")


def validate_and_plot_confusion_matrix(model, val_loader, device):
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predictions = torch.max(output, 1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal')
    plt.show()