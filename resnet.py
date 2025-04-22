import torchvision
import torchvision.transforms as transforms
import torch
from torchvision.models import resnet18

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
train_data_dir = "./train"
val_data_dir = "./val"
train_dataset = torchvision.datasets.ImageFolder(root=train_data_dir, transform=transform)
val_dataset = torchvision.datasets.ImageFolder(root=val_data_dir, transform=transform)

NUM_CLASSES = 25

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