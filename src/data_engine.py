import os
import torch
import kagglehub
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.config import IMG_SIZE, BATCH_SIZE

def download_and_get_paths():
    """Downloads dataset and returns local paths[cite: 4, 5]."""
    print("Connecting to Kaggle Hub...")
    path = kagglehub.dataset_download("bhaveshmittal/melanoma-cancer-dataset")
    train_dir = os.path.join(path, "train")
    test_dir = os.path.join(path, "test")
    return train_dir, test_dir

def get_dataloaders(train_dir, test_dir):
    """Creates PyTorch DataLoaders with professional transforms."""
    
    # 1. Image Transformations
    # We use standard ResNet normalization values
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. Load Datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    # 3. Create Loaders
    # Shuffle is True for training to improve model generalization[cite: 5]
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Data Pipeline Ready: {len(train_dataset)} training images, {len(test_dataset)} test images.")
    print(f"Detected Classes: {train_dataset.classes}") # Should be ['Benign', 'Malignant'][cite: 3]
    
    return train_loader, test_loader, train_dataset.classes