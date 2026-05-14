import torch
import numpy as np
from tqdm import tqdm
from src.config import set_seed
from src.data_engine import download_and_get_paths, get_dataloaders
from src.models import SkinCancerHybridModel

def run_pipeline():
    # 1. Initialization
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # 2. Data Preparation
    train_dir, test_dir = download_and_get_paths()
    train_loader, test_loader, classes = get_dataloaders(train_dir, test_dir)

    # 3. Model Initialization
    hybrid_system = SkinCancerHybridModel().to(device)

    # 4. Feature Extraction
    # We pass the training data through ResNet50 to get input for XGBoost[cite: 3]
    all_features = []
    all_labels = []

    print("Extracting features for XGBoost training...")
    with torch.no_grad():
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            # Forward pass through the ResNet50 backbone[cite: 3]
            features = hybrid_system(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    # Convert list of batches to a single NumPy array
    X_train = np.vstack(all_features)
    y_train = np.concatenate(all_labels)

    # 5. XGBoost Training
    # Now we train the "Head" of our hybrid architecture[cite: 3, 5]
    hybrid_system.train_xgboost(X_train, y_train)

    # 6. Save Artifacts
    # This generates the .pth and .json files for your portfolio
    hybrid_system.save_hybrid_model()
    print("\nPipeline Complete! Check the 'models/' folder for your saved files.")

if __name__ == "__main__":
    run_pipeline()