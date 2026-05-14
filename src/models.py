import torch
import torch.nn as nn
import torchvision.models as models
from xgboost import XGBClassifier
import joblib
import os
from src.config import XGB_PARAMS

class SkinCancerHybridModel(nn.Module):
    def __init__(self):
        super(SkinCancerHybridModel, self).__init__()
        # 1. Load Pre-trained ResNet50 backbone
        # We use the latest Weights API for PyTorch 2026 standards
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # 2. Extract Features: Remove the final Fully Connected (FC) layer[cite: 3]
        # ResNet50's internal layers output a 2048-d vector after Global Average Pooling
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # 3. Freeze the CNN: We only want to use it for extraction, not training
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # 4. Initialize the XGBoost Classifier[cite: 3, 5]
        self.classifier = XGBClassifier(**XGB_PARAMS)

    def forward(self, x):
        """Passes image through CNN to get features[cite: 3]."""
        with torch.no_grad():
            features = self.feature_extractor(x)
            # Flatten to (Batch_Size, 2048)
            return torch.flatten(features, 1)

    def train_xgboost(self, X_features, y_labels):
        """Trains the XGBoost head on the extracted features[cite: 3, 5]."""
        print("Training XGBoost Classifier...")
        self.classifier.fit(X_features, y_labels)
        print("XGBoost training complete.")

    def save_hybrid_model(self, model_dir="models"):
        """Saves both model components for production deployment."""
        os.makedirs(model_dir, exist_ok=True)
        # Save the CNN backbone weights
        torch.save(self.resnet.state_dict(), f"{model_dir}/resnet_backbone.pth")
        # Save the XGBoost model[cite: 5]
        self.classifier.save_model(f"{model_dir}/xgb_head.json")
        print(f"Artifacts saved in {model_dir}/")