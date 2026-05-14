import torch
import matplotlib.pyplot as plt
from src.models import SkinCancerHybridModel
from src.xai_vision import GradCAM, overlay_heatmap, generate_pixel_shap
from src.xai_tabular import TabularExplainer
from PIL import Image
from torchvision import transforms
import numpy as np
import shap

def run_research_explanation(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hybrid_system = SkinCancerHybridModel().to(device)
    hybrid_system.resnet.load_state_dict(torch.load("models/resnet_backbone.pth", map_location=device))
    hybrid_system.classifier.load_model("models/xgb_head.json")
    class_names = ['Benign', 'Malignant']

    # 1. Image Prep
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_raw = Image.open(image_path).convert('RGB')
    img_tensor = transform(img_raw).unsqueeze(0).to(device)

    # 2. Get Predictions
    features = hybrid_system(img_tensor).cpu().numpy()
    pred_idx = hybrid_system.classifier.predict(features)[0]

    # --- IMAGE 1: GRAD-CAM (Spotting) ---
    target_layer = hybrid_system.resnet.layer4[-1]
    cam = GradCAM(hybrid_system.resnet, target_layer)
    mask = cam.generate_heatmap(img_tensor, class_idx=pred_idx)
    grad_img = overlay_heatmap(image_path, mask)
    
    plt.figure("Vision: Grad-CAM", figsize=(8, 6))
    plt.imshow(grad_img)
    plt.axis('off')
    plt.title(f"Grad-CAM focus for {class_names[pred_idx]}")
    plt.show(block=False)

    # --- IMAGE 2: PIXEL SHAP (Red/Blue Heatmap - image_774ef7.png) ---
    print("Calculating Pixel SHAP... please wait (~60 seconds)")
    # This calls the function you provided
    # Change this line in explain.py:
    pixel_values = generate_pixel_shap(hybrid_system, img_tensor, class_names)
    
    # We must use shap.image_plot to render the red/blue pixels
    shap.image_plot(pixel_values, img_tensor.permute(0, 2, 3, 1).cpu().numpy())

    # --- IMAGE 3: FORCE PLOT (Red/Blue Arrows - image_774eb2.png) ---
    plt.figure("Logic: Force Plot", figsize=(12, 3))
    explainer = TabularExplainer(hybrid_system.classifier)
    
    # You need the 'base_value' from the explainer for this plot
    base_val = explainer.explainer.expected_value
    explainer.plot_force(features, base_value=base_val)
    plt.show()

if __name__ == "__main__":
    run_research_explanation("data/test/Malignant/melanoma.jpg")