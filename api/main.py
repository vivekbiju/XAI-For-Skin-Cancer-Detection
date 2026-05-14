import os
import uuid
import torch
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

import uuid

# Modular Imports
from src.models import SkinCancerHybridModel
from src.xai_vision import GradCAM, overlay_heatmap, generate_pixel_shap
from src.reporting import generate_clinical_report

app = FastAPI(title="UK Clinical AI Diagnostic Gateway")

# Setup static hosting for reports and heatmaps
os.makedirs("api/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Global Model Loading (UK Clinical Standard: Load once on startup)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkinCancerHybridModel().to(device)
model.resnet.load_state_dict(torch.load("models/resnet_backbone.pth", map_location=device))
model.classifier.load_model("models/xgb_head.json")
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    # Create a unique ID for this clinical session
    session_id = str(uuid.uuid4())[:8]
    input_path = f"api/static/input_{session_id}.jpg"
    
    # 1. Save input
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # 2. Prediction
    img_raw = Image.open(input_path).convert('RGB')
    img_tensor = transform(img_raw).unsqueeze(0).to(device)
    features = model(img_tensor).cpu().numpy()
    probs = model.classifier.predict_proba(features)[0]
    pred_idx = int(np.argmax(probs))
    class_names = ['Benign', 'Malignant']

    # 3. Generate XAI components
    cam = GradCAM(model.resnet, model.resnet.layer4[-1])
    heatmap = cam.generate_heatmap(img_tensor, class_idx=pred_idx)
    grad_img = overlay_heatmap(input_path, heatmap)
    
    # Optional: Generate Pixel SHAP (Warning: Slows down API response)
    # shap_val = generate_pixel_shap(model, img_tensor, class_names)

    # 4. TRIGGER THE AUTOMATED REPORT (Advance #3)
    report_file = generate_clinical_report(
        original_img=img_raw,
        gradcam_img=grad_img,
        shap_img=grad_img, # Placeholder: Replace with shap_val if using SHAP
        diagnosis=class_names[pred_idx],
        confidence=f"{probs[pred_idx]:.2%}",
        report_id=session_id
    )

    return {
        "diagnosis": class_names[pred_idx],
        "confidence": f"{probs[pred_idx]:.2%}",
        "report_url": f"/static/{report_file}",
        "session_id": session_id
    }