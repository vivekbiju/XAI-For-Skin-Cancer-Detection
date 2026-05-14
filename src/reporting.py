import matplotlib.pyplot as plt
import os

def generate_clinical_report(original_img, gradcam_img, shap_img, diagnosis, confidence, report_id):
    """
    Generates a triple-panel clinical audit report.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Original Image
    axes[0].imshow(original_img)
    axes[0].set_title("Clinical Input Image")
    axes[0].axis('off')
    
    # 2. Grad-CAM (Spatial Localization)
    axes[1].imshow(gradcam_img)
    axes[1].set_title("Grad-CAM: ROI Localization")
    axes[1].axis('off')
    
    # 3. SHAP (Feature Importance)
    # Note: If passing pixel-SHAP, ensure it's a renderable numpy array
    axes[2].imshow(shap_img)
    axes[2].set_title("SHAP: Diagnostic Pixel Influence")
    axes[2].axis('off')
    
    plt.suptitle(f"Diagnostic Audit - {diagnosis} (Confidence: {confidence})", fontsize=16)
    
    # Save to the static folder for API access
    report_filename = f"report_{report_id}.png"
    report_path = os.path.join("api/static", report_filename)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(report_path)
    plt.close()
    
    return report_filename