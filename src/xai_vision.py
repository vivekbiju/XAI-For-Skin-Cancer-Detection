import torch
import torch.nn.functional as F
import numpy as np
import cv2
import shap

def generate_pixel_shap(hybrid_model, img_tensor, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def predict(x):
        # Convert numpy back to torch and send to same device as model
        tmp = torch.tensor(x).permute(0, 3, 1, 2).float().to(device)
        with torch.no_grad():
            # Extract features from ResNet backbone
            features = hybrid_model(tmp).cpu().numpy()
            # Get probabilities from XGBoost head
            return hybrid_model.classifier.predict_proba(features)

    # REFINEMENT: Use 'blur' masker for better hierarchical 'square' definitions
    masker = shap.maskers.Image("blur(128,128)", shape=(224, 224, 3))
    
    explainer = shap.Explainer(predict, masker, output_names=class_names)

    # REFINEMENT: Increased max_evals to 500 for clearer square boundaries
    print("Calculating SHAP values (Hierarchical Squares)...")
    shap_values = explainer(img_tensor.permute(0, 2, 3, 1).cpu().numpy(), 
                            max_evals=500, batch_size=50)
    return shap_values

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        # Enable grads for the explainability pass
        for param in self.model.parameters():
            param.requires_grad = True

        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output)

        self.model.zero_grad()
        output[0, class_idx].backward()

        if self.gradients is None:
            raise RuntimeError("Gradients not captured. Ensure target_layer is correct.")

        # REFINEMENT: Global Average Pooling of gradients (mimics Colab accuracy)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Multiply each channel in activations by its corresponding gradient weight
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        # REFINEMENT: Mean across channels for a more 'centered' spotting result
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        
        heatmap = F.relu(heatmap)
        heatmap /= (torch.max(heatmap) + 1e-10) 
        
        # Re-freeze params
        for param in self.model.parameters():
            param.requires_grad = False
            
        return heatmap.detach().cpu().numpy()

def overlay_heatmap(img_path, heatmap):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match original image dimensions
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 0.4 weight on heatmap ensures the lesion underneath is still visible
    superimposed_img = heatmap * 0.4 + img
    return superimposed_img / superimposed_img.max()