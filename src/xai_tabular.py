import shap
import numpy as np
import matplotlib.pyplot as plt

class TabularExplainer:
    def __init__(self, xgb_model):
        """
        Initializes SHAP explainer for the XGBoost head.
        """
        self.model = xgb_model
        # TreeExplainer is the most accurate for XGBoost decisions
        self.explainer = shap.TreeExplainer(self.model)

    def plot_importance(self, features, ax):
        """
        Fills the specific subplot 'ax' with feature importance.
        """
        # Calculate SHAP values for the 2048 features
        shap_values = self.explainer.shap_values(features)
        
        # We must set 'show=False' so it doesn't create a new window
        # and manually target the third axis
        plt.sca(ax) 
        shap.summary_plot(
            shap_values, 
            features, 
            plot_type="bar", 
            max_display=10, 
            show=False, 
            plot_size=None
        )
        plt.title("Top 10 Decision Drivers (XGBoost)")
    
    def plot_force(self, features, base_value):
        """
        Generates the force plot seen in image_774eb2.png.
        """
        # Note: This usually requires a Jupyter-like display or saving as HTML
        plot = shap.force_plot(
            base_value, 
            self.explainer.shap_values(features), 
            features,
            matplotlib=True,
            show=False
        )
        plt.title("Force Plot: Feature Contribution")