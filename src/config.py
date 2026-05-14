import torch
import random
import numpy as np

# Image & Training Constants
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42

# XGBoost Parameters for the Hybrid Head[cite: 3]
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1
}

def set_seed():
    """Ensures reproducibility, vital for clinical AI[cite: 1, 5]."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)