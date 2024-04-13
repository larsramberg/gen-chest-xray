import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_mean_roc_auc(labels, pred, num_classes):
    results = []
    labels = labels.cpu()
    pred = pred.cpu()
    
    for i in range(num_classes):
        score = roc_auc_score(labels[:,i].detach().numpy(), pred[:,i].detach().numpy())
        results.append(score)

    return np.array(results).mean()