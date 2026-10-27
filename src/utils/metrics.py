from sklearn.metrics import roc_auc_score
import numpy as np

def compute_auroc(y_true, y_pred, disease_names):
    """
    y_true, y_pred: numpy arrays of shape [N, 14]
    Returns per-class AUROC dict + macro average.
    """
    per_class = {}
    for i, name in enumerate(disease_names):
        # Skip classes with no positive examples in this batch/split (AUROC undefined)
        if y_true[:, i].sum() == 0:
            continue
        per_class[name] = roc_auc_score(y_true[:, i], y_pred[:, i])

    macro_auroc = np.mean(list(per_class.values()))
    return per_class, macro_auroc