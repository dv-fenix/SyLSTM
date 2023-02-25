from sklearn.metrics import f1_score
import numpy as np


def f1(preds, labels):
    """
    Wrapper for weighted f1 measure
    """
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    weighted = f1_score(labels_flat, preds_flat, average="weighted")

    return weighted


def acpc(preds, labels):
    """
    Accuracy per class
    """
    inv_labels_dict = labels_dict  # {val:key for key,val in labels_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for idx in np.unique(labels_flat):
        y_hat = preds_flat[labels_flat == idx]
        y = labels_flat[labels_flat == idx]
        print(f"Class: {inv_labels_dict[idx]}")
        print(f"Accuracy: {len(y_hat[y_hat == idx]) / len(y)}\n")
