import torch

def compute_pos_weight(train_df, diseases):
    pos_counts = train_df[diseases].sum()
    neg_counts = len(train_df) - pos_counts
    # Avoid div-by-zero if a class had 0 positives (shouldn't happen here, but safe)
    pos_weight = (neg_counts / pos_counts.clip(lower=1)).values
    return torch.tensor(pos_weight, dtype=torch.float32)