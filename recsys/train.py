import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from .model import NeuMF
from .data import build_dataset


class IntentInteractionDataset(Dataset):
    """
    For each positive interaction (u, ipos), create 1 positive sample and
    neg_per_pos negatives, all sharing intent_vec = item_emb[ipos].
    Assumes negatives are grouped per positive as produced by negative_sampling.
    Memory-efficient: stores item indices and looks up embeddings on-the-fly.
    """
    def __init__(self, train_pos, train_neg, genre_mat, item_emb, neg_per_pos: int):
        self.users = []
        self.items = []
        self.genres = []
        self.labels = []
        # Store index of the positive item for intent lookup (memory efficient)
        self.positive_item_indices = []  # For each sample, store the original positive item's index
        
        for p_idx, (u, ipos) in enumerate(train_pos):
            # positive sample
            self.users.append(u)
            self.items.append(ipos)
            self.genres.append(genre_mat[ipos])
            self.labels.append(1.0)
            self.positive_item_indices.append(ipos)  # Positive sample uses its own item's intent
            # negatives for this positive
            start = p_idx * neg_per_pos
            end = start + neg_per_pos
            for ineg in train_neg[start:end, 1]:
                self.users.append(u)
                self.items.append(ineg)
                self.genres.append(genre_mat[ineg])
                self.labels.append(0.0)
                self.positive_item_indices.append(ipos)  # Negative samples use the positive item's intent
        
        self.users = np.array(self.users, dtype=np.int64)
        self.items = np.array(self.items, dtype=np.int64)
        self.genres = np.stack(self.genres).astype(np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        self.positive_item_indices = np.array(self.positive_item_indices, dtype=np.int64)
        
        # Store item_emb for on-the-fly lookup (this is a reference, not a copy)
        self.item_emb = item_emb

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        # Look up intent vector from the original positive item
        intent_vec = self.item_emb[self.positive_item_indices[idx]]
        return (
            int(self.users[idx]),
            int(self.items[idx]),
            self.genres[idx],
            float(self.labels[idx]),
            intent_vec.copy(),  # Copy to avoid reference issues
        )


def make_train_loader_with_intent(data, item_emb, neg_per_pos, batch_size=256):
    """Create a DataLoader with intent vectors from item embeddings."""
    ds = IntentInteractionDataset(
        data['train_pos'], data['train_neg'], data['genre_mat'], item_emb, neg_per_pos
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)


def train_model(
    data_dir,
    emb_dim_gmf=32,
    emb_dim_mlp=64,
    mlp_layers=(128, 64),
    genre_proj_dim=16,
    use_intent: bool = True,
    intent_hidden: int = 128,
    lr=0.001,
    batch_size=256,
    epochs=10,
    neg_per_pos=4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_ratings=None,
):
    # Load data and build dataset
    data = build_dataset(data_dir, neg_per_pos=neg_per_pos, max_ratings=max_ratings)

    # Load item embeddings for intent tower (titles+genres)
    item_emb = None
    intent_dim = None
    emb_path = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
        'checkpoints',
        'item_embeddings.npy',
    )
    if use_intent and os.path.exists(emb_path):
        item_emb = np.load(emb_path).astype(np.float32)
        # Validate that embeddings match the dataset size
        if item_emb.shape[0] != data['num_items']:
            print(f"Warning: Item embeddings shape {item_emb.shape[0]} does not match dataset items {data['num_items']}.")
            print(f"Embeddings were likely built on a different dataset. Skipping intent vectors.")
            print(f"To use intent system, rebuild embeddings with: python scripts/build_item_embeddings.py --data {data_dir}")
            item_emb = None
            intent_dim = None
        else:
            intent_dim = int(item_emb.shape[1])
            train_loader = make_train_loader_with_intent(
                data, item_emb, neg_per_pos, batch_size
            )
    
    if item_emb is None:
        # Fallback dataset without intent vectors (zeros placeholder)
        class SimpleDataset(Dataset):
            def __init__(self, pairs, genre_mat, label):
                self.pairs = pairs
                self.genres = genre_mat[pairs[:, 1]]
                self.label = label

            def __len__(self):
                return len(self.pairs)

            def __getitem__(self, idx):
                u, i = self.pairs[idx]
                return u, i, self.genres[idx], float(self.label), np.zeros(1, dtype=np.float32)

        pos = SimpleDataset(data['train_pos'], data['genre_mat'], 1)
        neg = SimpleDataset(data['train_neg'], data['genre_mat'], 0)
        pairs = torch.utils.data.ConcatDataset([pos, neg])
        train_loader = DataLoader(pairs, batch_size=batch_size, shuffle=True, num_workers=0)

    # Model
    model = NeuMF(
        data['num_users'],
        data['num_items'],
        data['genre_mat'].shape[1],
        emb_dim_gmf,
        emb_dim_mlp,
        mlp_layers,
        genre_proj_dim,
        intent_dim=intent_dim,
        intent_hidden=intent_hidden,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        losses = []
        for batch in train_loader:
            u, i, g, y, intent_vec = batch
            u = u.to(device).long()
            i = i.to(device).long()
            g = g.to(device).float()
            y = y.to(device).float()
            intent_vec = intent_vec.to(device).float() if intent_dim is not None else None
            logits = model(u, i, g, intent_vec)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {sum(losses)/len(losses):.4f}")
    return model, data
