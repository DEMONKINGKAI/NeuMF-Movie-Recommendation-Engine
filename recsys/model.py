import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuMF(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        num_genres,
        emb_dim_gmf=32,
        emb_dim_mlp=64,
        mlp_layers=(128, 64),
        genre_proj_dim=16,
        intent_dim: int | None = None,
        intent_hidden: int = 128,
        intent_dropout: float = 0.1,
    ):
        super().__init__()
        # GMF embeddings
        self.user_gmf = nn.Embedding(num_users, emb_dim_gmf)
        self.item_gmf = nn.Embedding(num_items, emb_dim_gmf)
        # MLP embeddings
        self.user_mlp = nn.Embedding(num_users, emb_dim_mlp)
        self.item_mlp = nn.Embedding(num_items, emb_dim_mlp)
        # Genre projector for multi-hot genre vectors
        self.genre_proj = nn.Linear(num_genres, genre_proj_dim)
        # Optional Intent tower
        self.intent_dim = intent_dim
        if intent_dim is not None and intent_dim > 0:
            self.intent_tower = nn.Sequential(
                nn.Linear(intent_dim, max(intent_hidden, 16)),
                nn.ReLU(),
                nn.Dropout(intent_dropout),
                nn.Linear(max(intent_hidden, 16), max(intent_hidden // 2, 16)),
                nn.ReLU(),
            )
            intent_out_dim = max(intent_hidden // 2, 16)
        else:
            self.intent_tower = None
            intent_out_dim = 0
        self.intent_out_dim = intent_out_dim
        # Build MLP tower (input = user_emb + item_emb + genre_emb + intent_emb)
        mlp_in_dim = emb_dim_mlp * 2 + genre_proj_dim + intent_out_dim
        mlp_modules = []
        for layer_size in mlp_layers:
            mlp_modules.append(nn.Linear(mlp_in_dim, layer_size))
            mlp_modules.append(nn.ReLU())
            mlp_in_dim = layer_size
        self.mlp = nn.Sequential(*mlp_modules)
        # Output layer: concat GMF and MLP outputs
        self.final = nn.Linear(emb_dim_gmf + mlp_layers[-1], 1)

    def forward(self, user_ids, item_ids, genres, intent_vec=None):
        """
        user_ids: (batch,) int64
        item_ids: (batch,) int64
        genres: (batch, num_genres) float32 multi-hot
        intent_vec: (batch, intent_dim) float32 or None
        """
        # GMF part
        gmf_u = self.user_gmf(user_ids)
        gmf_i = self.item_gmf(item_ids)
        gmf_out = gmf_u * gmf_i  # elementwise
        # MLP part
        mlp_u = self.user_mlp(user_ids)
        mlp_i = self.item_mlp(item_ids)
        genre_emb = self.genre_proj(genres)
        if self.intent_tower is not None:
            if intent_vec is not None:
                intent_latent = self.intent_tower(intent_vec)
            else:
                # zero intent latent to preserve expected MLP input width
                intent_latent = torch.zeros(mlp_u.size(0), self.intent_out_dim, device=mlp_u.device, dtype=mlp_u.dtype)
            mlp_input = torch.cat([mlp_u, mlp_i, genre_emb, intent_latent], dim=1)
        else:
            mlp_input = torch.cat([mlp_u, mlp_i, genre_emb], dim=1)
        mlp_out = self.mlp(mlp_input)
        # Concatenate final feature
        final_in = torch.cat([gmf_out, mlp_out], dim=1)
        logits = self.final(final_in).squeeze(-1)
        return logits

    def predict(self, user_ids, item_ids, genres, intent_vec=None):
        """Sigmoid output for predicted interaction probability."""
        return torch.sigmoid(self.forward(user_ids, item_ids, genres, intent_vec))
