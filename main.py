# CLI entry for training/evaluating recommender system.

import argparse
import yaml
import torch
import os
from recsys.train import train_model
from recsys.eval import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Train & Evaluate NeuMF Genre Recommender")
    parser.add_argument('--data', type=str, default='./data/ml-25m', help='Data directory (MovieLens 100K or 25M - will auto-detect format)')
    parser.add_argument('--config', type=str, default='./configs/starter.yaml', help='Config YAML file')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs in config')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device for torch (cpu or cuda)')
    parser.add_argument('--max-ratings', type=int, default=None, help='Optional limit on number of ratings to sample for faster experiments')
    args = parser.parse_args()
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    # Train
    print(f"Training NeuMF... (device: {args.device})")
    model, data = train_model(
        args.data,
        emb_dim_gmf=cfg['emb_dim_gmf'],
        emb_dim_mlp=cfg['emb_dim_mlp'],
        mlp_layers=tuple(cfg['mlp_layers']),
        genre_proj_dim=cfg.get('genre_proj_dim', 16),
        lr=cfg['lr'],
        batch_size=cfg['batch_size'],
        epochs=cfg['epochs'],
        neg_per_pos=cfg['neg_per_pos'],
        device=args.device,
        max_ratings=args.max_ratings,
    )
    # Eval
    print("Evaluating on test set...")
    hr, ndcg = evaluate_model(model, data, K=cfg['K'], device=args.device)
    # Save model
    os.makedirs('./checkpoints', exist_ok=True)
    model_path = f"./checkpoints/neumf_final.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}\nFinal HR@{cfg['K']}: {hr:.4f}  NDCG@{cfg['K']}: {ndcg:.4f}")

if __name__ == '__main__':
    main()
