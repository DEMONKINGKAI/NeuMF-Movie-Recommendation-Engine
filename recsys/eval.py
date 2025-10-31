import numpy as np
import torch
from tqdm import tqdm

def get_user_item_lists(test_set, num_users):
    """
    Build {user: [test_item]} dict for test set
    """
    user2item = {int(u): int(i) for u, i in test_set}
    return user2item

def hit_ratio(ranklist, gt_item):
    return int(gt_item in ranklist)

def ndcg(ranklist, gt_item):
    if gt_item in ranklist:
        idx = ranklist.index(gt_item)
        return np.log(2) / np.log(idx + 2)
    return 0.0

def evaluate_model(model, data, K=10, device="cuda" if torch.cuda.is_available() else "cpu", num_neg=99, seed=42):
    '''
    For each user in test, build (user, gt_item, 99 negs), predict, compute HR@K/NDCG@K
    '''
    model.eval()
    test = data['test']
    genre_mat = data['genre_mat']
    num_items = data['num_items']
    rng = np.random.RandomState(seed)
    hits, ndcgs = [], []
    for u, gt_i in tqdm(test, desc="Evaluating"):
        # Sample negatives
        negs = set()
        while len(negs) < num_neg:
            ni = rng.randint(0, num_items)
            if ni != gt_i:
                negs.add(ni)
        item_list = [gt_i] + list(negs)
        user_list = [u] * (1 + num_neg)
        genre_list = genre_mat[item_list]
        # Run model
        with torch.no_grad():
            user_t = torch.tensor(user_list).long().to(device)
            item_t = torch.tensor(item_list).long().to(device)
            genre_t = torch.tensor(genre_list).float().to(device)
            scores = model.predict(user_t, item_t, genre_t).cpu().numpy()
        # Top-K
        rank = np.argsort(-scores)
        topk_items = [item_list[r] for r in rank[:K]]
        hits.append(hit_ratio(topk_items, gt_i))
        ndcgs.append(ndcg(topk_items, gt_i))
    hr, ndcg_ = np.mean(hits), np.mean(ndcgs)
    print(f"Test HR@{K}: {hr:.4f}, NDCG@{K}: {ndcg_:.4f}")
    return hr, ndcg_
