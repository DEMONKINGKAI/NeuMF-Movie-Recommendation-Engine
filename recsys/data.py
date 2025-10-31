import os
import numpy as np
import pandas as pd
from collections import defaultdict

ML_100K_GENRES = [
    'Unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]


def load_movielens_100k(data_dir, min_rating=4):
    """
    Load MovieLens 100K ratings and item genre info.
    Returns:
        ratings_df: userId, itemId, rating, timestamp (original IDs)
        item_genres: dict {itemId (orig): np.array shape (19,) multi-hot}
    """
    udata_path = os.path.join(data_dir, 'ml-100k', 'u.data')
    uitem_path = os.path.join(data_dir, 'ml-100k', 'u.item')
    ratings_df = pd.read_csv(udata_path, sep='\t', names=['userId', 'itemId', 'rating', 'timestamp'])
    ratings_df = ratings_df[ratings_df['rating'] >= min_rating]
    # Genres
    genres = []
    with open(uitem_path, encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('|')
            iid = int(parts[0])
            gvec = np.array(list(map(int, parts[-19:])), dtype=np.float32)
            genres.append((iid, gvec))
    item_genres = {iid: gvec for (iid, gvec) in genres}
    return ratings_df, item_genres


def build_id_maps(ratings_df):
    """Assign int IDs to users/items. Return mappings and counts."""
    uid_map = {uid: ix for ix, uid in enumerate(sorted(ratings_df['userId'].unique()))}
    iid_map = {iid: ix for ix, iid in enumerate(sorted(ratings_df['itemId'].unique()))}
    return uid_map, iid_map, len(uid_map), len(iid_map)


def build_interactions(ratings_df, uid_map, iid_map):
    """
    Convert DF (after min_rating filter) to (user, item) interaction list using mapped IDs.
    Returns: numpy array shape (n_inter, 2)
    """
    users = ratings_df['userId'].map(uid_map)
    items = ratings_df['itemId'].map(iid_map)
    return np.stack([users.values, items.values], axis=1)


def create_genre_matrix(item_genres, iid_map):
    """
    From {itemId: onehot_vec}, build matrix shape (num_items, 19)
    """
    mat = np.zeros((len(iid_map), 19), dtype=np.float32)
    for orig_iid, idx in iid_map.items():
        mat[idx] = item_genres.get(orig_iid, np.zeros(19))
    return mat


def leave_one_out_split(interactions):
    """
    Splits user interactions into train/val/test using leave-one-out.
    Returns dict {'train': ..., 'val': ..., 'test': ...}, each an array of (u, i)
    """
    by_user = defaultdict(list)
    for u, i in interactions:
        by_user[u].append(i)
    train, val, test = [], [], []
    for u, items in by_user.items():
        if len(items) < 3:
            train += [(u, i) for i in items]
            continue
        items = sorted(items, key=lambda x: x)  # for reproducibility
        train += [(u, i) for i in items[:-2]]
        val.append((u, items[-2]))
        test.append((u, items[-1]))
    return {
        'train': np.array(train),
        'val': np.array(val),
        'test': np.array(test)
    }


def negative_sampling(pos_inter, num_users, num_items, num_neg_per=4, exclude_set=None, seed=None):
    """
    For each positive, sample num_neg_per negatives (user, item) not in exclude_set
    Returns: pos_inter, neg_inter (matched length but repeated per pos)
    """
    rng = np.random.RandomState(seed)
    exclude = set(tuple(x) for x in (exclude_set if exclude_set is not None else pos_inter))
    pos_users, pos_items = pos_inter[:, 0], pos_inter[:, 1]
    neg_users = np.repeat(pos_users, num_neg_per)
    neg_items = []
    for u in pos_users:
        user_negs = set()
        while len(user_negs) < num_neg_per:
            neg = rng.randint(0, num_items)
            tup = (u, neg)
            if tup not in exclude:
                user_negs.add(neg)
        neg_items.extend(list(user_negs))
    neg_items = np.array(neg_items)
    return np.stack([neg_users, neg_items], axis=1)


def build_dataset(data_dir, min_rating=4, neg_per_pos=4, seed=42):
    """
    Loads MovieLens, builds mappings, genre matrix, splits, negative samples for train.
    Returns: dict with train/val/test, mappings, genre matrix
    """
    ratings_df, item_genres = load_movielens_100k(data_dir, min_rating)
    uid_map, iid_map, num_users, num_items = build_id_maps(ratings_df)
    inter = build_interactions(ratings_df, uid_map, iid_map)
    genre_mat = create_genre_matrix(item_genres, iid_map)
    splits = leave_one_out_split(inter)
    # Train: sample negatives
    train_pos = splits['train']
    train_neg = negative_sampling(train_pos, num_users, num_items, neg_per_pos, exclude_set=inter, seed=seed)
    data = {
        'train_pos': train_pos,
        'train_neg': train_neg,
        'val': splits['val'],
        'test': splits['test'],
        'uid_map': uid_map,
        'iid_map': iid_map,
        'num_users': num_users,
        'num_items': num_items,
        'genre_mat': genre_mat
    }
    return data
