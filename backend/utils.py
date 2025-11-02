import os
import numpy as np
import pandas as pd

ML_100K_GENRES = [
    'Unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

def load_titles_and_genres(uitem_path):
    """Returns {item_idx: {mlid, title, genres_onehot, genres}} for MovieLens 100K format."""
    df = pd.read_csv(uitem_path, sep='|', header=None, encoding='latin-1',
        names=["movie_id","title","release","v","url"]+ML_100K_GENRES)
    item_meta = {}
    for _, row in df.iterrows():
        mlid = int(row['movie_id'])
        title = row['title']
        genres_onehot = row[ML_100K_GENRES].values.astype(np.float32)
        genres = [g for g, v in zip(ML_100K_GENRES, genres_onehot) if v]
        item_meta[mlid] = {
            "movie_id": mlid, "title": title, "genres_onehot": genres_onehot, "genres": genres
        }
    return item_meta

def load_titles_and_genres_25m(movies_path):
    """Returns {item_idx: {mlid, title, genres}} for MovieLens 25M format."""
    df = pd.read_csv(movies_path)
    item_meta = {}
    for _, row in df.iterrows():
        mlid = int(row['movieId'])
        title = str(row['title'])
        # Genres are pipe-separated, e.g., "Action|Adventure|Sci-Fi" or "(no genres listed)"
        genre_str = str(row['genres']).strip()
        if genre_str == '(no genres listed)' or genre_str == 'nan' or pd.isna(genre_str):
            genres = []
        else:
            genres = [g.strip() for g in genre_str.split('|') if g.strip()]
        item_meta[mlid] = {
            "movie_id": mlid, "title": title, "genres": genres
        }
    return item_meta

def filter_items_by_genre(item_ids, meta, genre, strict=True):
    """item_ids: list of internal mapped idx (int); meta: {mlid: ...}; genre: str; strict: bool (True=must have)."""
    if strict:
        return [iid for iid in item_ids if genre in meta[next(iter(meta))]['genres'] and genre in meta[meta[next(iter(meta))]['movie_id']]['genres'] if genre in meta[meta[next(iter(meta))]['movie_id']]['genres']]
    else: # soft: rank by genre present but allow all
        return [iid for iid in item_ids if genre in meta[next(iter(meta))]['genres']] # filtering logic will be refined in main
