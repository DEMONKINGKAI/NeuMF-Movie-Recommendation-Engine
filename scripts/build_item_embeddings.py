import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from recsys.data import build_dataset


def load_titles(data_dir: str) -> dict:
    """
    Returns {movie_id: title} from MovieLens 100K u.item.
    """
    uitem_path = os.path.join(data_dir, 'ml-100k', 'u.item')
    df = pd.read_csv(
        uitem_path,
        sep='|',
        header=None,
        encoding='latin-1',
        names=[
            'movie_id','title','release','video_release','imdb_url',
            'Unknown','Action','Adventure','Animation',"Children's",'Comedy','Crime',
            'Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery',
            'Romance','Sci-Fi','Thriller','War','Western'
        ]
    )
    return {int(r['movie_id']): str(r['title']) for _, r in df.iterrows()}


def build_texts_for_items(iid_map: dict, titles: dict, data_dir: str) -> list:
    """
    Returns list of texts aligned to internal item index (0..num_items-1).
    Enrich each text with genres to improve semantic recall for intent prompts.
    """
    import pandas as pd
    uitem_path = os.path.join(data_dir, 'ml-100k', 'u.item')
    df = pd.read_csv(
        uitem_path,
        sep='|',
        header=None,
        encoding='latin-1',
        names=[
            'movie_id','title','release','video_release','imdb_url',
            'Unknown','Action','Adventure','Animation',"Children's",'Comedy','Crime',
            'Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery',
            'Romance','Sci-Fi','Thriller','War','Western'
        ]
    )
    genre_cols = ['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary',
                  'Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance',
                  'Sci-Fi','Thriller','War','Western']
    by_id = {int(r['movie_id']): r for _, r in df.iterrows()}

    num_items = len(iid_map)
    texts = [''] * num_items
    for orig_iid, internal_idx in iid_map.items():
        title = titles.get(orig_iid, '')
        row = by_id.get(orig_iid)
        if row is not None:
            present = [g for g in genre_cols if int(row[g]) == 1]
            texts[internal_idx] = f"{title}. Genres: {', '.join(present)}."
        else:
            texts[internal_idx] = title
    return texts


def main():
    parser = argparse.ArgumentParser(description='Build item embeddings for intent-based retrieval')
    parser.add_argument('--data', type=str, default='./data/ml-100k', help='Data directory (contains ml-100k)')
    parser.add_argument('--model', type=str, default=os.environ.get('EMB_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2'))
    parser.add_argument('--out', type=str, default='./checkpoints/item_embeddings.npy')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    print('Loading dataset...')
    data = build_dataset(args.data)
    iid_map = data['iid_map']
    print(f"Found {len(iid_map)} items")

    print('Loading titles...')
    titles = load_titles(args.data)
    texts = build_texts_for_items(iid_map, titles, args.data)

    print(f"Loading embedding model: {args.model}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    print('Encoding items...')
    embeddings = []
    for i in tqdm(range(0, len(texts), args.batch_size)):
        batch = texts[i:i+args.batch_size]
        emb = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(emb)
    item_emb = np.vstack(embeddings).astype(np.float32)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, item_emb)
    print(f"Saved embeddings: {args.out} shape={item_emb.shape}")


if __name__ == '__main__':
    main()


