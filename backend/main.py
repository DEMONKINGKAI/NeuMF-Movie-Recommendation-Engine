import os
import sys
import torch
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from recsys.model import NeuMF
from recsys.data import build_dataset
from backend.utils import ML_100K_GENRES, load_titles_and_genres, load_titles_and_genres_25m
from fastapi.middleware.cors import CORSMiddleware
import logging
from difflib import get_close_matches
import yaml

 

# Resolve paths relative to the project root so running from backend works
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'ml-25m')
DATA_DIR = os.environ.get('MOVIELENS_PATH', DEFAULT_DATA_DIR)

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, 'checkpoints', 'neumf_final.pt')
MODEL_PATH = os.environ.get('MODEL_PATH', DEFAULT_MODEL_PATH)

# Config path for hyperparameters
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'configs', 'starter.yaml')
CONFIG_PATH = os.environ.get('CONFIG_PATH', DEFAULT_CONFIG_PATH)

# Embeddings
DEFAULT_EMB_PATH = os.path.join(PROJECT_ROOT, 'checkpoints', 'item_embeddings.npy')
EMB_PATH = os.environ.get('EMB_PATH', DEFAULT_EMB_PATH)
EMB_MODEL_NAME = os.environ.get('EMB_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2')

# Will detect dataset format after loading data
# UITEM_PATH will be set based on detected format
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOP_K_DEFAULT = 10
app = FastAPI()

# Logger
logger = logging.getLogger("intent_recsys")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)

# CORS for frontend (Vite default port and localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost",
        "http://127.0.0.1",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config for hyperparameters
try:
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded config from {CONFIG_PATH}")
except Exception as e:
    logger.warning(f"Failed to load config from {CONFIG_PATH}: {e}. Using defaults.")
    cfg = {}

# Load whole dataset, model, and movie meta at startup
logger.info("Loading dataset and model...")
data = build_dataset(DATA_DIR)
num_users, num_items, genre_mat = data['num_users'], data['num_items'], data['genre_mat']
item_invmap = {v:k for k, v in data['iid_map'].items()}
user_invmap = {v:k for k, v in data['uid_map'].items()}

# Detect dataset format and set paths accordingly
dataset_format = data['dataset_format']
if dataset_format == '100k':
    UITEM_PATH = os.path.join(DATA_DIR, 'ml-100k', 'u.item')
    meta = load_titles_and_genres(UITEM_PATH)
    ML_GENRES = ML_100K_GENRES
else:
    movies_path = os.path.join(DATA_DIR, 'ml-25m', 'movies.csv')
    if not os.path.exists(movies_path):
        movies_path = os.path.join(DATA_DIR, 'movies.csv')
    UITEM_PATH = movies_path
    meta = load_titles_and_genres_25m(UITEM_PATH)
    ML_GENRES = data['genre_list']  # Use genres from dataset

# Load embeddings to determine if intent tower should be used
item_emb_for_model = None
intent_dim = None
if os.path.exists(EMB_PATH):
    try:
        item_emb_for_model = np.load(EMB_PATH)
        if item_emb_for_model.shape[0] == num_items:
            intent_dim = int(item_emb_for_model.shape[1])
            logger.info(f"Intent dimension: {intent_dim}")
        else:
            logger.warning(f"Embeddings shape {item_emb_for_model.shape[0]} != num_items {num_items}; not using intent tower")
    except Exception as e:
        logger.warning(f"Failed to load embeddings for model: {e}")

# Initialize model with the same hyperparameters as training
model = NeuMF(
    num_users, 
    num_items, 
    genre_mat.shape[1],
    emb_dim_gmf=cfg.get('emb_dim_gmf', 32),
    emb_dim_mlp=cfg.get('emb_dim_mlp', 64),
    mlp_layers=tuple(cfg.get('mlp_layers', [128, 64])),
    genre_proj_dim=cfg.get('genre_proj_dim', 16),
    intent_dim=intent_dim,
    intent_hidden=128,
)
try:
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    # Filter incompatible keys (e.g., after adding intent tower or changing dims)
    current = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in current and v.shape == current[k].shape}
    missing = [k for k in current.keys() if k not in compatible]
    unexpected = [k for k in state.keys() if k not in current]
    if len(unexpected) > 0 or len(missing) > 0:
        logger.info(f"Partial checkpoint load. Missing: {len(missing)} Unexpected: {len(unexpected)}")
    model.load_state_dict(compatible, strict=False)
except Exception as e:
    logger.warning(f"Failed to fully load checkpoint: {e}. Proceeding with randomly initialized weights.")
model.eval()
model.to(DEVICE)

# Popularity based on interactions (train/val/test) normalized to [0,1]
item_pop = np.zeros(num_items, dtype=np.float32)
for arr_name in ['train_pos', 'val', 'test']:
    arr = data.get(arr_name)
    if arr is not None and len(arr) > 0:
        uniq, cnt = np.unique(arr[:, 1], return_counts=True)
        item_pop[uniq] += cnt.astype(np.float32)
if item_pop.max() > 0:
    item_pop = item_pop / item_pop.max()

# Use the embeddings loaded earlier for API purposes
item_emb = item_emb_for_model
if item_emb is None:
    logger.info(f"No embeddings available; semantic retrieval disabled")

# Build per-genre centroids for embedding-only intent mapping
genre_centroids = None
if 'item_emb' in globals() and item_emb is not None:
    try:
        # Collect internal item indices per genre
        per_genre = {g: [] for g in ML_GENRES}
        for internal_idx, mlid in item_invmap.items():
            if mlid in meta:
                for g in meta[mlid]['genres']:
                    if g in per_genre:
                        per_genre[g].append(internal_idx)
        cents = []
        for g in ML_GENRES:
            idxs = per_genre[g]
            if len(idxs) > 0:
                v = item_emb[idxs].mean(axis=0)
                n = np.linalg.norm(v) + 1e-8
                v = (v / n).astype(np.float32)
            else:
                v = np.zeros(item_emb.shape[1], dtype=np.float32)
            cents.append(v)
        genre_centroids = np.vstack(cents).astype(np.float32)
    except Exception as e:
        logger.warning(f"Failed to build genre centroids: {e}")

class RecResponse(BaseModel):
    movie_id: int
    title: str
    genres: List[str]
    score: float

@app.get("/genres")
def get_genres():
    return ML_GENRES

@app.get("/users")
def get_users():
    return [int(uid) for uid in user_invmap.values()]

@app.get("/recommendations", response_model=List[RecResponse])
def recommend(user_id: int = Query(...), genre: str = Query(...), top_k: int = Query(TOP_K_DEFAULT), strict: bool = Query(True)):
    """
    Returns top_k recs for this user_id, filtered to genre (strict/soft)
    """
    # Map user_id to internal
    uid = {v:k for k,v in user_invmap.items()}[user_id]
    # All (internal) items user has not seen
    all_items = set(range(num_items))
    user_interacted = set(np.concatenate([
        data['train_pos'][data['train_pos'][:,0] == uid][:,1],
        data['val'][data['val'][:,0] == uid][:,1] if len(data['val'])>0 else [],
        data['test'][data['test'][:,0] == uid][:,1] if len(data['test'])>0 else [],
    ]))
    candidates = np.array(list(all_items - user_interacted))
    # Filter candidate items by selected genre
    def has_genre(idx):
        mlid = item_invmap[idx]
        return genre in meta[mlid]['genres']
    if strict:
        candidates = np.array([idx for idx in candidates if has_genre(idx)])
        genre_bonus = None
    else:
        genre_bonus = np.array([1.0 if has_genre(idx) else 0.0 for idx in candidates])
    if len(candidates) == 0:
        return []
    user_tensor = torch.tensor([uid] * len(candidates)).long().to(DEVICE)
    item_tensor = torch.tensor(candidates).long().to(DEVICE)
    genre_tensor = torch.tensor(genre_mat[candidates]).float().to(DEVICE)
    with torch.no_grad():
        scores = model.predict(user_tensor, item_tensor, genre_tensor).cpu().numpy()
    if not strict:
        scores = scores + 0.20 * genre_bonus  # weighted preference for matching genre
    top_idx = np.argsort(-scores)[:top_k]
    result = []
    for i in top_idx:
        idx = candidates[i]
        mlid = item_invmap[idx]
        title = meta[mlid]['title']
        genres = meta[mlid]['genres']
        result.append(RecResponse(movie_id=mlid, title=title, genres=genres, score=float(scores[i])))
    return result

# ---------------- Embedding-based intent mapping and recommendations ----------------

# Basic synonyms/keywords toward canonical genres and popularity bias
def map_intent_to_weights_via_embedding(text: str):
    """
    Embedding-only intent mapping:
    - Compute cosine similarity between query embedding and per-genre centroids
    - Use positives-only normalized sims as genre weights
    - No popularity prior
    """
    if not text or item_emb is None or genre_centroids is None:
        return np.zeros(len(ML_GENRES), dtype=np.float32), 0.0
    qv = embed_text(text)
    if qv is None:
        return np.zeros(len(ML_GENRES), dtype=np.float32), 0.0
    sims = (genre_centroids @ qv).astype(np.float32)
    sims = np.clip(sims, 0.0, None)
    if sims.sum() > 0:
        sims = sims / sims.sum()
    return sims, 0.0

# Map each affect to genre priors (weights sum <= 1)
# Updated to work with any genre list

def steer_query_vector(qv: np.ndarray, top_k: int = 3, bottom_k: int = 2, pos_scale: float = 0.8, neg_scale: float = 1.0) -> np.ndarray:
    """
    Genre-agnostic steering: pull query toward its most similar genre centroids
    and push away from least similar ones. Works for any genre combination.
    """
    if qv is None or genre_centroids is None:
        return qv
    sims = genre_centroids @ qv
    top_idx = np.argsort(-sims)[:max(1, top_k)]
    bot_idx = np.argsort(sims)[:max(1, bottom_k)]
    pos = genre_centroids[top_idx].mean(axis=0) if len(top_idx) > 0 else 0.0
    neg = genre_centroids[bot_idx].mean(axis=0) if len(bot_idx) > 0 else 0.0
    steered = qv + pos_scale * pos - neg_scale * neg
    steered = steered / (np.linalg.norm(steered) + 1e-8)
    return steered.astype(np.float32)

# -------- Affect anchors (semantic, no keywords) --------
# Phrases are only used to get stable affect directions in embedding space
AFFECT_ANCHORS = {
    'sad': 'very sad heartbreaking tragic emotional tearjerker',
    'funny': 'very funny hilarious comedy laugh out loud',
    'scary': 'very scary terrifying horror chilling',
    'romantic': 'romantic heartfelt love story emotional',
    'exciting': 'exciting thrilling adrenaline fast paced action',
    'inspiring': 'inspiring uplifting motivational feel good',
    'family': 'family friendly suitable for children heartwarming',
    'dark': 'dark gritty noir intense',
}

# Map each affect to genre priors (weights sum <= 1)
AFFECT_GENRE_PRIOR = {
    'sad': {'Drama': 0.6, 'Romance': 0.3, 'Documentary': 0.1},
    'funny': {'Comedy': 0.9},
    'scary': {'Horror': 0.7, 'Thriller': 0.3},
    'romantic': {'Romance': 0.7, 'Drama': 0.3},
    'exciting': {'Action': 0.7, 'Thriller': 0.25, 'Adventure': 0.05},  # Boosted Action for exciting
    'inspiring': {'Drama': 0.6, 'Documentary': 0.4},
    'family': {"Children's": 0.6, 'Animation': 0.4},
    'dark': {'Thriller': 0.5, 'Crime': 0.3, 'Film-Noir': 0.2},
}

# Optional conflict suppression by affect
CONFLICT_SUPPRESS = {
    'sad': ['Horror'],
    'funny': ['Horror'],
    'scary': ['Romance', 'Comedy'],
    'romantic': ['Horror'],
    'exciting': ['Drama', 'War'],
    'inspiring': ['Horror'],
    'family': ['Horror', 'Film-Noir'],
    'dark': ['Children\'s'],
}

_affect_cache = {}

def get_affect_embeddings():
    if item_emb is None:
        return None
    try:
        global _affect_cache
        if _affect_cache:
            return _affect_cache
        for k, phrase in AFFECT_ANCHORS.items():
            vec = embed_text(phrase)
            if vec is not None:
                _affect_cache[k] = vec / (np.linalg.norm(vec) + 1e-8)
        return _affect_cache
    except Exception:
        return None

def embed_text(text: str):
    if item_emb is None:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        # Cache model per-process (simple singleton)
        global _emb_model
        if '_emb_model' not in globals():
            _emb_model = SentenceTransformer(EMB_MODEL_NAME)
        vec = _emb_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return vec[0]
    except Exception as e:
        print(f"[WARN] embed_text failed: {e}")
        return None

@app.get("/intent_recommendations", response_model=List[RecResponse])
def intent_recommend(
    q: str = Query(..., description="Free-form prompt (no keywords required)"),
    user_id: int = Query(...),
    top_k: int = Query(TOP_K_DEFAULT),
    strict: bool = Query(False),
    genre_alpha: float = Query(0.35),
    pop_alpha: float = Query(0.05),
    embed_alpha: float = Query(0.60),
    candidate_pool: int = Query(500),
):
    logger.info(f"/intent_recommendations q='{q}' user_id={user_id} strict={strict} alphas(g={genre_alpha},p={pop_alpha},e={embed_alpha})")
    uid = {v:k for k,v in user_invmap.items()}[user_id]
    all_items = set(range(num_items))
    user_interacted = set(np.concatenate([
        data['train_pos'][data['train_pos'][:,0] == uid][:,1],
        data['val'][data['val'][:,0] == uid][:,1] if len(data['val'])>0 else [],
        data['test'][data['test'][:,0] == uid][:,1] if len(data['test'])>0 else [],
    ]))
    remaining = np.array(list(all_items - user_interacted))
    if len(remaining) == 0:
        return []

    # Intent mapping (embedding-only) with embedding steering
    qv_raw = embed_text(q)
    if qv_raw is None:
        genre_weights = np.zeros(len(ML_GENRES), dtype=np.float32)
        pop_w = 0.0
        qv_use = None
    else:
        # Adjust steering based on dominant affect to better match user intent
        affect_embs_peek = get_affect_embeddings()
        pos_scale = 0.8
        neg_scale = 1.0
        if affect_embs_peek:
            aff_scores = {k: float(np.dot(v, qv_raw / (np.linalg.norm(qv_raw) + 1e-8))) for k, v in affect_embs_peek.items()}
            # Boost exciting for action-oriented queries
            action_keywords = ['pumping', 'pump', 'adrenaline', 'heart racing', 'action', 'thrilling', 'awesome']
            query_lower = q.lower()
            if any(kw in query_lower for kw in action_keywords) and 'exciting' in aff_scores:
                exciting_score = aff_scores.get('exciting', 0)
                scary_score = aff_scores.get('scary', 0)
                if exciting_score > 0 and scary_score > exciting_score * 0.7:
                    # Boost exciting for action queries
                    aff_scores['exciting'] = exciting_score * 1.3
            
            top_aff = max(aff_scores, key=aff_scores.get)
            if aff_scores[top_aff] > 0:
                # Stronger pull for high-energy affects, prioritize exciting
                if top_aff == 'exciting':
                    pos_scale = 1.1  # Strong pull for exciting
                    neg_scale = 1.0
                elif top_aff in ['scary']:
                    pos_scale = 1.0
                    neg_scale = 1.0
                elif top_aff in ['sad', 'romantic']:
                    pos_scale = 0.9
                    neg_scale = 1.0
                elif top_aff in ['funny', 'family']:
                    pos_scale = 0.9
                    neg_scale = 0.9
        qv_use = steer_query_vector(qv_raw, pos_scale=pos_scale, neg_scale=neg_scale)
        sims_for_genres = (genre_centroids @ qv_use).astype(np.float32) if genre_centroids is not None else np.zeros(len(ML_GENRES), dtype=np.float32)
        sims_for_genres = np.clip(sims_for_genres, 0.0, None)

        # Affect prior
        affect_embs = get_affect_embeddings()
        g2i = {g: i for i, g in enumerate(ML_GENRES)}
        affect_weights = np.zeros(len(ML_GENRES), dtype=np.float32)
        top_affect = None
        affect_conf = 0.0
        if affect_embs:
            affect_scores = {k: float(np.dot(v, qv_use)) for k, v in affect_embs.items()}
            # keep positives and normalize
            aff_items = [(k, s) for k, s in affect_scores.items() if s > 0]
            total = sum(s for _, s in aff_items)
            if total > 0:
                # Special handling: if "exciting" and "scary" are both present,
                # and query contains action-oriented keywords, boost "exciting"
                exciting_score = affect_scores.get('exciting', 0)
                scary_score = affect_scores.get('scary', 0)
                # Check for action-oriented phrases in query
                action_keywords = ['pumping', 'pump', 'adrenaline', 'heart racing', 'action', 'thrilling', 'awesome']
                if exciting_score > 0 and scary_score > 0:
                    query_lower = q.lower()
                    if any(kw in query_lower for kw in action_keywords):
                        # Boost exciting when action keywords present
                        exciting_boost = min(0.15, scary_score * 0.3)
                        affect_scores['exciting'] = exciting_score + exciting_boost
                        # Slightly reduce scary if exciting is boosted
                        if scary_score > exciting_score:
                            affect_scores['scary'] = scary_score * 0.85
                        # Recalculate aff_items with boosted scores
                        aff_items = [(k, s) for k, s in affect_scores.items() if s > 0]
                        total = sum(s for _, s in aff_items)
                
                for k, s in aff_items:
                    prior = AFFECT_GENRE_PRIOR.get(k, {})
                    for g, w in prior.items():
                        if g in g2i:
                            affect_weights[g2i[g]] += (s / total) * float(w)
                # remember the strongest affect for conflict suppression
                top_affect, affect_conf = max(aff_items, key=lambda x: x[1])
            # Log top-3 affect scores
            top_affects = sorted(affect_scores.items(), key=lambda x: -x[1])[:3]
            logger.info(f"affects_top={top_affects}")

        # Exclude weakly related genres before normalization (adaptive)
        exclude_thresh = 0.25 if affect_conf >= 0.35 else 0.20
        sims_for_genres = np.where(sims_for_genres >= exclude_thresh, sims_for_genres, 0.0)

        # Combine centroid sims and affect prior
        beta = 0.9 if affect_conf >= 0.35 else 0.7
        combined = sims_for_genres + beta * affect_weights

        # Conflict suppression
        if top_affect and top_affect in CONFLICT_SUPPRESS:
            for g in CONFLICT_SUPPRESS[top_affect]:
                if g in g2i:
                    combined[g2i[g]] *= (0.2 if affect_conf >= 0.35 else 0.5)

        genre_weights = combined / combined.sum() if combined.sum() > 0 else combined
        # Per-affect alpha and pool adjustments (soft runtime tuning)
        if top_affect in ['sad', 'romantic'] and affect_conf >= 0.35:
            # Lean more on genre; reduce semantic dominance and narrow pool
            new_embed_alpha = min(embed_alpha, 0.50)
            new_genre_alpha = max(genre_alpha, 0.50)
            new_pool = min(candidate_pool, 300)
            if (new_embed_alpha != embed_alpha) or (new_genre_alpha != genre_alpha) or (new_pool != candidate_pool):
                logger.info(f"affect_tune: top_affect={top_affect} conf={affect_conf:.3f} -> embed_alpha {embed_alpha}->${new_embed_alpha}, genre_alpha {genre_alpha}->{new_genre_alpha}, pool {candidate_pool}->{new_pool}")
            embed_alpha = new_embed_alpha
            genre_alpha = new_genre_alpha
            candidate_pool = new_pool
        # Log top-3 inferred genres with weights
        _topg = np.argsort(-genre_weights)[:3]
        logger.info(f"genres_top={[(ML_GENRES[i], float(genre_weights[i])) for i in _topg]}")
        pop_w = 0.0

    # Early strict filter: keep only items that contain one of top inferred genres
    if strict and genre_weights.sum() > 0:
        k = 2  # top-K inferred genres
        # When affect is confident, prefer top genres from affect prior; else from combined weights
        if 'affect_weights' in locals() and affect_weights.sum() > 0 and 'affect_conf' in locals() and affect_conf >= 0.35:
            topg = np.argsort(-affect_weights)[:k]
        else:
            topg = np.argsort(-genre_weights)[:k]
        topg = topg[genre_weights[topg] > 0]
        if len(topg) > 0:
            if 'top_affect' in locals() and top_affect in ['sad','romantic'] and affect_conf >= 0.35 and len(topg) >= 2:
                # require both top genres to be present for high-confidence sad/romantic
                keep_mask = (genre_mat[remaining][:, topg].sum(axis=1) >= len(topg))
            else:
                keep_mask = (genre_mat[remaining][:, topg].sum(axis=1) > 0)
            remaining = remaining[keep_mask]
            if len(remaining) == 0:
                return []
    logger.info(f"remaining_after_strict={len(remaining)}")

    # Retrieval by embeddings (optional) using the steered vector
    if item_emb is not None and 'qv_use' in locals() and qv_use is not None:
        sims_all = (item_emb @ qv_use).astype(np.float32)
        sims = sims_all[remaining]
        topN = min(candidate_pool, len(remaining))
        cand_idx = np.argpartition(-sims, topN-1)[:topN]
        candidates = remaining[cand_idx]
        embed_bonus = sims[cand_idx]
        logger.info(f"retrieval_pool={len(candidates)} embed_bonus_mean={float(embed_bonus.mean()):.4f}")
    else:
        candidates = remaining
        embed_bonus = np.zeros(len(candidates), dtype=np.float32)

    if len(candidates) == 0:
        return []

    user_tensor = torch.tensor([uid] * len(candidates)).long().to(DEVICE)
    item_tensor = torch.tensor(candidates).long().to(DEVICE)
    genre_tensor = torch.tensor(genre_mat[candidates]).float().to(DEVICE)
    with torch.no_grad():
        # intent vector for batch (if available)
        intent_tensor = None
        if 'qv_use' in locals() and qv_use is not None:
            intent_tensor = torch.tensor(np.tile(qv_use, (len(candidates), 1))).float().to(DEVICE)
        base_scores = model.predict(user_tensor, item_tensor, genre_tensor, intent_tensor).cpu().numpy()
    logger.info(f"base_scores_mean={float(base_scores.mean()):.4f}")

    g_bonus = (genre_mat[candidates] @ genre_weights).astype(np.float32)
    p_bonus = item_pop[candidates] * pop_w
    scores = base_scores + genre_alpha * g_bonus + pop_alpha * p_bonus + embed_alpha * embed_bonus
    logger.info(f"mix: g_mean={float(g_bonus.mean()):.4f} p_mean={float(p_bonus.mean()):.4f} e_mean={float(embed_bonus.mean()):.4f}")

    # Prioritize items matching the top inferred genres even when strict=False
    topg_final = np.argsort(-genre_weights)[:2]
    if len(topg_final) > 0:
        match_mask_final = (genre_mat[candidates][:, topg_final].sum(axis=1) > 0)
        order = np.argsort(-scores)
        primary = [i for i in order if match_mask_final[i]]
        secondary = [i for i in order if not match_mask_final[i]]
        chosen = (primary + secondary)[:top_k]
        top_idx = np.array(chosen, dtype=np.int64)
    else:
        top_idx = np.argsort(-scores)[:top_k]
    result = []
    for i in top_idx:
        idx = candidates[i]
        mlid = item_invmap[idx]
        title = meta[mlid]['title']
        genres = meta[mlid]['genres']
        result.append(RecResponse(movie_id=mlid, title=title, genres=genres, score=float(scores[i])))
    logger.info(f"top_titles={[r.title for r in result]}")
    return result
