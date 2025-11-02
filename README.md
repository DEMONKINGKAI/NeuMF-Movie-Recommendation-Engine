# NeuMF Genre-Aware Movie Recommendation Engine

A comprehensive Neural Matrix Factorization (NeuMF) recommender system that provides personalized movie recommendations using genre-aware collaborative filtering and natural language intent understanding. Built with PyTorch for training, FastAPI for serving, and React for the frontend interface.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Architecture](#architecture)
4. [Mathematical Foundations](#mathematical-foundations)
5. [NLP Intent System](#nlp-intent-system)
6. [Recent Improvements](#recent-improvements)
7. [System Workflow](#system-workflow)
8. [Installation & Setup](#installation--setup)
9. [Usage](#usage)
10. [Project Structure](#project-structure)

---

## Project Overview

This project implements a state-of-the-art recommendation system that combines:

1. **Neural Matrix Factorization (NeuMF)**: A hybrid deep learning model that fuses Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP) architectures to learn user-item interactions
2. **Genre-Aware Filtering**: Incorporates movie genre information as multi-hot vectors to enhance recommendation quality
3. **NLP Intent Understanding**: Uses semantic embeddings to interpret free-form user queries (e.g., "I want something exciting and thrilling") and map them to appropriate movie recommendations
4. **Advanced Affect Detection**: Automatically detects emotional intents (sad, funny, scary, romantic, etc.) with keyword-aware boosting to better interpret user intent
5. **Intent Tower Integration**: Optional neural network processing of semantic intent vectors during training and inference

The system supports both **MovieLens 100K** (small, older movies from 1990s) and **MovieLens 25M** (large, recent movies up to 2019) datasets, automatically detecting the format. It provides both genre-based and intent-based recommendation endpoints through a RESTful API, accessible via a modern React frontend.

**Recommended**: Use MovieLens 25M for better performance and more recent movies (default).

---

## Technologies Used

### Backend & Training
- **Python 3.9+**: Core programming language
- **PyTorch**: Deep learning framework for model training and inference
- **FastAPI**: High-performance web framework for the recommendation API
- **sentence-transformers**: Semantic text embeddings for NLP intent processing
- **NumPy & Pandas**: Data processing and manipulation
- **Uvicorn**: ASGI server for FastAPI
- **YAML**: Configuration file parsing

### Frontend
- **React**: UI framework
- **Vite**: Modern build tool and dev server
- **Anime.js**: Smooth animations for UI interactions
- **Axios**: HTTP client with interceptors for API communication

### Data
- **MovieLens 100K**: Movie rating dataset with 100,000 ratings from 943 users on 1,682 movies (1990s movies)
- **MovieLens 25M**: Movie rating dataset with 25,000,000 ratings from 162,541 users on 62,423 movies (recent movies up to 2019) - **Recommended**

---

## Architecture

The system consists of three main components:

### 1. Training Pipeline (`recsys/`)
- **Data Loading** (`data.py`): Automatically detects and loads MovieLens 100K or 25M format, creates train/val/test splits using leave-one-out methodology, supports optional sampling for faster experiments
- **Model Definition** (`model.py`): Implements the NeuMF architecture with optional intent tower
- **Training** (`train.py`): Trains the model using binary cross-entropy loss with negative sampling, memory-efficient intent vector handling
- **Evaluation** (`eval.py`): Evaluates model performance using Hit Rate (HR@K) and Normalized Discounted Cumulative Gain (NDCG@K)

### 2. Backend API (`backend/`)
- **FastAPI Service** (`main.py`): Serves recommendation endpoints with config-based hyperparameter loading
  - `/recommendations`: Genre-based recommendations
  - `/intent_recommendations`: NLP-based intent recommendations with advanced affect detection
  - `/genres`: List available genres (dynamically loaded from dataset)
  - `/users`: List available user IDs
- **Embedding System**: Loads pre-computed item embeddings for semantic search
- **Intent Mapping**: Maps natural language queries to genre weights and affects with keyword-aware boosting

### 3. Frontend UI (`frontend/`)
- **React Application**: Interactive interface with error handling and loading states
  - Selecting users and genres
  - Free-text prompt search with tunable alpha parameters
  - Displaying recommendations with scores
  - Real-time API request/response logging

---

## Mathematical Foundations

### Generalized Matrix Factorization (GMF)

GMF captures linear user-item interactions through elementwise product of embeddings:

Given user embedding **p<sub>u</sub>** ∈ ℝ<sup>k</sup> and item embedding **q<sub>i</sub>** ∈ ℝ<sup>k</sup> (where k = `emb_dim_gmf`), the GMF output is:

**h<sub>GMF</sub> = p<sub>u</sub> ⊙ q<sub>i</sub>**

Where ⊙ denotes elementwise (Hadamard) product:

**h<sub>GMF</sub>[j] = p<sub>u</sub>[j] × q<sub>i</sub>[j]** for j = 1, ..., k

This captures multiplicative interactions between user and item latent factors, similar to matrix factorization but in a neural framework.

**Embedding Initialization:**
- **p<sub>u</sub>** = **user_gmf_emb**(u) where **user_gmf_emb**: U → ℝ<sup>k</sup>
- **q<sub>i</sub>** = **item_gmf_emb**(i) where **item_gmf_emb**: I → ℝ<sup>k</sup>

Both embeddings are learned during training via backpropagation.

### Neural Matrix Factorization (NeuMF)

NeuMF combines GMF with a Multi-Layer Perceptron (MLP) to capture both linear and non-linear interactions:

#### 1. GMF Component:
```
GMF Output: h_GMF = user_gmf_emb(u) ⊙ item_gmf_emb(i)
           h_GMF ∈ ℝ^k where k = emb_dim_gmf
```

#### 2. MLP Component:

The MLP processes concatenated user, item, genre, and optional intent embeddings:

**MLP Input Construction:**

**x<sub>MLP</sub> = [user_mlp_emb(u) || item_mlp_emb(i) || genre_proj(g) || intent_tower(v)]**

Where:
- **user_mlp_emb(u)**: User embedding in MLP space ∈ ℝ<sup>d<sub>mlp</sub></sup>
- **item_mlp_emb(i)**: Item embedding in MLP space ∈ ℝ<sup>d<sub>mlp</sub></sup>
- **genre_proj(g)**: Projected genre vector ∈ ℝ<sup>d<sub>genre</sub></sup>
  - **g** is a multi-hot genre vector (e.g., [0,1,0,1,0,...] for Action + Adventure)
  - **genre_proj(g) = W<sub>genre</sub> · g + b<sub>genre</sub>**
  - W<sub>genre</sub> ∈ ℝ<sup>|G|×d<sub>genre</sub></sup>, where |G| is the number of genres (19 for 100K, dynamic for 25M)
- **intent_tower(v)**: Optional intent embedding processed through a neural network ∈ ℝ<sup>d<sub>intent</sub></sup>
  - If present: **intent_tower(v) = ReLU(W<sub>2</sub> · Dropout(ReLU(W<sub>1</sub> · v + b<sub>1</sub>)) + b<sub>2</sub>)**
  - Where v ∈ ℝ<sup>d<sub>embed</sub></sup> (typically 384 for all-MiniLM-L6-v2)
  - W<sub>1</sub> ∈ ℝ<sup>d<sub>embed</sub>×h</sup>, W<sub>2</sub> ∈ ℝ<sup>h×d<sub>intent</sub></sup>
  - If not present: **intent_tower(v) = 0** (zero vector)

**Total MLP Input Dimension:**
**d<sub>in</sub> = 2·d<sub>mlp</sub> + d<sub>genre</sub> + d<sub>intent</sub>**

The MLP then applies multiple fully-connected layers:

**h<sub>1</sub> = ReLU(W<sub>1</sub> · x<sub>MLP</sub> + b<sub>1</sub>)**

**h<sub>2</sub> = ReLU(W<sub>2</sub> · h<sub>1</sub> + b<sub>2</sub>)**

**...**

**h<sub>L</sub> = h<sub>MLP</sub>**  (final MLP layer output)

Where the layer sizes are specified by `mlp_layers` (e.g., [128, 64]).

#### 3. NeuMF Final Prediction:

The model concatenates GMF and MLP outputs and applies a final linear layer:

**Final Input:** **z = [h<sub>GMF</sub> || h<sub>MLP</sub>]**

**z ∈ ℝ<sup>k + |mlp_layers[-1]|</sup>**

**Final Score:** **ŷ = W<sub>final</sub> · z + b<sub>final</sub>**

Where W<sub>final</sub> ∈ ℝ<sup>(k+|mlp_layers[-1]|)×1</sup>

**Predicted Probability:** **p(interaction | u, i, g, v) = σ(ŷ) = 1 / (1 + exp(-ŷ))**

Where σ is the sigmoid function.

#### 4. Training Objective:

The model is trained using Binary Cross-Entropy Loss with negative sampling:

**L = -[y · log(σ(ŷ)) + (1-y) · log(1-σ(ŷ))]**

Where:
- **y = 1** for positive user-item interactions
- **y = 0** for negative samples (randomly sampled non-interacted items)

**Batch Loss:** For a batch of size B:

**L<sub>batch</sub> = (1/B) · Σ<sub>i=1</sub><sup>B</sup> L<sub>i</sub>**

**Optimization:** Adam optimizer with learning rate η:

**θ<sub>t+1</sub> = θ<sub>t</sub> - η · ∇<sub>θ</sub>L<sub>batch</sub>**

---

## NLP Intent System

The NLP intent system enables users to express preferences in natural language (e.g., "I want something exciting and funny") and maps these queries to personalized recommendations. The system operates in several stages with advanced keyword-aware affect boosting:

### 1. Text Embedding

User queries are converted to dense vectors using a sentence transformer model:

**q<sub>raw</sub> = SentenceTransformer(text)**  ∈ ℝ<sup>d<sub>embed</sub></sup>

Where d<sub>embed</sub> = 384 for `all-MiniLM-L6-v2`.

The embedding is L2-normalized: **q<sub>raw</sub> = q<sub>raw</sub> / ||q<sub>raw</sub>||<sub>2</sub>**

This ensures unit-length vectors for proper cosine similarity computation.

### 2. Genre Centroid Computation

For each genre, a centroid is computed as the mean embedding of all movies in that genre:

For genre **g**, with items **I<sub>g</sub>** = {i : genre<sub>i</sub>[g] = 1}:

**c<sub>g</sub> = (1/|I<sub>g</sub>|) · Σ<sub>i∈I<sub>g</sub></sub> e<sub>i</sub>**

Where **e<sub>i</sub>** is the pre-computed embedding for item i (computed from "title + genres" text).

The centroid is normalized: **c<sub>g</sub> = c<sub>g</sub> / ||c<sub>g</sub>||<sub>2</sub>**

**Genre Centroid Matrix:** **C = [c<sub>1</sub>, c<sub>2</sub>, ..., c<sub>|G|</sub>]<sup>T</sup>** ∈ ℝ<sup>|G|×d<sub>embed</sub></sup>

### 3. Query Steering

The query vector is "steered" toward relevant genre centroids and away from irrelevant ones:

**Step 1: Compute similarities**
**sims = C · q<sub>raw</sub>**  ∈ ℝ<sup>|G|</sup>

**Step 2: Identify top and bottom genres**
**top_genres = argmax_k(sims)**  (top K most similar genres, default k=3)
**bot_genres = argmin_k(sims)**  (bottom K least similar genres, default k=2)

**Step 3: Compute direction vectors**
**c<sub>top</sub> = mean({c<sub>g</sub> : g ∈ top_genres})**
**c<sub>bot</sub> = mean({c<sub>g</sub> : g ∈ bot_genres})**

**Step 4: Steer query vector**
**q<sub>steered</sub> = q<sub>raw</sub> + α<sub>pos</sub> · c<sub>top</sub> - α<sub>neg</sub> · c<sub>bot</sub>**

Where α<sub>pos</sub> and α<sub>neg</sub> are adaptive scaling factors (typically 0.8-1.1 for α<sub>pos</sub>, 0.9-1.0 for α<sub>neg</sub>).

**Step 5: Renormalize**
**q<sub>steered</sub> = q<sub>steered</sub> / ||q<sub>steered</sub>||<sub>2</sub>**

This sharpens the query vector to better match user intent.

### 4. Affect Detection

The system detects emotional intents using predefined affect anchors with keyword-aware boosting:

#### 4.1 Affect Anchor Embeddings

For each affect **a** ∈ {sad, funny, scary, romantic, exciting, inspiring, family, dark} with anchor phrase **p<sub>a</sub>**:

**a<sub>emb</sub> = embed(p<sub>a</sub>) / ||embed(p<sub>a</sub>)||<sub>2</sub>**

These are precomputed and cached for efficiency.

#### 4.2 Initial Affect Scoring

For the steered query vector **q<sub>steered</sub>**, compute cosine similarity with each affect:

**affect_score(a) = a<sub>emb</sub><sup>T</sup> · q<sub>steered</sub>**

**affect_scores = {a: affect_score(a) for a ∈ AFFECTS}**

#### 4.3 Keyword-Aware Affect Boosting

**Action Keyword Detection:**
Define action keywords: **K<sub>action</sub>** = {'pumping', 'pump', 'adrenaline', 'heart racing', 'action', 'thrilling', 'awesome'}

**If any keyword k ∈ K<sub>action</sub> appears in query q:**

**If affect_score('exciting') > 0 and affect_score('scary') > 0:**

**exciting_boost = min(0.15, affect_score('scary') × 0.3)**

**affect_score('exciting') ← affect_score('exciting') + exciting_boost**

**If affect_score('scary') > affect_score('exciting'):**

**affect_score('scary') ← affect_score('scary') × 0.85**

This ensures action-oriented queries prioritize "exciting" over "scary".

#### 4.4 Additional Steering Adjustment

**If top_affect == 'exciting' (after boosting):**

**α<sub>pos</sub> = 1.1**  (stronger pull toward action genres)

**If top_affect == 'scary':**

**α<sub>pos</sub> = 1.0**

**For other affects:**
**α<sub>pos</sub> = 0.9** (moderate pull)

#### 4.5 Affect Normalization

Keep only positive affect scores:
**aff_items = {(a, s) : (a, s) ∈ affect_scores.items() ∧ s > 0}**

**total = Σ<sub>(a,s)∈aff_items</sub> s**

**Normalized scores:** **aff_norm(a) = s / total** for each (a, s) ∈ aff_items

**Top affect:** **top_affect = argmax<sub>a</sub> aff_norm(a)**

**Affect confidence:** **affect_conf = max(aff_norm)**

### 5. Genre Weight Computation

Genre weights are computed by combining centroid similarities and affect priors:

#### 5.1 Centroid Similarities

**sim<sub>g</sub> = c<sub>g</sub><sup>T</sup> · q<sub>steered</sub>**

**sims_genres = [sim<sub>1</sub>, sim<sub>2</sub>, ..., sim<sub>|G|</sub>]**

Clip negative similarities: **sims_genres = max(0, sims_genres)**  (elementwise)

#### 5.2 Affect Genre Priors

For each affect **a** with normalized score **aff_norm(a)**, apply genre priors:

**P(g | a)** is predefined (e.g., P(Action | exciting) = 0.7, P(Thriller | exciting) = 0.25)

**affect_weights<sub>g</sub> = Σ<sub>a</sub> [aff_norm(a) × P(g | a)]**

For all genres g:
**affect_weights = [affect_weights<sub>1</sub>, affect_weights<sub>2</sub>, ..., affect_weights<sub>|G|</sub>]**

#### 5.3 Thresholding and Combination

**Exclude weakly related genres:**
**exclude_thresh = 0.25 if affect_conf ≥ 0.35 else 0.20**

**sims_genres = sims_genres if sims_genres ≥ exclude_thresh else 0**  (elementwise)

**Combine signals:**
**β = 0.9 if affect_conf ≥ 0.35 else 0.7**

**combined = sims_genres + β × affect_weights**

#### 5.4 Conflict Suppression

**If top_affect ∈ CONFLICT_SUPPRESS:**

**For each conflicting genre g ∈ CONFLICT_SUPPRESS[top_affect]:**

**combined[g] ← combined[g] × (0.2 if affect_conf ≥ 0.35 else 0.5)**

This reduces weight for genres that conflict with the detected affect (e.g., Horror suppressed for "exciting" queries).

#### 5.5 Final Normalization

**If Σ<sub>g</sub> combined[g] > 0:**

**genre_weights[g] = combined[g] / Σ<sub>g'</sub> combined[g']**

**Else:**

**genre_weights = combined**  (unchanged)

### 6. Candidate Retrieval & Scoring

#### 6.1 Embedding-Based Candidate Retrieval

**Compute item similarities:**
**item_sims = E · q<sub>steered</sub>**  ∈ ℝ<sup>|I|</sup>

Where **E** is the item embedding matrix ∈ ℝ<sup>|I|×d<sub>embed</sub></sup>

**Select top candidates:**
**candidates = argtop_k(item_sims, candidate_pool)**

**embed_bonus = item_sims[candidates]**  (embedding similarity scores for candidates)

#### 6.2 Base Model Scoring

For each candidate item **i** in candidates:

**base_score[i] = σ(ŷ) = model.predict(user_id, i, genre_vector[i], intent_vector)**

Where **intent_vector = q<sub>steered</sub>** if intent tower is enabled.

#### 6.3 Bonus Computations

**Genre Bonus:**
**genre_bonus[i] = Σ<sub>g</sub> [genre_weights[g] × genre_vector[i][g]]**

This is the dot product between genre weights and the item's genre vector.

**Popularity Bonus:**
**pop_bonus[i] = popularity[i] × pop_w**

Where **popularity[i]** is normalized interaction count for item i, and **pop_w** is popularity weight (typically 0 for affect-based queries).

**Embedding Bonus:**
**embed_bonus[i]** (already computed in step 6.1)

#### 6.4 Final Score Combination

**final_score[i] = base_score[i] + α<sub>genre</sub> × genre_bonus[i] + α<sub>pop</sub> × pop_bonus[i] + α<sub>embed</sub> × embed_bonus[i]**

Where:
- **α<sub>genre</sub>**: User-tunable weight for genre matching (default: 0.35)
- **α<sub>pop</sub>**: User-tunable weight for popularity (default: 0.05)
- **α<sub>embed</sub>**: User-tunable weight for semantic similarity (default: 0.60)

#### 6.5 Ranking and Selection

**Priority-based ranking:**

**top_genres_final = argmax_2(genre_weights)**  (top 2 inferred genres)

**For each candidate i:**

**match_mask[i] = (Σ<sub>g∈top_genres_final</sub> genre_vector[i][g]) > 0**

**Sort candidates by final_score in descending order:**

**order = argsort(-final_score)**

**Split into primary (matching top genres) and secondary:**

**primary = [i ∈ order : match_mask[i]]**

**secondary = [i ∈ order : ¬match_mask[i]]**

**Final selection:** **selected = (primary || secondary)[:top_k]**

This ensures items matching inferred genres are prioritized even when strict=False.

---

## Recent Improvements

### 1. MovieLens 25M Dataset Support

**Auto-detection**: The system automatically detects whether you're using MovieLens 100K or 25M format and adjusts accordingly.

**Format Differences:**
- **100K**: Tab-separated files, 19 fixed genres, binary genre encoding
- **25M**: CSV files, dynamic genre list (pipe-separated), more recent movies (up to 2019)

**Memory Optimization**: Added `max_ratings` parameter to sample subsets for faster experimentation:
```bash
python main.py --data ./data/ml-25m --max-ratings 500000 --epochs 3
```

### 2. Memory-Efficient Training

**Problem**: Original implementation tried to store all intent vectors in memory, requiring ~86.8 GB for 25M dataset.

**Solution**: Implemented lazy loading - intent vectors are looked up on-the-fly during batch iteration:
- Store only item indices (8 bytes each) instead of full embeddings (1536 bytes each)
- 99.4% memory reduction: ~86.8 GB → ~485 MB
- No performance degradation in practice

### 3. Keyword-Aware Affect Boosting

**Problem**: Queries like "make my blood pumping" incorrectly prioritized "scary" (0.47) over "exciting" (0.36), leading to horror movie recommendations instead of action.

**Solution**: 
- Detects action-oriented keywords: 'pumping', 'adrenaline', 'thrilling', 'awesome', etc.
- Boosts "exciting" affect when action keywords are present: **exciting_score ← exciting_score + min(0.15, scary_score × 0.3)**
- Reduces "scary" when it dominates: **scary_score ← scary_score × 0.85**
- Enhanced steering for exciting queries: **α<sub>pos</sub> = 1.1** (vs 1.0 for scary)

### 4. Enhanced Intent System

**Config-based hyperparameters**: Backend loads model hyperparameters from `configs/starter.yaml` to match training exactly.

**Dynamic genre lists**: Genre lists are dynamically loaded from the dataset (19 genres for 100K, 19-20+ for 25M including IMAX).

**Improved genre centroids**: Genre centroids are computed from actual embeddings, ensuring semantic coherence.

### 5. Frontend Improvements

**Error handling**: Added comprehensive error handling and logging:
- API request/response interceptors
- Network error handling
- 30-second timeout for API requests

**Loading states**: Proper loading indicators during initialization and recommendation fetching.

**User feedback**: Console logging for debugging and user feedback.

### 6. Dataset Compatibility

**Flexible format support**: Single codebase supports both 100K and 25M with automatic detection:
- Detects format based on file structure
- Handles different genre encoding (binary vs. pipe-separated)
- Adapts metadata loading accordingly

---

## System Workflow

### End-to-End Flow

```
1. Data Preparation
   └─> Load MovieLens dataset (auto-detect 100K or 25M format)
   └─> Create user/item ID mappings (internal indexing)
   └─> Build genre matrix (multi-hot vectors)
       • 100K: 19 fixed genres with binary encoding
       • 25M: Dynamic genres with pipe-separated encoding
   └─> Split into train/val/test (leave-one-out methodology)
   └─> Generate negative samples for training (4 per positive by default)

2. Embedding Generation (Optional, for NLP Intent System)
   └─> Build text descriptions: "Movie Title. Genres: Action, Adventure."
   └─> Encode using SentenceTransformer (all-MiniLM-L6-v2)
   └─> Normalize embeddings to unit length
   └─> Save to item_embeddings.npy (shape: [num_items, 384])

3. Model Training
   └─> Initialize NeuMF model:
       ├─> GMF embeddings: [num_users, emb_dim_gmf], [num_items, emb_dim_gmf]
       ├─> MLP embeddings: [num_users, emb_dim_mlp], [num_items, emb_dim_mlp]
       ├─> Genre projector: [num_genres, genre_proj_dim]
       └─> Intent tower (if embeddings available): [384 → 128 → 64]
   └─> For each epoch:
       ├─> Sample positive interactions from train set
       ├─> Sample negative interactions (neg_per_pos per positive)
       ├─> Forward pass: compute predictions
       ├─> Compute BCE loss: L = -[y·log(σ(ŷ)) + (1-y)·log(1-σ(ŷ))]
       └─> Backpropagate and update weights (Adam optimizer)
   └─> Evaluate on test set (HR@10, NDCG@10)
   └─> Save model checkpoint to checkpoints/neumf_final.pt

4. API Startup
   └─> Load config from configs/starter.yaml
   └─> Load dataset and build mappings
   └─> Load trained model weights (with compatibility checking)
   └─> Initialize model with exact hyperparameters from config
   └─> Load item embeddings (if available)
   └─> Compute genre centroids from embeddings
   └─> Pre-compute popularity scores (normalized interaction counts)
   └─> Cache affect anchor embeddings
   └─> Start FastAPI server with CORS enabled

5. Recommendation Request (Genre-based)
   └─> Receive: user_id, genre, top_k, strict
   └─> Map user_id to internal index
   └─> Get all items user hasn't interacted with
   └─> Filter candidates by genre:
       • Strict: Only items with selected genre
       • Soft: All items, but genre matching gets bonus
   └─> Run NeuMF model on candidates (batch inference)
   └─> Apply genre bonus if soft mode
   └─> Rank by prediction scores
   └─> Return top_k movies with metadata

6. Recommendation Request (Intent-based)
   └─> Receive: query text, user_id, top_k, alphas, strict
   └─> Embed query text → q_raw (384-dim vector)
   └─> Detect action keywords → boost exciting affect if present
   └─> Compute affect scores → normalize and identify top_affect
   └─> Adjust steering parameters based on top_affect
   └─> Steer query vector toward relevant genres
   └─> Compute genre weights (centroid sims + affect priors)
   └─> Apply conflict suppression if needed
   └─> Filter candidates by top genres if strict=True
   └─> Retrieve candidates via embedding similarity (top candidate_pool)
   └─> Run NeuMF model with intent vector on candidates
   └─> Combine scores: base + α_genre·genre + α_pop·pop + α_embed·embed
   └─> Prioritize items matching top genres
   └─> Return top_k movies
```

---

## Installation & Setup

### Prerequisites
- Python 3.9+ (3.10+ recommended, **Note**: PyTorch with CUDA requires Python ≤3.12)
- Node.js 18+ and npm
- Optional: CUDA-enabled GPU for faster training/inference

### 1. Clone and Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source .venv/bin/activate
```

### 2. Install Python Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install backend dependencies
pip install -r backend/requirements.txt
```

### 3. Download Dataset

**Important**: The data files are not included in the repository (they are too large for GitHub). You must download them using the provided script or manually from MovieLens.

**Option 1: MovieLens 25M (Recommended - larger, more recent movies)**
```bash
python scripts/download_mlwk.py --dataset 25m --target ./data/ml-25m
```

**Option 2: MovieLens 100K (Smaller, older movies from 1990s)**
```bash
python scripts/download_mlwk.py --dataset 100k --target ./data/ml-100k
```

**Note**: MovieLens 25M is ~250MB download and may take a few minutes. The system automatically detects the dataset format.

**Essential Files Required:**

For MovieLens 25M:
```
data/ml-25m/ml-25m/
  ├─ ratings.csv    ✓ REQUIRED (ratings: userId,movieId,rating,timestamp)
  ├─ movies.csv     ✓ REQUIRED (movie metadata: movieId,title,genres)
  └─ ... (other files are optional and excluded from git)
```

For MovieLens 100K:
```
data/ml-100k/ml-100k/
  ├─ u.data         ✓ REQUIRED (ratings)
  ├─ u.item         ✓ REQUIRED (movie metadata with genres)
  └─ ... (other files are optional and excluded from git)
```

**What's Excluded**: The repository uses `.gitignore` to exclude non-essential files like pre-split test sets (`u1.base`, `u1.test`, etc.), documentation (`README`, `u.info`), additional metadata (`u.user`, `genome-tags.csv`), and scripts (`allbut.pl`). Only the essential rating and movie metadata files listed above are needed for both training and backend runtime.

### 4. Train the Model

**For MovieLens 25M (recommended):**
```bash
python main.py --data ./data/ml-25m --epochs 10
```

**For MovieLens 100K:**
```bash
python main.py --data ./data/ml-100k --epochs 10
```

**Optional (faster experiments):** limit the number of ratings by sampling from the dataset:
```bash
python main.py --data ./data/ml-25m --max-ratings 500000 --epochs 3
```

**Note**: The system automatically detects the dataset format. For MovieLens 25M, you may want to adjust hyperparameters in `configs/starter.yaml` or use the `--max-ratings` parameter to limit training data size for faster experimentation.

This will:
- Load and preprocess the dataset
- Train the NeuMF model
- Evaluate on test set (reports HR@10 and NDCG@10)
- Save model to `./checkpoints/neumf_final.pt`

**Configuration**: Edit `configs/starter.yaml` to adjust hyperparameters:
- `emb_dim_gmf`: GMF embedding dimension (default: 32)
- `emb_dim_mlp`: MLP embedding dimension (default: 64)
- `mlp_layers`: MLP layer sizes (default: [128, 64])
- `lr`: Learning rate (default: 0.001)
- `batch_size`: Training batch size (default: 256)
- `epochs`: Number of training epochs (default: 10)
- `neg_per_pos`: Negative samples per positive (default: 4)

### 5. Build Item Embeddings (Optional, for NLP Intent System)

**For MovieLens 25M:**
```bash
python scripts/build_item_embeddings.py --data ./data/ml-25m --out ./checkpoints/item_embeddings.npy
```

**For MovieLens 100K:**
```bash
python scripts/build_item_embeddings.py --data ./data/ml-100k --out ./checkpoints/item_embeddings.npy
```

This creates semantic embeddings for each movie (title + genres) using sentence-transformers, enabling the NLP intent recommendation feature. The embeddings are stored as a NumPy array with shape `[num_items, 384]`.

### 6. Start the Backend API

**Windows PowerShell:**
```powershell
$env:MOVIELENS_PATH = ".\data\ml-25m"  # or ".\data\ml-100k" for 100K dataset
$env:MODEL_PATH = ".\checkpoints\neumf_final.pt"
$env:EMB_PATH = ".\checkpoints\item_embeddings.npy"  # optional
$env:EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # optional
uvicorn backend.main:app --reload --port 8000
```

**macOS/Linux:**
```bash
export MOVIELENS_PATH=./data/ml-25m  # or ./data/ml-100k for 100K dataset
export MODEL_PATH=./checkpoints/neumf_final.pt
export EMB_PATH=./checkpoints/item_embeddings.npy  # optional
export EMB_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2  # optional
uvicorn backend.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

**Note**: On first startup, the API loads the dataset and model into memory. This may take 1-3 minutes for MovieLens 25M. Wait for "Application startup complete" before making requests.

**API Endpoints:**
- `GET /genres` → List of available genres (dynamically loaded from dataset)
- `GET /users` → List of user IDs (original MovieLens user IDs)
- `GET /recommendations?user_id=<id>&genre=<name>&top_k=<n>&strict=<true|false>` → Genre-based recommendations
- `GET /intent_recommendations?q=<prompt>&user_id=<id>&top_k=<n>&strict=<bool>&genre_alpha=<f>&pop_alpha=<f>&embed_alpha=<f>` → NLP intent-based recommendations

### 7. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the URL shown by Vite (typically `http://localhost:5173`)

---

## Usage

### Genre-Based Recommendations

1. Select a user from the dropdown
2. Select a genre (e.g., "Action", "Comedy")
3. Choose "Strict" mode (only movies with selected genre) or "Soft" mode (prefers matching genre but allows others)
4. Adjust "Top K" to control number of recommendations
5. Recommendations appear automatically

**Soft Mode Formula:**
**final_score = base_score + 0.20 × genre_match_bonus**

Where genre_match_bonus = 1.0 if item has selected genre, 0.0 otherwise.

### NLP Intent-Based Recommendations

1. Enter a free-form text query in the prompt box, e.g.:
   - "I want something exciting and thrilling"
   - "show me a sad romantic movie"
   - "the movie should be awesome and make my blood pumping"
   - "something funny for the family"
2. Adjust the alpha parameters (live tuning):
   - **Genre α**: Weight for genre matching (0.0-1.0, default: 0.35)
   - **Popularity α**: Weight for popular movies (0.0-1.0, default: 0.05)
   - **Embedding α**: Weight for semantic similarity (0.0-1.0, default: 0.60)
3. Click "Recommend"
4. The system will:
   - Parse your query semantically using sentence transformers
   - Detect emotional affects (sad, funny, scary, etc.) with keyword-aware boosting
   - Boost "exciting" affect if action keywords detected
   - Infer genre preferences through centroid similarity and affect priors
   - Steer query vector toward relevant genres
   - Retrieve candidates via embedding similarity
   - Score using NeuMF model + genre/popularity/embedding bonuses
   - Prioritize items matching top inferred genres
   - Return personalized recommendations

**Example Query Processing:**
- Query: "make my blood pumping"
- Action keywords detected: "pumping" → boost exciting
- Affects: exciting (boosted), scary (reduced), dark
- Genres inferred: Action (0.15), Thriller (0.12), Horror (suppressed)
- Results: High-energy action/thriller movies

### API Examples

**Genre-based:**
```bash
curl "http://localhost:8000/recommendations?user_id=1&genre=Action&top_k=10&strict=true"
```

**Intent-based:**
```bash
curl "http://localhost:8000/intent_recommendations?q=exciting%20thrilling&user_id=1&top_k=10&strict=false&genre_alpha=0.35&pop_alpha=0.05&embed_alpha=0.60"
```

**Intent-based with action keywords:**
```bash
curl "http://localhost:8000/intent_recommendations?q=awesome%20blood%20pumping&user_id=1&top_k=10&strict=false"
```

---

## Project Structure

```
NeuMF-Movie-Recommendation-Engine/
├── recsys/              # Core recommendation system
│   ├── data.py         # Data loading and preprocessing (supports 100K & 25M)
│   ├── model.py        # NeuMF model definition with optional intent tower
│   ├── train.py        # Training loop with memory-efficient intent handling
│   └── eval.py         # Evaluation metrics (HR@K, NDCG@K)
│
├── backend/             # FastAPI service
│   ├── main.py         # API endpoints and advanced intent system
│   └── utils.py        # Helper functions for metadata loading
│
├── frontend/            # React UI
│   ├── src/
│   │   ├── App.jsx     # Main application with error handling
│   │   ├── api.js      # API client with interceptors and timeout
│   │   └── components/
│   │       ├── GenreSelector.jsx
│   │       ├── PromptSearch.jsx    # NLP intent UI with alpha tuning
│   │       ├── Recommendations.jsx
│   │       └── StrictToggle.jsx
│   └── package.json
│
├── scripts/             # Utility scripts
│   ├── download_mlwk.py           # Download MovieLens 100K or 25M
│   └── build_item_embeddings.py   # Generate semantic embeddings
│
├── configs/             # Configuration files
│   └── starter.yaml    # Training hyperparameters (loaded by backend)
│
├── checkpoints/         # Saved models and embeddings
│   ├── neumf_final.pt  # Trained model weights
│   └── item_embeddings.npy  # Pre-computed embeddings [num_items, 384]
│
├── data/                # Dataset (not in repository - must download)
│   ├── ml-100k/        # MovieLens 100K data (download via script)
│   │   └── ml-100k/
│   │       ├── u.data      ✓ Essential
│   │       └── u.item      ✓ Essential
│   └── ml-25m/         # MovieLens 25M data (download via script)
│       └── ml-25m/
│           ├── ratings.csv ✓ Essential
│           └── movies.csv  ✓ Essential
│
├── main.py             # CLI entry point for training
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

---

## Evaluation Metrics

The system uses standard recommendation metrics:

### Hit Rate @ K (HR@K)

Fraction of users for whom at least one relevant item appears in top-K recommendations:

**HR@K = (1/|U|) · Σ<sub>u∈U</sub> I(top_K<sub>u</sub> contains test_item<sub>u</sub>)**

Where:
- **U**: Set of all users in test set
- **top_K<sub>u</sub>**: Top K recommended items for user u
- **test_item<sub>u</sub>**: Ground truth test item for user u
- **I(·)**: Indicator function (1 if condition is true, 0 otherwise)

### Normalized Discounted Cumulative Gain @ K (NDCG@K)

Measures ranking quality, giving higher weight to items ranked higher:

**DCG@K = Σ<sub>i=1</sub><sup>K</sup> (2<sup>rel<sub>i</sub></sup> - 1) / log<sub>2</sub>(i + 1)**

Where:
- **rel<sub>i</sub>**: Relevance of item at position i (1 if item matches test item, 0 otherwise)

**IDCG@K**: Ideal DCG (DCG if test item is ranked first)

**NDCG@K = DCG@K / IDCG@K**

NDCG ranges from 0 to 1, where 1 indicates perfect ranking.

---

## Troubleshooting

- **Backend fails to find dataset**: 
  - Check `MOVIELENS_PATH` points to the directory containing either `ml-100k/` (for 100K) or `ml-25m/` (for 25M) subfolder, or `ratings.csv` in the root
  - Ensure the essential files exist:
    - **MovieLens 100K**: `ml-100k/u.data` and `ml-100k/u.item` must be present
    - **MovieLens 25M**: `ml-25m/ratings.csv` and `ml-25m/movies.csv` must be present
  - If you cloned the repo, remember to download the dataset using `scripts/download_mlwk.py` (data files are not in the repository)
- **Backend fails to load model**: Ensure `MODEL_PATH` points to `checkpoints/neumf_final.pt` created after training. Check that hyperparameters in `configs/starter.yaml` match training configuration.
- **NLP intent system not working**: Ensure `item_embeddings.npy` exists (run `build_item_embeddings.py` first). Check that embedding shape matches number of items in dataset.
- **CORS errors**: Confirm backend runs on `http://localhost:8000` and frontend on `http://localhost:5173`
- **Out of memory with 25M dataset**: 
  - Reduce `batch_size` in `configs/starter.yaml`
  - Use `--max-ratings` parameter to sample subset (e.g., `--max-ratings 500000`)
  - Use CPU instead of CUDA if GPU memory is insufficient
- **Training is slow**: MovieLens 25M is much larger - consider using GPU or reducing training epochs for experimentation. Use `--max-ratings` for quick tests.
- **Frontend hangs on startup**: 
  - Wait for backend to fully start ("Application startup complete" message)
  - Check browser console for API errors
  - Verify backend is responding: `curl http://localhost:8000/genres`
  - Check Network tab in browser DevTools for pending requests
- **PyTorch CUDA not available**: PyTorch CUDA builds require Python ≤3.12. Use CPU mode or create a Python 3.12 environment.
- **Intent recommendations skewed wrong**: Adjust alpha parameters in UI or via API. Increase `genre_alpha` for stronger genre matching, increase `embed_alpha` for semantic similarity.

---

## License

For research and educational purposes.

---

## Acknowledgments

- MovieLens dataset: https://grouplens.org/datasets/movielens/
- NeuMF paper: "Neural Collaborative Filtering" by He et al., 2017 (WWW)
- Sentence Transformers: https://www.sbert.net/
- FastAPI: https://fastapi.tiangolo.com/
- React: https://react.dev/

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@misc{neumf-movie-recommendation,
  title={NeuMF Genre-Aware Movie Recommendation Engine},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/NeuMF-Movie-Recommendation-Engine}}
}
```
