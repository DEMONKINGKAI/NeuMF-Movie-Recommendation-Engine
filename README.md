# NeuMF Genre-Aware Movie Recommendation Engine

A comprehensive Neural Matrix Factorization (NeuMF) recommender system that provides personalized movie recommendations using genre-aware collaborative filtering and natural language intent understanding. Built with PyTorch for training, FastAPI for serving, and React for the frontend interface.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Architecture](#architecture)
4. [Mathematical Foundations](#mathematical-foundations)
5. [NLP Intent System](#nlp-intent-system)
6. [System Workflow](#system-workflow)
7. [Installation & Setup](#installation--setup)
8. [Usage](#usage)
9. [Project Structure](#project-structure)

---

## Project Overview

This project implements a state-of-the-art recommendation system that combines:

1. **Neural Matrix Factorization (NeuMF)**: A hybrid deep learning model that fuses Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP) architectures to learn user-item interactions
2. **Genre-Aware Filtering**: Incorporates movie genre information as multi-hot vectors to enhance recommendation quality
3. **NLP Intent Understanding**: Uses semantic embeddings to interpret free-form user queries (e.g., "I want something exciting and thrilling") and map them to appropriate movie recommendations
4. **Affect Detection**: Automatically detects emotional intents (sad, funny, scary, romantic, etc.) from user prompts and adjusts recommendations accordingly

The system is trained on the MovieLens 100K dataset and provides both genre-based and intent-based recommendation endpoints through a RESTful API, accessible via a modern React frontend.

---

## Technologies Used

### Backend & Training
- **Python 3.9+**: Core programming language
- **PyTorch**: Deep learning framework for model training and inference
- **FastAPI**: High-performance web framework for the recommendation API
- **sentence-transformers**: Semantic text embeddings for NLP intent processing
- **NumPy & Pandas**: Data processing and manipulation
- **Uvicorn**: ASGI server for FastAPI

### Frontend
- **React**: UI framework
- **Vite**: Modern build tool and dev server
- **Anime.js**: Smooth animations for UI interactions

### Data
- **MovieLens 100K**: Movie rating dataset with 100,000 ratings from 943 users on 1,682 movies

---

## Architecture

The system consists of three main components:

### 1. Training Pipeline (`recsys/`)
- **Data Loading** (`data.py`): Loads MovieLens 100K, creates train/val/test splits using leave-one-out methodology
- **Model Definition** (`model.py`): Implements the NeuMF architecture
- **Training** (`train.py`): Trains the model using binary cross-entropy loss with negative sampling
- **Evaluation** (`eval.py`): Evaluates model performance using Hit Rate (HR@K) and Normalized Discounted Cumulative Gain (NDCG@K)

### 2. Backend API (`backend/`)
- **FastAPI Service** (`main.py`): Serves recommendation endpoints
  - `/recommendations`: Genre-based recommendations
  - `/intent_recommendations`: NLP-based intent recommendations
  - `/genres`: List available genres
  - `/users`: List available user IDs
- **Embedding System**: Loads pre-computed item embeddings for semantic search
- **Intent Mapping**: Maps natural language queries to genre weights and affects

### 3. Frontend UI (`frontend/`)
- **React Application**: Interactive interface for:
  - Selecting users and genres
  - Free-text prompt search with tunable alpha parameters
  - Displaying recommendations with scores

---

## Mathematical Foundations

### Generalized Matrix Factorization (GMF)

GMF captures linear user-item interactions through elementwise product of embeddings:

Given user embedding **p<sub>u</sub>** ∈ ℝ<sup>k</sup> and item embedding **q<sub>i</sub>** ∈ ℝ<sup>k</sup> (where k is the embedding dimension), the GMF output is:

**h<sub>GMF</sub> = p<sub>u</sub> ⊙ q<sub>i</sub>**

Where ⊙ denotes elementwise (Hadamard) product:

**h<sub>GMF</sub>[j] = p<sub>u</sub>[j] × q<sub>i</sub>[j]** for j = 1, ..., k

This captures multiplicative interactions between user and item latent factors, similar to matrix factorization but in a neural framework.

### Neural Matrix Factorization (NeuMF)

NeuMF combines GMF with a Multi-Layer Perceptron (MLP) to capture both linear and non-linear interactions:

#### 1. GMF Component:
```
GMF Output: h_GMF = user_gmf_emb(u) ⊙ item_gmf_emb(i)
```

#### 2. MLP Component:
The MLP processes concatenated user, item, genre, and optional intent embeddings:

```
MLP Input: x_MLP = [user_mlp_emb(u) || item_mlp_emb(i) || genre_proj(g) || intent_tower(v)]
```

Where:
- `user_mlp_emb(u)`: User embedding in MLP space (dimension: d_mlp)
- `item_mlp_emb(i)`: Item embedding in MLP space (dimension: d_mlp)
- `genre_proj(g)`: Projected genre vector (dimension: genre_proj_dim)
  - **g** is a multi-hot genre vector (e.g., [0,1,0,1,0,...] for Action + Adventure)
  - **genre_proj(g) = W<sub>genre</sub> · g + b**, where W<sub>genre</sub> ∈ ℝ<sup>19×genre_proj_dim</sup>
- `intent_tower(v)`: Optional intent embedding processed through a neural network
  - If present: **intent_tower(v) = ReLU(W<sub>2</sub> · Dropout(ReLU(W<sub>1</sub> · v + b<sub>1</sub>)) + b<sub>2</sub>)**

The MLP then applies multiple fully-connected layers:

```
h_1 = ReLU(W_1 · x_MLP + b_1)
h_2 = ReLU(W_2 · h_1 + b_2)
...
h_MLP = h_L  (final MLP layer output)
```

#### 3. NeuMF Final Prediction:

The model concatenates GMF and MLP outputs and applies a final linear layer:

```
Final Input: z = [h_GMF || h_MLP]
Final Score: ŷ = W_final · z + b_final
Predicted Probability: p(interaction | u, i, g, v) = σ(ŷ) = 1 / (1 + exp(-ŷ))
```

Where σ is the sigmoid function.

#### 4. Training Objective:

The model is trained using Binary Cross-Entropy Loss with negative sampling:

**L = -[y · log(σ(ŷ)) + (1-y) · log(1-σ(ŷ))]**

Where:
- **y = 1** for positive user-item interactions
- **y = 0** for negative samples (randomly sampled non-interacted items)

---

## NLP Intent System

The NLP intent system enables users to express preferences in natural language (e.g., "I want something exciting and funny") and maps these queries to personalized recommendations. The system operates in several stages:

### 1. Text Embedding

User queries are converted to dense vectors using a sentence transformer model:

**q = SentenceTransformer(text)**  ∈ ℝ<sup>384</sup> (for `all-MiniLM-L6-v2`)

The embedding is L2-normalized: **q = q / ||q||<sub>2</sub>**

### 2. Genre Centroid Computation

For each genre, a centroid is computed as the mean embedding of all movies in that genre:

For genre **g**, with items **I<sub>g</sub>**:

**c<sub>g</sub> = (1/|I<sub>g</sub>|) · Σ<sub>i∈I<sub>g</sub></sub> e<sub>i</sub>**

Where **e<sub>i</sub>** is the pre-computed embedding for item i (computed from "title + genres" text).

The centroid is normalized: **c<sub>g</sub> = c<sub>g</sub> / ||c<sub>g</sub>||<sub>2</sub>**

### 3. Query Steering

The query vector is "steered" toward relevant genre centroids and away from irrelevant ones:

**sims = C · q**  (where C is the genre centroid matrix)

**top_genres = argmax_k(sims)**  (top K most similar genres)

**bot_genres = argmin_k(sims)**  (bottom K least similar genres)

**q_steered = q + α<sub>pos</sub> · mean(c<sub>top</sub>) - α<sub>neg</sub> · mean(c<sub>bot</sub>)**

**q_steered = q_steered / ||q_steered||<sub>2</sub>**

This sharpens the query vector to better match user intent.

### 4. Affect Detection

The system detects emotional intents using predefined affect anchors:

For each affect **a** (e.g., "sad", "funny", "scary") with anchor phrase **p<sub>a</sub>**:

**affect_score(a) = cosine_similarity(q_steered, embed(p<sub>a</sub>))**

**top_affect = argmax(affect_score)**

Each affect has genre priors: **P(genre | affect)**, e.g.:
- "funny" → Comedy: 0.9
- "scary" → Horror: 0.7, Thriller: 0.3
- "romantic" → Romance: 0.7, Drama: 0.3

### 5. Genre Weight Computation

Genre weights are computed by combining:
1. **Centroid Similarities**: **sim<sub>g</sub> = c<sub>g</sub><sup>T</sup> · q_steered**
2. **Affect Priors**: **prior<sub>g</sub> = Σ<sub>a</sub> P(g|a) · affect_score(a)**

**combined<sub>g</sub> = sim<sub>g</sub> + β · prior<sub>g</sub>**

Where β is tuned based on affect confidence (β = 0.9 for high confidence, 0.7 otherwise).

Genre weights are then normalized and thresholded:

**w<sub>g</sub> = combined<sub>g</sub> / Σ<sub>g'</sub> combined<sub>g'</sub>**, if combined<sub>g</sub> ≥ threshold

### 6. Candidate Retrieval & Scoring

Candidates are retrieved using embedding similarity:

**item_sims = E · q_steered**  (where E is the item embedding matrix)

**candidates = top_k_by_similarity(item_sims)**

Final scores combine multiple signals:

**final_score = base_score + α<sub>genre</sub> · genre_bonus + α<sub>pop</sub> · pop_bonus + α<sub>embed</sub> · embed_bonus**

Where:
- **base_score**: NeuMF model prediction σ(ŷ)
- **genre_bonus**: **Σ<sub>g</sub> w<sub>g</sub> · g<sub>item</sub>[g]** (dot product of genre weights and item genre vector)
- **pop_bonus**: Normalized popularity (interaction count) of the item
- **embed_bonus**: Cosine similarity between item embedding and query embedding

The alpha parameters (α<sub>genre</sub>, α<sub>pop</sub>, α<sub>embed</sub>) are tunable via the API and UI, allowing users to control the balance between model predictions, genre preferences, popularity, and semantic matching.

---

## System Workflow

### End-to-End Flow

```
1. Data Preparation
   └─> Load MovieLens 100K dataset
   └─> Create user/item ID mappings
   └─> Build genre matrix (multi-hot vectors)
   └─> Split into train/val/test (leave-one-out)
   └─> Generate negative samples for training

2. Embedding Generation (Optional)
   └─> Build text descriptions: "Movie Title. Genres: Action, Adventure."
   └─> Encode using SentenceTransformer
   └─> Save to item_embeddings.npy

3. Model Training
   └─> Initialize NeuMF model (GMF + MLP + Genre + Intent towers)
   └─> For each epoch:
       ├─> Sample positive interactions
       ├─> Sample negative interactions (4 per positive)
       ├─> Forward pass: compute predictions
       ├─> Compute BCE loss
       └─> Backpropagate and update weights
   └─> Evaluate on test set (HR@10, NDCG@10)
   └─> Save model checkpoint

4. API Startup
   └─> Load dataset and metadata
   └─> Load trained model weights
   └─> Load item embeddings (if available)
   └─> Compute genre centroids from embeddings
   └─> Pre-compute popularity scores
   └─> Start FastAPI server

5. Recommendation Request (Genre-based)
   └─> Receive: user_id, genre, top_k, strict
   └─> Filter candidates by genre (strict or soft)
   └─> Run NeuMF model on candidates
   └─> Rank by prediction scores
   └─> Return top_k movies

6. Recommendation Request (Intent-based)
   └─> Receive: query text, user_id, top_k, alphas
   └─> Embed query text → q
   └─> Detect affects from q
   └─> Compute genre weights (centroid + affect priors)
   └─> Steer query vector
   └─> Retrieve candidates via embedding similarity
   └─> Run NeuMF model with intent vector
   └─> Combine scores: base + genre + pop + embed
   └─> Return top_k movies
```

---

## Installation & Setup

### Prerequisites
- Python 3.9+ (3.10+ recommended)
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

```bash
python scripts/download_mlwk.py --target ./data/ml-100k
```

Expected structure:
```
data/ml-100k/ml-100k/
  ├─ u.data          (ratings)
  ├─ u.item          (movie metadata)
  ├─ u.user          (user metadata)
  └─ ... (other files)
```

### 4. Train the Model

```bash
python main.py --data ./data/ml-100k --epochs 10
```

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

```bash
python scripts/build_item_embeddings.py --data ./data/ml-100k --out ./checkpoints/item_embeddings.npy
```

This creates semantic embeddings for each movie (title + genres) using sentence-transformers, enabling the NLP intent recommendation feature.

### 6. Start the Backend API

**Windows PowerShell:**
```powershell
$env:MOVIELENS_PATH = ".\data\ml-100k"
$env:MODEL_PATH = ".\checkpoints\neumf_final.pt"
$env:EMB_PATH = ".\checkpoints\item_embeddings.npy"  # optional
$env:EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # optional
uvicorn backend.main:app --reload --port 8000
```

**macOS/Linux:**
```bash
export MOVIELENS_PATH=./data/ml-100k
export MODEL_PATH=./checkpoints/neumf_final.pt
export EMB_PATH=./checkpoints/item_embeddings.npy  # optional
export EMB_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2  # optional
uvicorn backend.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

**API Endpoints:**
- `GET /genres` → List of available genres
- `GET /users` → List of user IDs
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

### NLP Intent-Based Recommendations

1. Enter a free-form text query in the prompt box, e.g.:
   - "I want something exciting and thrilling"
   - "show me a sad romantic movie"
   - "something funny for the family"
2. Adjust the alpha parameters:
   - **Genre α**: Weight for genre matching (0.0-1.0)
   - **Popularity α**: Weight for popular movies (0.0-1.0)
   - **Embedding α**: Weight for semantic similarity (0.0-1.0)
3. Click "Recommend"
4. The system will:
   - Parse your query semantically
   - Detect emotional affects (sad, funny, scary, etc.)
   - Infer genre preferences
   - Retrieve candidates via embedding similarity
   - Score using NeuMF model + genre/popularity/embedding bonuses
   - Return personalized recommendations

### API Examples

**Genre-based:**
```bash
curl "http://localhost:8000/recommendations?user_id=1&genre=Action&top_k=10&strict=true"
```

**Intent-based:**
```bash
curl "http://localhost:8000/intent_recommendations?q=exciting%20thrilling&user_id=1&top_k=10&strict=false&genre_alpha=0.35&pop_alpha=0.05&embed_alpha=0.60"
```

---

## Project Structure

```
NeuMF-Movie-Recommendation-Engine/
├── recsys/              # Core recommendation system
│   ├── data.py         # Data loading and preprocessing
│   ├── model.py        # NeuMF model definition
│   ├── train.py        # Training loop
│   └── eval.py         # Evaluation metrics (HR@K, NDCG@K)
│
├── backend/             # FastAPI service
│   ├── main.py         # API endpoints and intent system
│   └── utils.py        # Helper functions
│
├── frontend/            # React UI
│   ├── src/
│   │   ├── App.jsx     # Main application
│   │   ├── api.js      # API client
│   │   └── components/
│   │       ├── GenreSelector.jsx
│   │       ├── PromptSearch.jsx    # NLP intent UI
│   │       ├── Recommendations.jsx
│   │       └── StrictToggle.jsx
│   └── package.json
│
├── scripts/             # Utility scripts
│   ├── download_mlwk.py           # Download MovieLens dataset
│   └── build_item_embeddings.py   # Generate semantic embeddings
│
├── configs/             # Configuration files
│   └── starter.yaml    # Training hyperparameters
│
├── checkpoints/         # Saved models and embeddings
│   ├── neumf_final.pt  # Trained model weights
│   └── item_embeddings.npy  # Pre-computed embeddings
│
├── data/                # Dataset
│   └── ml-100k/        # MovieLens 100K data
│
├── main.py             # CLI entry point for training
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

---

## Evaluation Metrics

The system uses standard recommendation metrics:

- **Hit Rate @ K (HR@K)**: Fraction of users for whom at least one relevant item appears in top-K recommendations
  - HR@K = (1/|U|) · Σ<sub>u∈U</sub> I(top_K<sub>u</sub> contains test_item<sub>u</sub>)

- **Normalized Discounted Cumulative Gain @ K (NDCG@K)**: Measures ranking quality, giving higher weight to items ranked higher
  - DCG@K = Σ<sub>i=1</sub><sup>K</sup> (2<sup>rel<sub>i</sub></sup> - 1) / log<sub>2</sub>(i + 1)
  - NDCG@K = DCG@K / IDCG@K (where IDCG is the ideal DCG)

---

## Troubleshooting

- **Backend fails to find dataset**: Check `MOVIELENS_PATH` points to the directory containing the `ml-100k` subfolder
- **Backend fails to load model**: Ensure `MODEL_PATH` points to `checkpoints/neumf_final.pt` created after training
- **NLP intent system not working**: Ensure `item_embeddings.npy` exists (run `build_item_embeddings.py` first)
- **CORS errors**: Confirm backend runs on `http://localhost:8000` and frontend on `http://localhost:5173`
- **Out of memory**: Reduce `batch_size` in config or use CPU instead of CUDA

---

## License

For research and educational purposes.

---

## Acknowledgments

- MovieLens dataset: https://grouplens.org/datasets/movielens/
- NeuMF paper: "Neural Collaborative Filtering" by He et al., 2017
- Sentence Transformers: https://www.sbert.net/
