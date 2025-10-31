# NeuMF Genre‑Aware Recommender (Movies + Music)

Genre‑aware Neural Matrix Factorization (NeuMF) recommender built with PyTorch and FastAPI, paired with a React (Vite) frontend. Ships with MovieLens 100K as a starter dataset and is easy to extend.

---

## Prerequisites
- Python 3.9+ (3.10+ recommended)
- Node.js 18+ and npm
- Optional: CUDA‑enabled GPU for faster training/inference

---

## Project Structure
- `recsys/`: Core library for data loading, model, training, evaluation
- `backend/`: FastAPI service to serve recommendations
- `frontend/`: React UI (Vite) to browse genres and get recs
- `scripts/`: Utilities (e.g., MovieLens download)
- `configs/`: Training configuration(s)
- `checkpoints/`: Saved model weights (created after training)
- `main.py`: CLI to train and evaluate

---

## 1) Setup & Data

Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

Install Python deps (training + backend):
```bash
pip install -r requirements.txt -r backend/requirements.txt
```

Download MovieLens 100K:
```bash
python scripts/download_mlwk.py --target ./data/ml-100k
```

Expected layout after download:
```
data/ml-100k/ml-100k/u.data
data/ml-100k/ml-100k/u.item
... (other MovieLens files)
```

---

## 2) Train the Model

Train with defaults (config in `configs/starter.yaml`):
```bash
python main.py --data ./data/ml-100k --epochs 10
```

What this does:
- Loads data and builds the NeuMF model
- Trains for the specified epochs
- Evaluates HR/NDCG on the test set
- Saves weights to `./checkpoints/neumf_final.pt`

Common flags:
- `--data`: path that contains `ml-100k` folder inside
- `--config`: override hyperparams via YAML (defaults to `configs/starter.yaml`)
- `--epochs`: override number of epochs from config
- `--device`: `cpu` or `cuda` (auto‑detected by default)

---

## 3) Run the Backend API (FastAPI)

The backend loads:
- Dataset from `MOVIELENS_PATH` (defaults to `project_root/data/ml-100k`)
- Trained model from `MODEL_PATH` (defaults to `project_root/checkpoints/neumf_final.pt`)

Optionally build semantic embeddings for NLP intent search (improves free‑text prompts):
```bash
python scripts/build_item_embeddings.py --data ./data/ml-100k --out ./checkpoints/item_embeddings.npy
```

Start the API (dev, auto‑reload):
```bash
uvicorn backend.main:app --reload --port 8000
```

If your paths are custom, set environment variables first.

Windows PowerShell:
```powershell
$env:MOVIELENS_PATH = "C:\\path\\to\\data\\ml-100k"
$env:MODEL_PATH = "C:\\path\\to\\checkpoints\\neumf_final.pt"
$env:EMB_PATH = "C:\\path\\to\\checkpoints\\item_embeddings.npy"  # optional
$env:EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"          # optional
uvicorn backend.main:app --reload --port 8000
```

macOS/Linux:
```bash
export MOVIELENS_PATH=./data/ml-100k
export MODEL_PATH=./checkpoints/neumf_final.pt
export EMB_PATH=./checkpoints/item_embeddings.npy   # optional
export EMB_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2   # optional
uvicorn backend.main:app --reload --port 8000
```

API endpoints:
- `GET /genres` → list of available genres
- `GET /users` → list of user IDs
- `GET /recommendations?user_id=<id>&genre=<name>&top_k=<n>&strict=<true|false>` → ranked items
- `GET /intent_recommendations?q=<prompt>&user_id=<id>&top_k=<n>&strict=<bool>&genre_alpha=<f>&pop_alpha=<f>&embed_alpha=<f>` → free‑text prompt to recommendations (uses embeddings if available)

Notes:
- On first start, the API builds the dataset and loads the model into memory.
- CPU is supported; GPU will be used automatically if available.

---

## 4) Run the Frontend (React + Vite)

Install dependencies and launch the dev server:
```bash
cd frontend
npm install
npm run dev
```

Open the URL shown by Vite (typically `http://localhost:5173`). The app calls the backend at `http://localhost:8000` (see `frontend/src/api.js`). Ensure the backend is running.

Free‑text prompt search: use the new text box in the UI (“Or describe what you want to watch”). You can tune `Genre α`, `Popularity α`, and `Embedding α` live.

Build for production:
```bash
npm run build
npm run preview
```

---

## Troubleshooting
- Backend fails to find dataset: check `MOVIELENS_PATH` points to the directory that contains the `ml-100k` subfolder.
- Backend fails to load model: ensure `MODEL_PATH` points to `checkpoints/neumf_final.pt` created after training.
- CORS or network errors in the frontend: confirm backend is on `http://localhost:8000` and frontend on `http://localhost:5173`.
- Windows PowerShell env vars: use `$env:VAR = "value"` syntax as shown above.

---

## License
For research and educational purposes.
