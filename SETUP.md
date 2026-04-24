# CTMAS — Setup Guide

Federated, privacy-aware anomaly detection on the SWaT dataset. This guide walks you from a fresh clone to a trained model.

## Requirements

- **Python 3.11** (other 3.x versions may work, 3.11 is what this was developed on)
- **NVIDIA GPU with CUDA** strongly recommended. CPU-only works but training takes hours instead of minutes. Apple Silicon (MPS) also works but is slow.
- **~8 GB RAM**, **~5 GB free disk**
- **`merged.csv`** — the SWaT dataset (~427 MB, not in this repo). Ask the project owner for the link.

## 1. Clone the repo

```bash
git clone https://github.com/YOUR_USER/CTMAS.git
cd CTMAS
```

## 2. Put the dataset in place

Download `merged.csv` and move it to `Data/merged.csv`:

```bash
# macOS/Linux
mv ~/Downloads/merged.csv Data/merged.csv
# Windows (PowerShell)
Move-Item $env:USERPROFILE\Downloads\merged.csv Data\merged.csv
```

## 3. Create and activate a virtual environment

```bash
python3.11 -m venv .venv
```

Activate it:
- **macOS/Linux:** `source .venv/bin/activate`
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Windows (cmd):** `.venv\Scripts\activate.bat`

## 4. Install PyTorch with CUDA **first**

Check your CUDA version:
```bash
nvidia-smi
```
Look for `CUDA Version:` in the top-right. Then install the matching PyTorch build from https://pytorch.org/get-started/locally/.

**CUDA 12.1 (most common):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 11.8:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**CPU-only (no GPU):**
```bash
pip install torch
```

## 5. Install the rest of the dependencies

```bash
pip install -r requirements.txt
```

## 6. Verify GPU is visible

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

Expected output on a CUDA setup:
```
CUDA: True NVIDIA GeForce RTX 3060
```

If it says `False` and you have an NVIDIA GPU, your PyTorch install picked up the CPU build — redo step 4 with the correct CUDA index URL.

## 7. Regenerate processed data

Takes ~2–3 minutes. Produces `X_train / X_val / X_test / y_test / metadata / scaler` in `Data/processed/`.

```bash
cd Data
python Data_Preprocessing.py
cd ..
```

## 8. Train

```bash
python main.py --rounds 10
```

### Options
- `--subsample 0.1` — use 10% of training data for a fast sanity run (~3 min)
- `--rounds N` — override the federated round count
- `--eval-only` — skip training, just evaluate the saved `ctmas_model.pt`

Training produces `ctmas_model.pt` at the repo root (~500 KB).

## 9. Run the API dashboard (optional)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Then open `frontend/index.html` in a browser.

---

## Troubleshooting

**`torch.cuda.is_available()` returns False despite having an NVIDIA GPU**
You installed the CPU wheel. Uninstall and reinstall with the CUDA index URL from step 4:
```bash
pip uninstall -y torch torchvision torchaudio
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**`ModuleValidator` / `GradSampleModule` errors from Opacus**
Version mismatch with torch. Try:
```bash
pip install opacus==1.5.2
```

**`torch-geometric` install fails on Windows**
Install the prebuilt wheels that match your torch + CUDA versions:
```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cu121.html
pip install torch-geometric
```

**Out-of-memory on a small GPU**
Reduce batch size inside `federated/config.py`, or use `--subsample 0.3` to train on less data.

**Preprocessing errors about missing `merged.csv`**
Check that the file is at `Data/merged.csv` (case-sensitive on Linux/Mac) and is the unzipped CSV, not a zip archive.
