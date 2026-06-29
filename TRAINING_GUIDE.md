# Model Training Guide — Route Resilience Analyzer
### ISRO Bharatiya Antariksh Hackathon 2026

---

## Your Hardware
| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop |
| VRAM | 4 GB |
| CUDA Version | 12.7 |
| OS | Windows 11 |

---

## Total Time Required

| Stage | Time Estimate |
|---|---|
| Step 1 — Install dependencies | 5–10 min |
| Step 2 — Download Indian road data | 30–60 min (internet speed) |
| Step 3 — Train the model | **2–3 hours** |
| Step 4 — Test in app | 5 min |
| **TOTAL** | **~4 hours** |

> Your RTX 3050 has only 4 GB VRAM. The guide uses batch size 2 to fit within this limit.
> If you get a CUDA out-of-memory error, drop batch size to 1 in `train/train.py`.

---

## Step 1 — Install Dependencies

Open PowerShell in the `isro` folder and run:

```powershell
& "C:\Users\KEVINDEEP SINGH\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts\pip.exe" install osmnx mercantile albumentations tqdm segmentation-models-pytorch
```

Verify everything is installed:

```powershell
& "C:\Users\KEVINDEEP SINGH\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts\pip.exe" show osmnx albumentations segmentation-models-pytorch
```

All three should show version numbers. If any are missing, install them individually.

---

## Step 2 — Download Indian Road Training Data

This script downloads satellite tiles + road masks for **12 Indian cities**:
Mumbai, Delhi, Bengaluru, Chennai, Hyderabad, Pune, Ahmedabad,
Jaipur, Kolkata, Lucknow, Gandhinagar, Bhopal.

```powershell
cd "C:\Users\KEVINDEEP SINGH\OneDrive\Desktop\isro"
python train/prepare_indian_data.py
```

### What it does
1. Connects to OpenStreetMap and downloads road geometry for each city
2. Fetches matching satellite image tiles (256×256 px each, 2×2 grid = 512×512)
3. Rasterizes road vectors onto each tile as a binary mask (white = road, black = background)
4. Saves pairs to:
   - `data/indian_roads/images/` — satellite images
   - `data/indian_roads/masks/`  — binary road masks

### Expected output
```
[Mumbai] Downloaded 45 tile pairs
[Delhi] Downloaded 52 tile pairs
...
Total: ~500 image-mask pairs
```

### If it fails / times out
OSM tile servers sometimes rate-limit. If you see errors:
```powershell
# Wait 2 minutes then re-run — it skips already-downloaded tiles
python train/prepare_indian_data.py
```

---

## Step 3 — Train the Model

Before training, open `train/train.py` and verify these settings match your GPU:

```python
BATCH_SIZE  = 2      # 4 GB VRAM → keep at 2. Drop to 1 if CUDA OOM error.
NUM_EPOCHS  = 40     # 40 epochs gives good convergence on ~500 images
LEARNING_RATE = 3e-4
IMG_SIZE    = 512
```

Then run:

```powershell
cd "C:\Users\KEVINDEEP SINGH\OneDrive\Desktop\isro"
python train/train.py
```

### What you will see
```
Epoch  1/40  |  Loss: 0.6821  |  IoU: 0.312  |  [saved]
Epoch  2/40  |  Loss: 0.5934  |  IoU: 0.401  |  [saved]
Epoch  5/40  |  Loss: 0.4102  |  IoU: 0.551
...
Epoch 20/40  |  Loss: 0.2341  |  IoU: 0.712
Epoch 40/40  |  Loss: 0.1876  |  IoU: 0.758  |  [saved]

Best model saved to: models/road_seg.pth
```

### What good IoU looks like
| IoU Score | Road Detection Quality |
|---|---|
| Below 0.40 | Poor — barely detects roads |
| 0.40 – 0.60 | Fair — main roads visible, gaps in smaller roads |
| 0.60 – 0.75 | Good — most roads detected |
| Above 0.75 | Excellent — fine details, intersections clear |

> **Target: IoU > 0.65** — achievable on your GPU with 40 epochs.

### Time per epoch on RTX 3050 (estimated)
- Batch size 2, 512×512, ~500 images → ~250 iterations/epoch
- Each iteration: ~0.4 seconds on RTX 3050
- 1 epoch ≈ 2 minutes
- 40 epochs ≈ **80–100 minutes**

### If you get CUDA Out of Memory
```python
# In train/train.py, reduce:
BATCH_SIZE = 1       # halves VRAM usage
IMG_SIZE   = 384     # reduces VRAM further if needed
```

### Monitor GPU usage while training
Open a second PowerShell and run:
```powershell
nvidia-smi -l 2
```
You should see GPU utilization at 80–100% and memory at 3–4 GB.

---

## Step 4 — Use Trained Weights in the App

1. Confirm the model was saved:
```powershell
ls "C:\Users\KEVINDEEP SINGH\OneDrive\Desktop\isro\models\road_seg.pth"
```

2. Open the app (`run.bat`)

3. In the sidebar:
   - Check **"Use fine-tuned weights (optional)"**
   - Path field auto-fills: `models/road_seg.pth`

4. Upload your satellite image and click **Run Full Pipeline**

The sidebar will now show:
```
Mode: DeepLabV3+ (fine-tuned weights)
```

---

## Expected Improvement After Training

| Metric | Before Training (Heuristic) | After Training (DL) |
|---|---|---|
| Road detection rate | ~50–60% | ~75–85% |
| Roads under trees | Missed | Partially detected |
| Laterite/mud roads | Missed | Detected |
| False positives (buildings) | Moderate | Low |
| Intersections found | Few | Many more |

---

## Optional — Improve Further

### More training data (better accuracy)
Edit `train/prepare_indian_data.py` and add more cities:
```python
CITIES = [
    "Mumbai, India",
    "Delhi, India",
    # add any city name here
    "Varanasi, India",
    "Amritsar, India",
    "Surat, India",
]
```

### More epochs (better convergence)
```python
NUM_EPOCHS = 60   # adds ~40 more minutes but IoU improves by ~3–5%
```

### Resume training from checkpoint
If training was interrupted, the best checkpoint is already saved at `models/road_seg.pth`.
The next run will start fresh — to resume, you would need to save optimizer state
(advanced — not required for the hackathon).

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: osmnx` | Run Step 1 pip install again |
| `CUDA out of memory` | Set `BATCH_SIZE = 1` in train.py |
| `No tile pairs downloaded` | OSM rate limit — wait 5 min and retry |
| `IoU stuck at 0.0` | Check masks folder is not empty: `ls data/indian_roads/masks/` |
| App shows blank Road Mask | You forgot to check "Use fine-tuned weights" in sidebar |
| `FileNotFoundError: models/road_seg.pth` | Training did not complete — re-run train.py |

---

## Quick Reference Commands

```powershell
# 1. Download data
python train/prepare_indian_data.py

# 2. Train
python train/train.py

# 3. Check model saved
ls models/road_seg.pth

# 4. Launch app
.\run.bat
```

---

*Built for ISRO Bharatiya Antariksh Hackathon 2026 — Problem Statement 4*
