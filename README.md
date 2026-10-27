# Chest X-Ray Multi-Disease Classifier

A deep learning pipeline that predicts 14 thoracic conditions from a single chest X-ray, trained on the NIH ChestX-ray14 dataset. It covers the full path from data preparation and training to a REST API and a web UI.

> **Disclaimer:** This is an educational project. It is **not** a medical device and must not be used for diagnosis or any clinical decision.

![UI screenshot](docs/ui_screenshot.png)
<!-- Add a screenshot of the Streamlit app at docs/ui_screenshot.png -->

---

## What it does

- Takes a chest X-ray image (PNG/JPG).
- Returns a score for each of 14 conditions. This is a **multi-label** problem, so one image can have several conditions at once, or none.
- Serves predictions through a FastAPI endpoint and a Streamlit upload page.

## Results

Evaluated on the official NIH test split (25,596 images, 2,797 patients). The test set was never used for training or model selection.

| Metric | Value |
|---|---|
| Test macro-AUROC | **0.7985** |
| Test loss (weighted BCE) | 1.5958 |

| Disease | Test AUROC |
|---|---|
| Hernia | 0.9198 |
| Emphysema | 0.9063 |
| Cardiomegaly | 0.8841 |
| Edema | 0.8395 |
| Pneumothorax | 0.8362 |
| Effusion | 0.8148 |
| Mass | 0.8075 |
| Fibrosis | 0.8064 |
| Pleural_Thickening | 0.7602 |
| Atelectasis | 0.7560 |
| Nodule | 0.7407 |
| Consolidation | 0.7314 |
| Pneumonia | 0.6963 |
| Infiltration | 0.6800 |

**Reading the results**

- Hernia scores highest despite being the rarest label (about 0.2% of images). It has a visually distinctive pattern, so rarity did not hurt as much as expected.
- Infiltration and Pneumonia score lowest despite Infiltration being the most common label. Both look like diffuse haze and are easily confused with each other and with normal lung markings. The dataset labels were extracted from radiology reports by NLP rather than verified image by image, and this noise is concentrated in these classes.
- Hernia's score rests on only a few dozen positive test images, so treat it as less reliable than the headline number suggests.

## Approach

| Component | Choice |
|---|---|
| Backbone | DenseNet-121, ImageNet-pretrained, final layer replaced with 14 outputs |
| Loss | `BCEWithLogitsLoss` with per-class `pos_weight` computed from the training set |
| Optimizer | AdamW, lr 1e-4, `ReduceLROnPlateau` on validation macro-AUROC |
| Input | 224x224, ImageNet normalization |
| Augmentation | Small rotation and brightness/contrast jitter. No horizontal flip, because left/right anatomy matters |
| Metric | Per-class and macro AUROC (accuracy is misleading under heavy class imbalance) |
| Model selection | Checkpoint with the best validation macro-AUROC |

**Splits.** The official NIH patient-level train/validation list was split into train and validation by patient ID, so no patient appears in more than one set.

| Split | Images | Patients |
|---|---|---|
| Train | 77,724 | 25,208 |
| Validation | 8,800 | 2,800 |
| Test | 25,596 | 2,797 |

Training was stopped manually at about epoch 13 of a planned 20; the best-validation checkpoint was kept.

## Project structure

```
chest-xray-classifier/
â”œâ”€â”€ README.md
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ api/
â”‚   â””â”€â”€ main.py              # FastAPI service (/health, /predict)
â”œâ”€â”€ ui/
â”‚   â””â”€â”€ app.py               # Streamlit front end
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ data/                # Dataset, transforms, dataloaders
â”‚   â”œâ”€â”€ models/              # Model definition
â”‚   â”œâ”€â”€ training/            # Training loop, loss
â”‚   â”œâ”€â”€ inference/           # ChestXrayPredictor
â”‚   â””â”€â”€ utils/               # Metrics
â”œâ”€â”€ scripts/                 # Entry points (make_splits, train, evaluate, ...)
â”œâ”€â”€ config/
â”œâ”€â”€ tests/
â”œâ”€â”€ data/                    # Not tracked (raw images and split CSVs)
â””â”€â”€ checkpoints/             # Not tracked (model weights)
```

## Setup

Tested on Windows 11 with Python 3.11 and an NVIDIA RTX 4060 Laptop GPU.

```
conda create -n xray311 python=3.11 -y
conda activate xray311

# Install PyTorch first, matching the CUDA build
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

Verify the GPU is visible:

```
python -c "import torch; print(torch.cuda.is_available())"
```

## Data

1. Download NIH ChestX-ray14 from Kaggle (`nih-chest-xrays/data`) with the Kaggle CLI.
2. Extract it so that `data/raw/` contains `Data_Entry_2017.csv`, `train_val_list.txt`, `test_list.txt` and the `images_*` folders.
3. Build the split files:

```
python -m scripts.make_splits
```

## Usage

Run everything from the project root.

**Train**
```
python -m scripts.train
```
Saves the best checkpoint to `checkpoints/best_model.pt`.

**Evaluate on the test set**
```
python -m scripts.evaluate
```

**Run the API** (Terminal 1)
```
uvicorn api.main:app --reload
```
Interactive docs: http://127.0.0.1:8000/docs

**Run the UI** (Terminal 2)
```
streamlit run ui/app.py
```
Opens at http://localhost:8501.

**API example**
```
curl.exe -X POST -F "file=@path/to/xray.png" http://127.0.0.1:8000/predict
```

## Limitations

- **Not a clinical tool.** Scores are relative model confidences, not calibrated probabilities. The class-weighted loss deliberately inflates scores for rare diseases, so a score of 0.8 does not mean an 80% chance.
- **Label noise.** Labels come from automated report parsing, so the "ground truth" is imperfect, especially for Infiltration and Pneumonia.
- **No real input validation.** The model always returns 14 scores, even for images that are not chest X-rays. The API only applies a basic grayscale check, which will not catch black-and-white photos or X-rays of other body parts.
- **Single-dataset training.** Performance on images from other hospitals, scanners or populations is untested and is likely to be lower.
- **Limited training.** One training run, stopped around epoch 13, with no hyperparameter search.

## Possible next steps

- A gatekeeper model that rejects non-chest-X-ray inputs.
- Grad-CAM heatmaps to show which regions drive each prediction.
- Experiment tracking and a backbone/hyperparameter comparison.
- Probability calibration and per-class thresholds.
- Dockerfile for deployment.

## Data and citation

Dataset: NIH ChestX-ray14. Please follow the dataset's terms of use.

> Wang X., Peng Y., Lu L., Lu Z., Bagheri M., Summers R. "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases." CVPR 2017.
