import os
import sys
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from tqdm import tqdm

# ======================================================
# DEVICE
# ======================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



# ======================================================
# CONFIG
# ======================================================
CSV_PATH = "ptbxl_scalograms_with_age.csv"   
BASE_DIR = "scalograms_1"                 # cartella base scalogrammi

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
NUM_WORKERS = 4

# ======================================================
# DATASET
# ======================================================
class PTBXLScalogramAgeDataset(Dataset):
    def __init__(self, df, transform=None, base_dir="scalograms_1"):
        self.df = df.reset_index(drop=True)
        self.img_cols = df.columns[:-1]   
        self.ages = df.iloc[:, -1].values # età (float)
        self.transform = transform
        self.base_dir = base_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        channels = []
        for col in self.img_cols:
            img_path = os.path.join(self.base_dir, row[col])
            img = Image.open(img_path).convert("L")

            if self.transform:
                img = self.transform(img)

            channels.append(img)

        # (15, H, W)
        x = torch.cat(channels, dim=0)

        # età come float
        y = torch.tensor(float(self.ages[idx]), dtype=torch.float32)

        return x, y

# ======================================================
# MAIN
# ======================================================
def main():

    # -------------------------
    # LOAD CSV
    # -------------------------
    df = pd.read_csv(CSV_PATH)
    print("[INFO] Dataset caricato:", df.shape)

    # -------------------------
    # SPLIT TRAIN / VAL / TEST
    # -------------------------
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42
    )

    # -------------------------
    # NORMALIZZAZIONE ETÀ (solo train)
    # -------------------------
    y_mean = train_df.iloc[:, -1].mean()
    y_std  = train_df.iloc[:, -1].std()

    for d in [train_df, val_df, test_df]:
        d.iloc[:, -1] = (d.iloc[:, -1] - y_mean) / y_std

    print(f"[INFO] Età normalizzate | mean={y_mean:.2f}, std={y_std:.2f}")

    # -------------------------
    # TRANSFORMS
    # -------------------------
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),            # (1, H, W)
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # -------------------------
    # DATASETS
    # -------------------------
    train_ds = PTBXLScalogramAgeDataset(train_df, transform, BASE_DIR)
    val_ds   = PTBXLScalogramAgeDataset(val_df, transform, BASE_DIR)
    test_ds  = PTBXLScalogramAgeDataset(test_df, transform, BASE_DIR)

    # -------------------------
    # DATALOADERS
    # -------------------------
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )

    # ======================================================
    # MODELLO — ResNet18 per regressione (15 canali)
    # ======================================================
    model = models.resnet18(weights=None)

    # --- conv1: 15 canali ---
    model.conv1 = nn.Conv2d(
        in_channels=15,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    # --- fc: regressione ---
    model.fc = nn.Linear(model.fc.in_features, 1)

    # ------------------------------------------------------
    # Caricamento pesi ImageNet (SENZA conv1 e fc)
    # ------------------------------------------------------
    state_dict = torch.load(
        "resnet18-f37072fd.pth",  # path corretto
        map_location="cpu"
    )

    # Rimuovi layer incompatibili
    for k in ["conv1.weight", "fc.weight", "fc.bias"]:
        if k in state_dict:
            state_dict.pop(k)

    # Carica backbone
    model.load_state_dict(state_dict, strict=False)

    model.to(device)

    print("[INFO] ResNet18 caricato:")
    print("       - backbone ")
    print("       - conv1 (15 canali) random")
    print("       - fc (regressione) random")



    # -------------------------
    # LOSS & OPTIMIZER
    # -------------------------
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_val = np.inf

    # ======================================================
    # TRAINING LOOP
    # ======================================================
    # ======================================================
    # ======================================================
    # TRAINING — single phase fine-tuning (pretraining-aware)
    # ======================================================
    print("\n[TRAINING] Fine-tuning ResNet18 con scalogrammi")

    # ------------------------------------------------------
    # FREEZE / UNFREEZE STRATEGY
    # ------------------------------------------------------
    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if (
            name.startswith("layer3")
            or name.startswith("layer4")
            or name.startswith("fc")
        ):
            param.requires_grad = True

    # ------------------------------------------------------
    # OPTIMIZER (LR differenziato)
    # ------------------------------------------------------
    optimizer = optim.AdamW(
        [
            {"params": model.layer3.parameters(), "lr": 5e-5},
            {"params": model.layer4.parameters(), "lr": 5e-5},
            {"params": model.fc.parameters(),     "lr": 1e-4},
        ],
        weight_decay=1e-4
    )

    best_val = np.inf

    # ------------------------------------------------------
    # utility: MAE per fascia d'età
    # ------------------------------------------------------
    def mae_by_age_bins(y_true, y_pred, bins=(0, 30, 40, 50, 60, 70, 120)):
        out = {}
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            mask = (y_true >= lo) & (y_true < hi)
            if mask.sum() > 0:
                out[f"{lo}-{hi}"] = mean_absolute_error(
                    y_true[mask], y_pred[mask]
                )
        return out


    # ======================================================
    # TRAIN LOOP
    # ======================================================
    for epoch in range(1, EPOCHS + 1):
        # -------------------------
        # TRAIN
        # -------------------------
        model.train()
        train_loss = 0.0

        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS}",
            unit="batch",
            leave=False
        )

        for xb, yb in loop:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb).squeeze(1)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            loop.set_postfix(loss=f"{loss.item():.3f}")

        train_loss /= len(train_loader.dataset)

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                preds = model(xb).squeeze(1)
                loss = criterion(preds, yb)

                val_loss += loss.item() * xb.size(0)

                y_true.append(yb.cpu().numpy())
                y_pred.append(preds.cpu().numpy())

        val_loss /= len(val_loader.dataset)

        y_true = np.concatenate(y_true) * y_std + y_mean
        y_pred = np.concatenate(y_pred) * y_std + y_mean

        val_mae = mean_absolute_error(y_true, y_pred)
        age_mae = mae_by_age_bins(y_true, y_pred)

        # -------------------------
        # LOG
        # -------------------------
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val MAE: {val_mae:.2f}"
        )

        print("  MAE per fascia età:", end=" ")
        for k, v in age_mae.items():
            print(f"{k}:{v:.1f}", end="  ")
        print()

        # -------------------------
        # CHECKPOINT
        # -------------------------
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "y_mean": y_mean,
                    "y_std": y_std
                },
                "best_age_scalogram_model.pth"
            )
            print("✓ Best model saved")


    # ======================================================
    # TEST
    # ======================================================
    print("\n[INFO] Testing best model")

    ckpt = torch.load(
        "best_age_scalogram_model.pth",
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)

            preds = model(xb).squeeze(1)          # [batch]
            preds = preds.cpu().numpy()

            y_pred.append(preds)
            y_true.append(yb.numpy())

    # ======================================================
    # CONCATENAZIONE BATCH → ARRAY 1D
    # ======================================================
    y_true = np.concatenate(y_true).astype(np.float32)
    y_pred = np.concatenate(y_pred).astype(np.float32)

    # inverse normalization (ENTRAMBI)
    y_true = y_true * y_std + y_mean
    y_pred = y_pred * y_std + y_mean     

    print("y_true sample:", y_true[:5])
    print("y_pred sample:", y_pred[:5])


    # ======================================================
    # ERRORE ASSOLUTO
    # ======================================================
    abs_error = np.abs(y_true - y_pred)

    # 🔑 CLIPPING (es. massimo 20 anni)
    CLIP_VALUE = 20.0
    abs_error_clipped = np.minimum(abs_error, CLIP_VALUE)

    # MAE standard
    mae = abs_error.mean()

    # MAE clipped
    mae_clipped = abs_error_clipped.mean()

    # ======================================================
    # ALTRE METRICHE
    # ======================================================
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    pearson = pearsonr(y_true, y_pred)[0]

    print("\n[FINAL TEST RESULTS]")
    print(f"MAE (standard) : {mae:.2f}")
    print(f"MAE (clipped)  : {mae_clipped:.2f}")
    print(f"RMSE           : {rmse:.2f}")
    print(f"PearsonR       : {pearson:.3f}")

    # ======================================================
    # SAVE FINAL MODEL
    # ======================================================
    torch.save(model.state_dict(), "ptbxl_scalogram_age_resnet18.pth")
    print("[INFO] Modello finale salvato")

if __name__ == "__main__":
    main()
