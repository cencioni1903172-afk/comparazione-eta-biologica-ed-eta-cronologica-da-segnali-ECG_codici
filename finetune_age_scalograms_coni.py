import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# ======================================================
# DEVICE
# ======================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# ======================================================
# CONFIG
# ======================================================
CSV_PATH = "dataset_età_scalogrammi.csv"
IMG_SIZE = 224
BATCH_SIZE = 8

# ======================================================
# DATASET
# ======================================================
class CONIScalogramDataset(Dataset):
    def __init__(self, df, transform=None, base_dir="scalograms"):
        self.df = df.reset_index(drop=True)
        self.img_cols = df.columns[:-1]
        self.labels = df.iloc[:, -1].values.astype(np.float32)
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

        # padding a 15 canali
        for _ in range(3):
            channels.append(torch.zeros_like(channels[0]))

        x = torch.cat(channels, dim=0)  # [15, H, W]
        y = torch.tensor(self.labels[idx], dtype=torch.float32)

        return x, y

# ======================================================
# MAIN
# ======================================================
def main():

    # =====================
    # LOAD CSV
    # =====================
    df = pd.read_csv(CSV_PATH, sep=',')
    df[df.columns[-1]] = pd.to_numeric(df[df.columns[-1]], errors="raise").astype(np.float32)

    print("Target stats:")
    print(df.iloc[:, -1].describe())

    _, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    _, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_ds = CONIScalogramDataset(test_df, transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ======================================================
    # MODELLO – REGRESSIONE
    # ======================================================
    model = models.resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        in_channels=15,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device).float()

    # ======================================================
    # LOAD MODELLO ADDESTRATO
    # ======================================================
    ckpt = torch.load(
    "best_age_scalogram_model.pth",
    map_location=device,
    weights_only=False   # 🔑 FONDAMENTALE in PyTorch ≥ 2.6
)


    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 🔑 recupera normalizzazione target
    y_mean = ckpt["y_mean"]
    y_std  = ckpt["y_std"]

    print(f"[INFO] Loaded y_mean={y_mean:.2f}, y_std={y_std:.2f}")


    # ======================================================
    # INFERENZA
    # ======================================================
    print("\n[INFO] Running inference")

    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).squeeze(1).cpu().numpy()

            y_pred.append(preds)
            y_true.append(yb.numpy())

    # ======================================================
    # CONCATENAZIONE
    # ======================================================
    y_true = np.concatenate(y_true).astype(np.float32)
    y_pred = np.concatenate(y_pred).astype(np.float32)

    # ======================================================
    # 🔑 INVERSE NORMALIZATION (SOLO y_pred)
    # ======================================================
    y_pred = y_pred * y_std + y_mean

    # ======================================================
    # METRICHE (ORA IN ANNI)
    # ======================================================
    abs_error = np.abs(y_true - y_pred)

    CLIP_VALUE = 20.0
    abs_error_clipped = np.minimum(abs_error, CLIP_VALUE)

    mae = abs_error.mean()
    mae_clipped = abs_error_clipped.mean()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pearson = pearsonr(y_true, y_pred)[0]

    print("\n[FINAL TEST RESULTS]")
    print(f"MAE (standard) : {mae:.2f}")
    print(f"MAE (clipped)  : {mae_clipped:.2f}")
    print(f"RMSE           : {rmse:.2f}")
    print(f"PearsonR       : {pearson:.3f}")

    
    
    # ======================================================
    # FINE-TUNING SU DATASET CONI
    # ======================================================
    print("\n[INFO] Starting fine-tuning on CONI dataset")

    # -------------------------
    # SPLIT TRAIN / VAL / TEST
    # -------------------------
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
    val_df, test_df   = train_test_split(temp_df, test_size=0.50, random_state=42)

    # -------------------------
    # NORMALIZZAZIONE TARGET (SU CONI)
    # -------------------------
    y_mean_coni = train_df.iloc[:, -1].mean()
    y_std_coni  = train_df.iloc[:, -1].std()

    for d in [train_df, val_df, test_df]:
        d.iloc[:, -1] = (d.iloc[:, -1] - y_mean_coni) / y_std_coni

    print(f"[INFO] CONI normalization | mean={y_mean_coni:.2f}, std={y_std_coni:.2f}")

    # -------------------------
    # DATASETS / LOADERS
    # -------------------------
    train_ds = CONIScalogramDataset(train_df, transform)
    val_ds   = CONIScalogramDataset(val_df, transform)
    test_ds  = CONIScalogramDataset(test_df, transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # -------------------------
    # FREEZE / UNFREEZE
    # -------------------------
    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if name.startswith("layer4") or name.startswith("fc"):
            p.requires_grad = True

    # -------------------------
    # LOSS & OPTIMIZER
    # -------------------------
    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.layer4.parameters(), "lr": 5e-5},
            {"params": model.fc.parameters(),     "lr": 1e-4},
        ],
        weight_decay=1e-4
    )

    best_val = np.inf
    EPOCHS_CONI = 20

    # ======================================================
    # TRAINING LOOP
    # ======================================================
    for epoch in range(1, EPOCHS_CONI + 1):
        # -------- TRAIN --------
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb).squeeze(1)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        # -------- VALIDATION --------
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

        y_true = np.concatenate(y_true) * y_std_coni + y_mean_coni
        y_pred = np.concatenate(y_pred) * y_std_coni + y_mean_coni
        val_mae = np.mean(np.abs(y_true - y_pred))

        print(
            f"[CONI] Epoch {epoch:02d}/{EPOCHS_CONI} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val MAE: {val_mae:.2f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "y_mean": y_mean_coni,
                    "y_std": y_std_coni
                },
                "best_coni_age_scalogram_model.pth"
            )
            print("✓ Best CONI model saved")

    # ======================================================
    # TEST FINALE CONI
    # ======================================================
    print("\n[INFO] Testing best CONI model")

    ckpt = torch.load(
        "best_coni_age_scalogram_model.pth",
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(ckpt["model_state"])
    y_mean_coni = ckpt["y_mean"]
    y_std_coni  = ckpt["y_std"]
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).squeeze(1).cpu().numpy()
            y_pred.append(preds)
            y_true.append(yb.numpy())

    y_true = np.concatenate(y_true) * y_std_coni + y_mean_coni
    y_pred = np.concatenate(y_pred) * y_std_coni + y_mean_coni

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    pearson = pearsonr(y_true, y_pred)[0]

    print("\n[FINAL CONI RESULTS]")
    print(f"MAE      : {mae:.2f}")
    print(f"RMSE     : {rmse:.2f}")
    print(f"PearsonR : {pearson:.3f}")

        # ======================================================
    # PREVISIONI VS ETÀ VERE (prime N)
    # ======================================================
    N_SHOW = 20  # quante righe stampare

    print("\n[PREVISIONI vs ETÀ VERE]")
    print("Idx | Età vera | Età predetta | Errore")

    for i in range(min(N_SHOW, len(y_true))):
        err = y_pred[i] - y_true[i]
        print(
            f"{i:03d} | "
            f"{y_true[i]:7.1f} | "
            f"{y_pred[i]:11.1f} | "
            f"{err:+6.1f}"
        )

    # ======================================================
    # PLOT: True vs Predicted ages (CONI test)
    # ======================================================
    try:
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        mn = min(np.min(y_true), np.min(y_pred))
        mx = max(np.max(y_true), np.max(y_pred))
        plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
        plt.xlabel('True age')
        plt.ylabel('Predicted age')
        plt.title(f'CONI Test — True vs Predicted Age')
        plt.xlim(mn - 1, mx + 1)
        plt.ylim(mn - 1, mx + 1)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('true_vs_pred_coni.png', dpi=150)
        plt.show()
        print('\n[INFO] Saved plot to true_vs_pred_coni.png')
    except Exception as e:
        print(f'[WARN] Unable to show/save plot: {e}')



# ======================================================
if __name__ == "__main__":
    main()




