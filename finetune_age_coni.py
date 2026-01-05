from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path



from ageRegressionPTB_Coni import (
    ResNet1D,
    compute_regression_stats
)

from infer_age_external import pad_trim_signals


# ======================================================
# Device
# ======================================================
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# ==================================================
# CONFIG
# ==================================================
BATCH_SIZE = 16        # oppure 8 se MPS va in memoria
LR = 1e-4              # learning rate fine-tuning
EPOCHS = 20
WEIGHT_DECAY = 1e-4
TARGET_SAMPLES = 5000
LABELS_CSV = "dataset_eta_segnali.csv"   
ROOT_ECG_FOLDER = "ECG_signals"
SAVE_PATH = "models/resnet1d_ptbxl_coni_finetuned.pth"
MODEL_PATH = "models/resnet1d_ptbxl_age.pth"


def freeze_layers(model, freeze_until="layer2"):
    freeze = True
    for name, param in model.named_parameters():
        if freeze:
            param.requires_grad = False
        if freeze_until in name:
            freeze = False


class ECGFinetuneDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ======================================================
# Load CONI (AGE REGRESSION)
# ======================================================
def load_coni_age_dataset(csv_path):
    df = pd.read_csv(csv_path)

    X_list, y_list = [], []

    for _, row in df.iterrows():
        path = str(row.iloc[0]).strip().replace("\\", "/")
        age  = float(row.iloc[1])

        full_path = Path(path)

        if not full_path.exists():
            print(f"[WARN] File non trovato: {full_path}")
            continue

        try:
            arr = pd.read_csv(full_path).values
        except Exception as e:
            print(f"[WARN] Errore lettura {full_path}: {e}")
            continue

        if arr.ndim != 2:
            continue

        if arr.shape[1] == 12:
            X_list.append(arr)
        elif arr.shape[0] == 12:
            X_list.append(arr.T)
        else:
            continue

        y_list.append(age)

    if len(X_list) == 0:
        raise RuntimeError("Nessun ECG CONI caricato correttamente")

    X = pad_trim_signals(X_list, TARGET_SAMPLES)
    y = np.array(y_list, dtype=np.float32)

    # rimuovi NaN / Inf
    X = np.nan_to_num(X)

    # normalizzazione per-ECG per-lead (COME PTB)
    mean = X.mean(axis=1, keepdims=True)
    std  = X.std(axis=1, keepdims=True)
    X = (X - mean) / (std + 1e-8)
    X = np.clip(X, -5, 5)

    # (N, samples, channels) → (N, 12, samples)
    X = X.transpose(0, 2, 1)

    print(f"[INFO] Caricati {len(X)} ECG CONI")

    return X, y




# ======================================================
# MAIN
# ======================================================
def main():

    # -------------------------
    # Load data
    # -------------------------
    X, y = load_coni_age_dataset(LABELS_CSV)
    print(f"[INFO] CONI dataset: {X.shape}, ages {y.min()}–{y.max()}")
    print("\n[INFO] Distribuzione età CONI")
    print(f"N totale: {len(y)}")
    print(f"Mean: {y.mean():.1f}")
    print(f"Median: {np.median(y):.1f}")

    bins = [0, 20, 30, 40, 50, 60]
    labels = ["<20", "20–30", "30–40", "40–50", "50+"]

    counts = pd.cut(y, bins=bins, labels=labels).value_counts().sort_index()

    print("\n[INFO] Conteggio per fascia d'età")
    print(counts)



    # -------------------------
    # Train / Val / Test split
    # -------------------------
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42
    )

    print(f"[INFO] Split → Train {len(X_tr)}, Val {len(X_val)}, Test {len(X_test)}")

    # -------------------------
    # Normalize target (ONCE)
    # -------------------------
    y_mean = y_tr.mean()
    y_std  = y_tr.std()

    y_tr_n  = (y_tr  - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std

    # -------------------------
    # Datasets & loaders
    # -------------------------
    train_ds = ECGFinetuneDataset(X_tr, y_tr_n)
    val_ds   = ECGFinetuneDataset(X_val, y_val_n)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # -------------------------
    # Load pretrained model
    # -------------------------
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    model = ResNet1D(in_channels=12)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    # -------------------------
    # Freeze backbone
    # -------------------------
    freeze_layers(model, freeze_until="layer3")

    print("\n[INFO] Trainable parameters:")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(" ✓", n)

    # -------------------------
    # Training setup
    # -------------------------
    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_val = float("inf")

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * xb.size(0)

        tr_loss /= len(train_ds)

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device).unsqueeze(1)
                preds = model(xb)
                val_loss += criterion(preds, yb).item() * xb.size(0)

        val_loss /= len(val_ds)

        print(f"Epoch {epoch:02d} | Train {tr_loss:.3f} | Val {val_loss:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "y_mean": y_mean,
                    "y_std": y_std
                },
                SAVE_PATH
            )

    # -------------------------
    # Final TEST evaluation
    # -------------------------
    print("\n[INFO] Evaluating fine-tuned model on TEST set")

    ckpt = torch.load(SAVE_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().to(device)
        preds_n = model(X_test_t).cpu().numpy().squeeze()

    preds_age = preds_n * y_std + y_mean

    stats = compute_regression_stats(preds_age, y_test)

    print("\n[FINAL TEST RESULTS - CONI]")
    for k, v in stats.items():
        if k in ["PearsonR", "SpearmanR"]:
            print(f"{k:10s}: {v:.3f}")
        else:
            print(f"{k:10s}: {v:.2f}")


    print("\n[DEBUG] Prime 10 predizioni TEST")
    for i in range(min(50, len(y_test))):
        print(
            f"True age: {y_test[i]:5.1f} | "
            f"Predicted age: {preds_age[i]:5.1f} | "
            f"Error: {preds_age[i] - y_test[i]:+5.1f}"
        )

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds_age, alpha=0.6)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        label="Ideal prediction"
    )
    plt.xlabel("True age")
    plt.ylabel("Predicted age")
    plt.title("CONI TEST — True vs Predicted Age")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ==================================================
    # 📊 PLOT: distribuzione errori
    # ==================================================
    errors = preds_age - y_test

    plt.figure(figsize=(6, 4))
    plt.hist(errors, bins=25, edgecolor="black")
    plt.xlabel("Prediction error (years)")
    plt.ylabel("Count")
    plt.title("CONI TEST — Prediction Error Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__": main()
