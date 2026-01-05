#!/usr/bin/env python3
# %% 

import os
import ast
from pathlib import Path
import numpy as np
import pandas as pd

import wfdb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, median_absolute_error
from scipy.stats import pearsonr, spearmanr




# ======================================================
# DEVICE (Mac: usa MPS se disponibile)
# ======================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# %%
# -------------------------
# Config
# -------------------------
BASE_DIR = Path("ptb-xl")
RECORDS_FOLDER = "records500"
OUTPUT_ROOT = Path("ptbxl_csv")

records_dir = BASE_DIR / RECORDS_FOLDER
out_root = OUTPUT_ROOT / RECORDS_FOLDER
out_root.mkdir(parents=True, exist_ok=True)


# %%
def load_all_ptbxl_csv_regression(
    csv_folder,
    df,
    filename_col='filename_hr',
    target_col='age',
    verbose=False
):
    """
    Carica segnali ECG PTB-XL e restituisce:
        X: np.ndarray (N, samples, channels)
        y: np.ndarray (N,)  -> età (regressione)
    """
    csv_root = Path(csv_folder)

    X_list = []
    y_list = []
    missing = []

    for idx, row in df.iterrows():
        file_name = str(row[filename_col])
        age = row.get(target_col, None)

        # scarta record senza età valida
        if age is None or pd.isna(age):
            continue

        # normalizza filename
        p = Path(file_name)
        parts = list(p.parts)
        while parts and (parts[0] == RECORDS_FOLDER or parts[0] == csv_root.name):
            parts.pop(0)
        file_rel = Path(*parts) if parts else Path(p.name)

        # possibili path
        candidates = [
            csv_root / (str(file_rel) + ".npy"),
            csv_root / (str(file_rel) + ".csv"),
            csv_root / (file_rel.name + ".npy"),
            csv_root / (file_rel.name + ".csv")
        ]

        if file_rel.parent != Path('.'):
            candidates.extend([
                csv_root / file_rel.parent / (file_rel.name + ".npy"),
                csv_root / file_rel.parent / (file_rel.name + ".csv")
            ])

        found_path = None
        for cand in candidates:
            if cand.exists():
                found_path = cand
                break

        if found_path is None:
            missing.append(str(csv_root / (str(file_rel) + ".csv")))
            if verbose and (idx + 1) % 50 == 0:
                print(f"[warn] File non trovato: {file_rel}")
            continue

        # carica segnale
        try:
            if found_path.suffix == ".npy":
                ecg_signal = np.load(found_path)
            else:
                ecg_signal = pd.read_csv(found_path, header=0).values
        except Exception as e:
            if verbose:
                print(f"[errore] Lettura fallita {found_path}: {e}")
            missing.append(str(found_path))
            continue

        X_list.append(ecg_signal)
        y_list.append(float(age))

        if verbose and len(X_list) % 50 == 0:
            print(f"Caricati {len(X_list)} ECG")

    if len(X_list) == 0:
        raise RuntimeError(
            "Nessun file caricato.\n"
            f"Esempi filename:\n{df[filename_col].head(10).tolist()}\n"
            f"Missing:\n{missing[:10]}"
        )

    # verifica shape coerente
    shapes = [x.shape for x in X_list]
    uniq_shapes = list(dict.fromkeys(shapes))
    if len(uniq_shapes) != 1:
        raise RuntimeError(
            f"Segnali con shape diverse: {uniq_shapes}. "
            "Serve padding o trimming."
        )

    X = np.stack(X_list, axis=0)  # (N, samples, channels)
    y = np.array(y_list, dtype=np.float32)

    if verbose:
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"Età: min={y.min():.1f}, max={y.max():.1f}, mean={y.mean():.1f}")

    return X, y



# %%


def ptbxl_to_csv(record_id, records_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    record_path = os.path.join(records_dir, str(record_id))
    record = wfdb.rdrecord(record_path)
    data = record.p_signal  # (samples, channels)
    df = pd.DataFrame(data, columns=[f"lead_{i+1}" for i in range(data.shape[1])])
    out_file = os.path.join(output_dir, f"{record_id}.csv")
    df.to_csv(out_file, index=False)
    return out_file



class ECGDataset(Dataset):
    def __init__(self, X, y):
        # X: numpy array (N, channels, samples)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# %%
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class ResNet1D(nn.Module):
    def __init__(self, in_channels, layers=[2, 2, 2, 2]):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(
            in_channels, 64,
            kernel_size=7, stride=2,
            padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, 1)  # regressione

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )

        layers = []
        layers.append(
            BasicBlock1D(self.in_channels, out_channels, stride, downsample)
        )
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(
                BasicBlock1D(self.in_channels, out_channels)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)

        return x
    
  




# %%
def train_and_evaluate_regression(
    X,
    y,
    epochs=20,
    batch_size=32,
    lr=1e-4,
    weight_decay=1e-4,
    verbose=True
):
    """
    X: numpy array (N, samples, channels)
    y: numpy array (N,) -> età
    """

    # -----------------------------
    # Preparazione dati
    # -----------------------------
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError("X deve avere shape (N, samples, channels)")

    # (N, samples, channels) -> (N, channels, samples)
    X = X.transpose(0, 2, 1)

    y = np.asarray(y, dtype=np.float32)

    # normalizzazione target (IMPORTANTISSIMO)
    y_mean = y.mean()
    y_std = y.std()
    y_norm = (y - y_mean) / y_std

    # split (no stratify in regressione)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_norm, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    if verbose:
        print("TRAIN :", X_train.shape, y_train.shape)
        print("VAL   :", X_val.shape, y_val.shape)
        print("TEST  :", X_test.shape, y_test.shape)

    # -----------------------------
    # Dataset & DataLoader
    # -----------------------------
    train_ds = ECGDataset(X_train, y_train)
    val_ds   = ECGDataset(X_val, y_val)
    test_ds  = ECGDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # -----------------------------
    # Modello
    # -----------------------------
    model = ResNet1D(in_channels=X.shape[1]).to(device)

    criterion = nn.HuberLoss()   # migliore di MSE per ECG
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1)  # (B, 1)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_ds)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device).unsqueeze(1)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_ds)

        if verbose:
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"Train MAE (norm): {train_loss:.4f} | "
                f"Val MAE (norm): {val_loss:.4f}"
            )

    # -----------------------------
    # Test
    # -----------------------------
    # -----------------------------
# Test
# -----------------------------
    model.eval()
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).cpu().numpy().squeeze()
            preds_all.append(preds)
            targets_all.append(yb.numpy())

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    # -----------------------------
    # Denormalizzazione
    # -----------------------------
    preds_age = preds_all * y_std + y_mean
    targets_age = targets_all * y_std + y_mean

    # -----------------------------
    # Statistiche
    # -----------------------------
    errors = preds_age - targets_age
    abs_errors = np.abs(errors)

    stats = {
        "MAE": mean_absolute_error(targets_age, preds_age),
        "MedAE": median_absolute_error(targets_age, preds_age),
        "RMSE": np.sqrt(np.mean(errors ** 2)),
        "Bias": np.mean(errors),
        "StdError": np.std(errors),
        "MAPE": np.mean(abs_errors / (targets_age + 1e-6)) * 100,
        "PearsonR": pearsonr(targets_age, preds_age)[0],
        "SpearmanR": spearmanr(targets_age, preds_age)[0],
    }

    return model, stats





def compute_regression_stats(preds_age, targets_age):
    
    errors = preds_age - targets_age
    abs_errors = np.abs(errors)

    stats = {
            "MAE": mean_absolute_error(targets_age, preds_age),
            "MedAE": median_absolute_error(targets_age, preds_age),
            "RMSE": np.sqrt(np.mean(errors ** 2)),
            "Bias": np.mean(errors),
            "StdError": np.std(errors),
            "MAPE": np.mean(abs_errors / (targets_age + 1e-6)) * 100,
            "PearsonR": pearsonr(targets_age, preds_age)[0],
            "SpearmanR": spearmanr(targets_age, preds_age)[0],
            }

    return stats



# %%

def main():
  
    # ----------------------------------
    # 1) Carica database PTB-XL
    # ----------------------------------
    scp_file = BASE_DIR / "ptbxl_database.csv"
    if not scp_file.exists():
        raise FileNotFoundError(f"File non trovato: {scp_file}")

    df_db = pd.read_csv(scp_file)

    # Usa SOLO record con età valida
    df_db = df_db[df_db["age"].notna()].reset_index(drop=True)

    print(f"[INFO] Record totali con età valida: {len(df_db)}")

    # ----------------------------------
    # 2) Genera CSV ECG se non esistono
    # ----------------------------------
    csv_folder = OUTPUT_ROOT / RECORDS_FOLDER
    records_dir = BASE_DIR / RECORDS_FOLDER

    csv_folder.mkdir(parents=True, exist_ok=True)
    print("[INFO] Controllo/generazione CSV ECG...")
    print(f"[INFO] Totale record da processare: {len(df_db)}")

    n_total = len(df_db)
    n_converted = 0
    n_skipped = 0
    n_missing = 0

    for i, (_, row) in enumerate(df_db.iterrows(), start=1):
        fh = row["filename_hr"]

        # rimuovi eventuale prefisso
        if fh.startswith(f"{RECORDS_FOLDER}/"):
            fh = fh.replace(f"{RECORDS_FOLDER}/", "", 1)

        record_id = os.path.basename(fh)
        record_dir = records_dir / os.path.dirname(fh)
        out_dir = csv_folder / os.path.dirname(fh)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"{record_id}.csv"

        # --- CSV già esistente ---
        if out_csv.exists():
            n_skipped += 1
            continue

        hea_path = record_dir / f"{record_id}.hea"

        # --- File WFDB mancante ---
        if not hea_path.exists():
            n_missing += 1
            print(f"[WARN] File .hea mancante: {hea_path}")
            continue

        # --- Conversione ---
        try:
            ptbxl_to_csv(record_id, str(record_dir), str(out_dir))
            n_converted += 1
        except Exception as e:
            print(f"[ERROR] Errore conversione {record_id}: {e}")
            continue

        # --- Stampa di avanzamento ogni 100 record ---
        if i % 100 == 0 or i == n_total:
            print(
                f"[INFO] Processati {i}/{n_total} | "
                f"Creati: {n_converted} | "
                f"Saltati: {n_skipped} | "
                f"Mancanti: {n_missing}"
            )

    print("\n[INFO] Conversione completata.")
    print(f"[INFO] CSV creati    : {n_converted}")
    print(f"[INFO] CSV già esistenti: {n_skipped}")
    print(f"[INFO] Record mancanti : {n_missing}")


    # ----------------------------------
    # 3) Carica segnali + target età
    # ----------------------------------
    print("[INFO] Caricamento segnali ECG...")

    X, y = load_all_ptbxl_csv_regression(
        csv_folder=csv_folder.as_posix(),
        df=df_db,
        filename_col="filename_hr",
        target_col="age",
        verbose=True
    )

    print(f"[INFO] Dataset finale: X={X.shape}, y={y.shape}")
    print(f"[INFO] Età: min={y.min():.1f}, max={y.max():.1f}, mean={y.mean():.1f}")

    # ----------------------------------
    # 4) Training & valutazione
    # ----------------------------------
    print("[INFO] Avvio training regressione età (ResNet-18 1D)")

    model, stats = train_and_evaluate_regression(
        X,
        y,
        epochs=30,
        batch_size=32,
        lr=1e-4,
        weight_decay=1e-4,
        verbose=True
    )


    print("\n[RESULTS FINALI]")
    for k, v in stats.items():
        if isinstance(v, float):
            if k in ["PearsonR", "SpearmanR"]:
                print(f"{k:10s}: {v:.3f}")
            else:
                print(f"{k:10s}: {v:.2f}")
        else:
            print(f"{k:10s}: {v}")


    


# ----------------------------------
# Entry point
# ----------------------------------
if __name__ == "__main__":
    main()
