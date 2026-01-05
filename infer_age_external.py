import torch
import numpy as np
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader


# ==================================================
# Config inferenza (DEVE combaciare col training)
# ==================================================
target_samples = 5000   

class ECGInferenceDataset(Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]



def pad_trim_signals(signals, target_len):
    """
    Pad o trim dei segnali ECG a lunghezza fissa.

    Parametri
    ----------
    signals : list of np.ndarray
        Lista di array (samples, channels)
    target_len : int
        Numero target di samples

    Ritorna
    -------
    np.ndarray
        Array di shape (N, target_len, channels)
    """

    out = []

    for sig in signals:
        sig = np.asarray(sig)

        if sig.ndim != 2:
            raise ValueError(f"Segnale con shape non valida: {sig.shape}")

        n_samples, n_channels = sig.shape

        if n_samples == target_len:
            sig_out = sig
        elif n_samples > target_len:
            # trim centrale
            start = (n_samples - target_len) // 2
            sig_out = sig[start:start + target_len, :]
        else:
            # pad con zeri
            pad_before = (target_len - n_samples) // 2
            pad_after = target_len - n_samples - pad_before
            sig_out = np.pad(
                sig,
                ((pad_before, pad_after), (0, 0)),
                mode="constant"
            )

        out.append(sig_out)

    return np.stack(out, axis=0)


from ageRegressionPTB_Coni import (
    ResNet1D,
    ECGDataset,
    compute_regression_stats
)
# -----------------------------
# Device
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -----------------------------
# Carica modello
# -----------------------------
ckpt = torch.load(
    "models/resnet1d_ptbxl_age.pth",
    map_location=device,
    weights_only=False
)

model = ResNet1D(in_channels=12)
model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()

y_mean = ckpt["y_mean"]
y_std = ckpt["y_std"]

print("[INFO] Modello caricato correttamente")

# -----------------------------
# Carica nuovo dataset (250 ECG)
# -----------------------------


def load_external_ecg(
    labels_csv_path,
    root_folder="ECG_signals",
    filename_col=0,
    label_col=1,
    has_header=False,
    verbose=False,
    return_labels=False
):
    """
    Carica ECG da dataset esterno (es. CONI) per inferenza o valutazione.

    Parametri
    ----------
    labels_csv_path : str or Path
        CSV con almeno una colonna contenente il path relativo dei file ECG.
        Può essere passato anche senza estensione .csv.
    root_folder : str
        Cartella base dove si trovano i file ECG.
    filename_col : int
        Indice colonna con il path del file ECG.
    label_col : int
        Indice colonna con label (usato SOLO se return_labels=True).
    has_header : bool
        Se True, il CSV ha header.
    verbose : bool
        Stampa messaggi di debug.
    return_labels : bool
        Se True ritorna anche y (utile se hai età reali).
        Se False ritorna solo X.

    Ritorna
    -------
    X : np.ndarray or list
        (N, samples, channels) oppure lista di array se lunghezze diverse
    y : np.ndarray (opzionale)
        Label associate (se return_labels=True)
    """



    # -----------------------------
    # Trova file labels
    # -----------------------------
    labels_path = Path(labels_csv_path)
    if not labels_path.exists():
        if Path(str(labels_csv_path) + ".csv").exists():
            labels_path = Path(str(labels_csv_path) + ".csv")
        else:
            raise FileNotFoundError(f"File labels non trovato: {labels_csv_path}")

    # -----------------------------
    # Leggi CSV labels
    # -----------------------------
   # -----------------------------
# Leggi CSV labels (robusto a ; o ,)
# -----------------------------
    try:
        if has_header:
            df_labels = pd.read_csv(labels_path, sep=';')
        else:
            df_labels = pd.read_csv(labels_path, sep=';', header=None)

        # se ha una sola colonna, il separatore è sbagliato
        if df_labels.shape[1] == 1:
            raise ValueError("Separatore ; non corretto")

    except Exception:
        if has_header:
            df_labels = pd.read_csv(labels_path, sep=',')
        else:
            df_labels = pd.read_csv(labels_path, sep=',', header=None)


    if df_labels.shape[1] < 1:
        raise RuntimeError(
            f"File labels {labels_path} deve avere almeno una colonna (path file ECG)"
        )

    fname_series = df_labels.iloc[:, filename_col]
    label_series = df_labels.iloc[:, label_col] if return_labels and df_labels.shape[1] > label_col else None

    root = Path(root_folder)

    X_list = []
    y_list = []
    missing = []

    # -----------------------------
    # Loop sui file
    # -----------------------------
    for idx, fname_raw in enumerate(fname_series):
        fname = str(fname_raw).strip().replace("\\", "/")

        if fname == "" or pd.isna(fname):
            if verbose:
                print(f"[external] riga {idx}: filename vuoto, skip")
            missing.append(f"<empty at {idx}>")
            continue

        p = Path(fname)

        # Candidati possibili
        cand_paths = [
            Path(fname),
            root / p,
            root / (p.name + ".csv"),
            root / p.parent / (p.name + "_hr.csv"),
            root / Path(str(fname))
        ]

        found = None
        for cand in cand_paths:
            if cand.exists():
                found = cand
                break

        if found is None:
            if verbose:
                print(f"[external] file non trovato (riga {idx}): {fname}")
            missing.append(fname)
            continue

        # -----------------------------
        # Carica ECG
        # -----------------------------
        try:
            arr = pd.read_csv(found, header=0).values
        except Exception as e:
            if verbose:
                print(f"[external] errore leggendo {found}: {e}")
            missing.append(str(found))
            continue

        if arr.ndim != 2:
            if verbose:
                print(f"[external] formato non valido {found}, skip")
            continue

        # Normalizza forma → (samples, channels)
        if arr.shape[1] == 12:
            X_list.append(arr)
        elif arr.shape[0] == 12:
            X_list.append(arr.T)
        else:
            if verbose:
                print(f"[external] {found} ha shape {arr.shape} (non 12 lead)")
            continue

        if return_labels:
            y_list.append(label_series.iloc[idx])

    # -----------------------------
    # Check finale
    # -----------------------------
    if len(X_list) == 0:
        raise RuntimeError(
            f"Nessun ECG esterno caricato. Alcuni path mancanti:\n{missing[:10]}"
        )

    # Se tutte le forme sono uguali → stack
    shapes = [x.shape for x in X_list]
    if len(set(shapes)) == 1:
        X = np.stack(X_list, axis=0)
    else:
        X = X_list
        if verbose:
            print("[external] Lunghezze diverse, restituita lista (serve pad/trim)")

    if return_labels:
        y = np.array(y_list)
        if verbose:
            print(f"[external] Caricati {len(X_list)} ECG con label")
        return X, y
    else:
        if verbose:
            print(f"[external] Caricati {len(X_list)} ECG (inferenza pura)")
        return X

# ==================================================
# Carica dataset esterno
# ==================================================
X_ext, y_ext = load_external_ecg(
    labels_csv_path="dataset_eta_segnali",
    return_labels=True,   # metti False se NON hai età vere
    verbose=True
)

# ==================================================
# 1) Pad / Trim (PRIMA DI QUALSIASI NUMPY OP)
# ==================================================
if isinstance(X_ext, list):
    print("[INFO] ECG con lunghezze diverse → applico pad/trim")
    X_ext = pad_trim_signals(X_ext, target_samples)

print("[DEBUG] X_ext shape dopo pad/trim:", X_ext.shape)
# atteso: (N, 5000, 12)

# ==================================================
# 2) Rimozione NaN / Inf (ORA è sicuro)
# ==================================================
X_ext = np.nan_to_num(
    X_ext,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

# ==================================================
# 3) Normalizzazione z-score PER ECG e PER LEAD
# X_ext shape: (N, samples, channels)
# ==================================================
eps = 1e-8
mean = X_ext.mean(axis=1, keepdims=True)   # (N, 1, 12)
std  = X_ext.std(axis=1, keepdims=True)    # (N, 1, 12)
X_ext = (X_ext - mean) / (std + eps)

# (opzionale ma consigliato)
X_ext = np.clip(X_ext, -5, 5)

# ==================================================
# DEBUG: verifica normalizzazione
# ==================================================
print("\n[DEBUG] Verifica normalizzazione ECG esterni")
print("Shape X_ext:", X_ext.shape)
print("Valori globali -> min:", X_ext.min(),
      "max:", X_ext.max(),
      "mean:", X_ext.mean(),
      "std:", X_ext.std())

# Controllo per-lead sul primo ECG
mean_leads = X_ext[0].mean(axis=0)
std_leads = X_ext[0].std(axis=0)

print("\n[DEBUG] Primo ECG - statistiche per lead")
for i, (m, s) in enumerate(zip(mean_leads, std_leads), start=1):
    print(f"Lead {i:02d} | mean={m:+.3f} | std={s:.3f}")

# Riga randomica
rng = np.random.default_rng(seed=42)
ecg_idx = rng.integers(0, X_ext.shape[0])
sample_idx = rng.integers(0, X_ext.shape[1])

print("\n[DEBUG] Riga randomica normalizzata")
print(f"ECG index: {ecg_idx}, Sample index: {sample_idx}")
print(X_ext[ecg_idx, sample_idx, :])

print("\n[DEBUG] Check valori non validi")
print("NaN:", np.isnan(X_ext).sum())
print("Inf:", np.isinf(X_ext).sum())

# ==================================================
# 4) Conversione età a float (SE ci sono)
# ==================================================
if y_ext is not None:
    y_ext = np.asarray(y_ext, dtype=np.float32)
    print("\n[DEBUG] y_ext")
    print("dtype:", y_ext.dtype)
    print("Min:", y_ext.min(), "Max:", y_ext.max())
    print("Prime 10:", y_ext[:10])

# ==================================================
# 5) TRANSPOSE CORRETTO (UNA SOLA VOLTA!)
# (N, samples, channels) → (N, channels, samples)
# ==================================================
X_ext = X_ext.transpose(0, 2, 1).astype(np.float32)
print("[DEBUG] X_ext shape finale:", X_ext.shape)
# atteso: (N, 12, 5000)

# ==================================================
# 6) Dataset & DataLoader (INFERENZA)
# ==================================================
dataset = ECGInferenceDataset(X_ext)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# ==================================================
# 7) Inferenza
# ==================================================
model.eval()
preds_all = []

with torch.no_grad():
    for xb in loader:
        xb = xb.to(device)
        preds = model(xb).cpu().numpy().squeeze()
        preds_all.append(preds)

preds_all = np.concatenate(preds_all)

# ==================================================
# 8) Denormalizzazione età
# ==================================================
preds_age = preds_all * y_std + y_mean

# ==================================================
# 9) Output / Metriche
# ==================================================
if y_ext is not None:
    stats = compute_regression_stats(preds_age, y_ext)

    print("\n[RESULTS - DATASET ESTERNO]")
    for k, v in stats.items():
        if isinstance(v, float):
            if k in ["PearsonR", "SpearmanR"]:
                print(f"{k:10s}: {v:.3f}")
            else:
                print(f"{k:10s}: {v:.2f}")
        else:
            print(f"{k:10s}: {v}")
else:
    print("[INFO] Inferenza completata (etichette non disponibili)")
    print("Prime 10 età predette:", preds_age[:10])
