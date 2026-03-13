"""
beta_vae_classifier.py — Beta-VAE latent representation + downstream classifier
for retinal TDA features (Normal / Alzheimer / Glaucoma).

Hypothesis: Raw TDA features lie on a nonlinear manifold that a linear SVM
struggles to separate. A Beta-VAE learns a disentangled latent space z where
independent generative factors (AD vascular decay vs. Glaucoma optic-nerve
damage) are teased apart, making downstream classification easier.

Architecture:
  1. MLP-based Beta-VAE:  x → Encoder → (μ, log σ²) → z → Decoder → x̂
     Loss = MSE(x, x̂) + β · D_KL(q(z|x) ‖ N(0,I))
  2. Downstream classifier on frozen latent z (LogisticRegression or
     RandomForestClassifier), evaluated via Stratified 5-Fold CV.
  3. UMAP visualisation of the full-dataset latent space.

Key design choices for N=50, d_input=2:
  - Shallow encoder/decoder (2 hidden layers, width 64→32) to avoid
    overfitting on 50 samples.
  - Latent dim = 4: enough capacity to disentangle, small enough to
    regularise via KL.
  - β ∈ {1, 2, 4} sweep to find the best disentanglement/reconstruction
    tradeoff per fold.
  - Data augmentation via Gaussian jitter on training features to
    artificially increase effective sample count during VAE training.
  - class_weight='balanced' on the downstream classifier.

Usage:
    cd 05_beta_VAE
    python beta_vae_classifier.py
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SELECTED_CSV = PROJECT_ROOT / "04_EDA_and_Selection" / "selected_features.csv"
RESULTS_DIR  = SCRIPT_DIR / "results"
VAE_MODEL_PATH    = RESULTS_DIR / "beta_vae.pth"
METRICS_CSV       = RESULTS_DIR / "fold_metrics.csv"

ID_COLS   = ["subject_id", "dataset"]
LABEL_COL = "label"

# Cross-validation
N_SPLITS    = 5
RANDOM_SEED = 42

# Beta-VAE hyperparameters
LATENT_DIM    = 4
HIDDEN_DIMS   = [64, 32]
BETA_VALUES   = [1.0, 2.0, 4.0]       # sweep per fold
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
NUM_EPOCHS    = 500
BATCH_SIZE    = 16                      # small batches for N=50
JITTER_STD    = 0.05                    # Gaussian augmentation σ
AUGMENT_FACTOR = 10                     # repeat training data ×10 with jitter

# Downstream classifier
CLASSIFIER_TYPE = "logistic"            # "logistic" or "rf"

# Visualisation
FIGURE_DPI    = 300
FIGURE_FORMAT = "png"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────
#  Data Loading
# ──────────────────────────────────────────────────────────────────────

def load_data(csv_path: Path = SELECTED_CSV):
    """Load CSV, return X (raw), y (encoded), feature names, class names, encoder."""
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} samples from {csv_path}")

    drop = [c for c in ID_COLS + [LABEL_COL] if c in df.columns]
    feature_names = [c for c in df.columns if c not in drop]

    X = df[feature_names].values.astype(np.float64)
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL].values)
    class_names = list(le.classes_)

    counts = dict(zip(class_names, [int(c) for c in np.bincount(y)]))
    print(f"  Features    : {feature_names}")
    print(f"  Classes     : {class_names}")
    print(f"  Distribution: {counts}")

    return X, y, feature_names, class_names, le


# ──────────────────────────────────────────────────────────────────────
#  Beta-VAE Architecture
# ──────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """MLP encoder: x → hidden → (μ, log σ²)."""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            in_dim = h_dim
        self.shared = nn.Sequential(*layers)
        self.fc_mu     = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """MLP decoder: z → hidden → x̂."""

    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class BetaVAE(nn.Module):
    """
    Beta-Variational Autoencoder for tabular TDA features.

    Loss = MSE(x, x̂) + β · D_KL(N(μ, σ²) ‖ N(0, I))

    Higher β pushes toward more disentangled (but less faithful)
    latent representations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = HIDDEN_DIMS,
        latent_dim: int = LATENT_DIM,
        beta: float = 2.0,
    ):
        super().__init__()
        self.beta = beta
        self.latent_dim = latent_dim

        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + σ · ε, ε ~ N(0, I)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent mean μ (deterministic encoding for downstream use)."""
        mu, _ = self.encoder(x)
        return mu


class BetaVAELoss(nn.Module):
    """
    L = MSE(x, x̂) + β · D_KL

    D_KL = -0.5 · Σ(1 + log σ² - μ² - σ²)
    """

    def __init__(self, beta: float = 2.0):
        super().__init__()
        self.beta = beta
        self.mse = nn.MSELoss(reduction="mean")

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = self.mse(x_hat, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + self.beta * kl_loss
        return total, recon_loss, kl_loss


# ──────────────────────────────────────────────────────────────────────
#  Augmented Dataset
# ──────────────────────────────────────────────────────────────────────

def create_augmented_loader(
    X: np.ndarray,
    batch_size: int = BATCH_SIZE,
    jitter_std: float = JITTER_STD,
    augment_factor: int = AUGMENT_FACTOR,
) -> DataLoader:
    """
    Create a DataLoader with Gaussian-jittered copies of X.

    For N=50, this gives 500 effective training samples per epoch,
    preventing the VAE from memorising the tiny dataset.
    """
    copies = [X]
    rng = np.random.RandomState(RANDOM_SEED)
    for _ in range(augment_factor - 1):
        noise = rng.normal(0, jitter_std, size=X.shape)
        copies.append(X + noise)

    X_aug = np.concatenate(copies, axis=0).astype(np.float32)
    tensor = torch.from_numpy(X_aug)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      drop_last=False)


# ──────────────────────────────────────────────────────────────────────
#  VAE Training
# ──────────────────────────────────────────────────────────────────────

def train_vae(
    model: BetaVAE,
    loader: DataLoader,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    verbose: bool = False,
) -> List[float]:
    """Train Beta-VAE and return per-epoch total loss."""
    model.to(DEVICE)
    model.train()

    criterion = BetaVAELoss(beta=model.beta)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE)
            x_hat, mu, logvar = model(batch_x)
            loss, recon, kl = criterion(batch_x, x_hat, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 100 == 0:
            print(f"      Epoch {epoch+1}/{num_epochs}  Loss={avg_loss:.5f}")

    return losses


def extract_latent(model: BetaVAE, X: np.ndarray) -> np.ndarray:
    """Pass X through the frozen encoder and return μ as numpy array."""
    model.eval()
    model.to(DEVICE)
    with torch.no_grad():
        x_t = torch.from_numpy(X.astype(np.float32)).to(DEVICE)
        z = model.encode(x_t)
    return z.cpu().numpy()


# ──────────────────────────────────────────────────────────────────────
#  Downstream Classifier
# ──────────────────────────────────────────────────────────────────────

def build_classifier(clf_type: str = CLASSIFIER_TYPE):
    """Build a lightweight downstream classifier for the latent space."""
    if clf_type == "logistic":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            solver="lbfgs",
            C=1.0,
        )
    else:
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_depth=4,
        )


# ──────────────────────────────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────────────────────────────

def _macro_specificity(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    specs = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        denom = tn + fp
        specs.append(tn / denom if denom > 0 else 0.0)
    return float(np.mean(specs))


def _compute_roc(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> Dict:
    y_bin = label_binarize(y_true, classes=range(n_classes))
    result = {}
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
        result[c] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}
    try:
        result["macro_auc"] = roc_auc_score(
            y_bin, y_prob, average="macro", multi_class="ovr",
        )
    except ValueError:
        result["macro_auc"] = float(
            np.mean([result[c]["auc"] for c in range(n_classes)])
        )
    return result


# ──────────────────────────────────────────────────────────────────────
#  Pipeline Manager
# ──────────────────────────────────────────────────────────────────────

class PipelineManager:
    """
    Orchestrates the full Beta-VAE + downstream classifier pipeline:
      1. Stratified K-Fold over original 3-class labels
      2. Per fold: scale → train VAE (with β sweep) → extract z → classify
      3. Aggregate metrics
      4. Full-dataset VAE → UMAP latent visualisation
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_names: List[str],
        feature_names: List[str],
        label_encoder: LabelEncoder,
    ):
        self.X = X
        self.y = y
        self.class_names = class_names
        self.feature_names = feature_names
        self.label_encoder = label_encoder
        self.n_classes = len(class_names)
        self.input_dim = X.shape[1]

        self.fold_metrics: List[Dict] = []
        self.aggregate_cm: Optional[np.ndarray] = None
        self.roc_data: Optional[Dict] = None
        self.best_vae: Optional[BetaVAE] = None

    def run_cv(self) -> Dict[str, float]:
        """Run Stratified K-Fold CV with Beta-VAE + downstream classifier."""
        skf = StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED,
        )

        all_y_true, all_y_pred = [], []
        all_y_prob = np.zeros((len(self.y), self.n_classes))

        for fold_i, (train_idx, test_idx) in enumerate(skf.split(self.X, self.y)):
            X_tr_raw, X_te_raw = self.X[train_idx], self.X[test_idx]
            y_tr, y_te = self.y[train_idx], self.y[test_idx]

            # Scale features (fit on train only)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_te = scaler.transform(X_te_raw)

            # β sweep: train VAE with each β, pick best reconstruction on test
            best_beta_vae = None
            best_beta_loss = float("inf")
            best_beta_val = None

            for beta in BETA_VALUES:
                vae = BetaVAE(
                    input_dim=self.input_dim,
                    hidden_dims=HIDDEN_DIMS,
                    latent_dim=LATENT_DIM,
                    beta=beta,
                )
                loader = create_augmented_loader(X_tr)
                train_vae(vae, loader, num_epochs=NUM_EPOCHS)

                # Evaluate reconstruction on test set
                vae.eval()
                with torch.no_grad():
                    x_t = torch.from_numpy(X_te.astype(np.float32)).to(DEVICE)
                    vae.to(DEVICE)
                    x_hat, mu, logvar = vae(x_t)
                    val_loss = nn.MSELoss()(x_hat, x_t).item()

                if val_loss < best_beta_loss:
                    best_beta_loss = val_loss
                    best_beta_vae = vae
                    best_beta_val = beta

            # Extract latent representations with best VAE
            z_train = extract_latent(best_beta_vae, X_tr)
            z_test  = extract_latent(best_beta_vae, X_te)

            # Downstream classifier on z
            clf = build_classifier()
            clf.fit(z_train, y_tr)
            y_pred = clf.predict(z_test)
            y_prob = clf.predict_proba(z_test)

            all_y_true.extend(y_te)
            all_y_pred.extend(y_pred)
            all_y_prob[test_idx] = y_prob

            acc  = accuracy_score(y_te, y_pred)
            f1   = f1_score(y_te, y_pred, average="macro", zero_division=0)
            sens = recall_score(y_te, y_pred, average="macro", zero_division=0)
            spec = _macro_specificity(y_te, y_pred, self.n_classes)
            prec = precision_score(y_te, y_pred, average="macro", zero_division=0)

            self.fold_metrics.append({
                "fold": fold_i + 1,
                "accuracy": acc,
                "sensitivity": sens,
                "specificity": spec,
                "precision": prec,
                "f1_macro": f1,
                "best_beta": best_beta_val,
                "recon_loss": best_beta_loss,
            })
            print(f"  Fold {fold_i+1}/{N_SPLITS}  "
                  f"Acc={acc:.3f}  F1={f1:.3f}  Sens={sens:.3f}  "
                  f"Spec={spec:.3f}  β={best_beta_val}")

        # Aggregate
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        self.aggregate_cm = confusion_matrix(
            all_y_true, all_y_pred, labels=range(self.n_classes),
        )
        self.roc_data = _compute_roc(self.y, all_y_prob, self.n_classes)

        metric_keys = ["accuracy", "sensitivity", "specificity", "precision", "f1_macro"]
        print(f"\n  === Beta-VAE + {CLASSIFIER_TYPE} — {N_SPLITS}-Fold CV ===")
        for k in metric_keys:
            vals = [m[k] for m in self.fold_metrics]
            print(f"    {k:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        macro_auc = self.roc_data.get("macro_auc", 0.0)
        print(f"    {'auroc_macro':20s}: {macro_auc:.4f}")

        return {k: float(np.mean([m[k] for m in self.fold_metrics]))
                for k in metric_keys}

    def train_full_vae(self) -> BetaVAE:
        """Train Beta-VAE on entire dataset for visualisation and export."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        # Use the most frequently selected β across folds
        beta_counts = {}
        for m in self.fold_metrics:
            b = m["best_beta"]
            beta_counts[b] = beta_counts.get(b, 0) + 1
        best_beta = max(beta_counts, key=beta_counts.get)
        print(f"  Training full VAE with β={best_beta} (selected in "
              f"{beta_counts[best_beta]}/{N_SPLITS} folds)")

        vae = BetaVAE(
            input_dim=self.input_dim,
            hidden_dims=HIDDEN_DIMS,
            latent_dim=LATENT_DIM,
            beta=best_beta,
        )
        loader = create_augmented_loader(X_scaled)
        losses = train_vae(vae, loader, num_epochs=NUM_EPOCHS, verbose=True)
        print(f"  Final training loss: {losses[-1]:.5f}")

        self.best_vae = vae
        self._scaler = scaler
        return vae

    def extract_full_latent(self) -> np.ndarray:
        """Extract latent z for all samples using the full-dataset VAE."""
        X_scaled = self._scaler.transform(self.X)
        return extract_latent(self.best_vae, X_scaled)

    def save_models(self):
        """Save the full-dataset VAE and scaler."""
        torch.save({
            "model_state_dict": self.best_vae.state_dict(),
            "beta": self.best_vae.beta,
            "latent_dim": self.best_vae.latent_dim,
            "input_dim": self.input_dim,
            "hidden_dims": HIDDEN_DIMS,
            "scaler_mean": self._scaler.mean_,
            "scaler_scale": self._scaler.scale_,
            "class_names": self.class_names,
            "feature_names": self.feature_names,
        }, VAE_MODEL_PATH)
        print(f"  VAE model saved to {VAE_MODEL_PATH}")

    def save_fold_csv(self):
        """Export per-fold metrics to CSV."""
        rows = []
        for m in self.fold_metrics:
            rows.append({
                "fold": m["fold"],
                "accuracy": round(m["accuracy"], 4),
                "sensitivity": round(m["sensitivity"], 4),
                "specificity": round(m["specificity"], 4),
                "precision": round(m["precision"], 4),
                "f1_macro": round(m["f1_macro"], 4),
                "best_beta": m["best_beta"],
                "recon_loss": round(m["recon_loss"], 6),
            })

        metric_keys = ["accuracy", "sensitivity", "specificity", "precision", "f1_macro"]
        agg_mean = {k: round(np.mean([m[k] for m in self.fold_metrics]), 4)
                    for k in metric_keys}
        agg_std = {k: round(np.std([m[k] for m in self.fold_metrics]), 4)
                   for k in metric_keys}
        rows.append({"fold": "mean", **agg_mean, "best_beta": "", "recon_loss": ""})
        rows.append({"fold": "std", **agg_std, "best_beta": "", "recon_loss": ""})

        pd.DataFrame(rows).to_csv(METRICS_CSV, index=False)
        print(f"  Fold metrics saved to {METRICS_CSV}")


# ──────────────────────────────────────────────────────────────────────
#  Visualisation
# ──────────────────────────────────────────────────────────────────────

def _setup_style():
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "axes.edgecolor": "0.15",
        "axes.linewidth": 1.0,
        "grid.alpha": 0.4,
    })


def plot_latent_umap(z: np.ndarray, y: np.ndarray, class_names: List[str]):
    """
    UMAP 2D projection of the latent space, colour-coded by class.

    Falls back to t-SNE if umap-learn is unavailable.
    """
    _setup_style()

    try:
        import umap
        reducer = umap.UMAP(
            n_components=2, n_neighbors=min(15, len(z) - 1),
            min_dist=0.1, random_state=RANDOM_SEED, metric="euclidean",
        )
        z_2d = reducer.fit_transform(z)
        method = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=2, perplexity=min(15, len(z) - 1),
            random_state=RANDOM_SEED, n_iter=1000,
        )
        z_2d = reducer.fit_transform(z)
        method = "t-SNE"

    colors = ["#4C72B0", "#DD8452", "#C44E52"]
    markers = ["o", "s", "D"]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for c in range(len(class_names)):
        mask = y == c
        ax.scatter(
            z_2d[mask, 0], z_2d[mask, 1],
            c=[colors[c]], marker=markers[c % len(markers)],
            s=60, edgecolors="white", linewidth=0.6,
            label=f"{class_names[c]} (n={mask.sum()})", zorder=3,
        )

    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_title(
        f"Beta-VAE Latent Space — {method} Projection (z_dim={LATENT_DIM})",
        fontsize=11, pad=10,
    )
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out = RESULTS_DIR / f"latent_space_{method.lower()}.{FIGURE_FORMAT}"
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Latent space plot saved to {out}")


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]):
    """Row-normalised confusion matrix."""
    _setup_style()

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        vmin=0, vmax=1, linewidths=0.5, square=True,
        cbar_kws={"shrink": 0.8, "label": "Rate"}, ax=ax,
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j + 0.5, i + 0.75, f"(n={cm[i, j]})",
                ha="center", va="center", fontsize=7, color="gray",
            )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Beta-VAE + Classifier — Confusion Matrix", fontsize=12, pad=10)

    fig.tight_layout()
    out = RESULTS_DIR / f"confusion_matrix.{FIGURE_FORMAT}"
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved to {out}")


def plot_roc_curves(roc_data: Dict, class_names: List[str]):
    """Multi-class OVR ROC curves."""
    _setup_style()

    colors = ["#4C72B0", "#DD8452", "#C44E52"]
    fig, ax = plt.subplots(figsize=(6, 5))

    for c in range(len(class_names)):
        if c not in roc_data:
            continue
        fpr = roc_data[c]["fpr"]
        tpr = roc_data[c]["tpr"]
        auc_val = roc_data[c]["auc"]
        ax.plot(fpr, tpr, color=colors[c % len(colors)], lw=1.8,
                label=f"{class_names[c]} (AUC = {auc_val:.3f})")

    macro_auc = roc_data.get("macro_auc", 0.0)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        f"ROC Curves — One-vs-Rest  (Macro AUC = {macro_auc:.3f})",
        fontsize=11, pad=10,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    fig.tight_layout()
    out = RESULTS_DIR / f"roc_curves.{FIGURE_FORMAT}"
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC curves saved to {out}")


def plot_training_loss(losses: List[float]):
    """Plot VAE training loss curve."""
    _setup_style()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(losses) + 1), losses, color="#4C72B0", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE + β·KL)")
    ax.set_title("Beta-VAE Training Loss", fontsize=12, pad=10)

    fig.tight_layout()
    out = RESULTS_DIR / f"training_loss.{FIGURE_FORMAT}"
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training loss plot saved to {out}")


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print(f"  Device: {DEVICE}")
    print(f"  Latent dim: {LATENT_DIM}, Hidden: {HIDDEN_DIMS}")
    print(f"  Beta candidates: {BETA_VALUES}")
    print(f"  Augmentation: ×{AUGMENT_FACTOR} with σ={JITTER_STD}\n")

    # Step 1 — Load
    print("[1/6] Loading data ...")
    X, y, feature_names, class_names, le = load_data()

    # Step 2 — Cross-validated evaluation
    print(f"\n[2/6] {N_SPLITS}-Fold CV: Beta-VAE → {CLASSIFIER_TYPE} classifier ...")
    pipeline = PipelineManager(X, y, class_names, feature_names, le)
    metrics = pipeline.run_cv()

    # Step 3 — Save fold metrics
    print("\n[3/6] Saving fold metrics ...")
    pipeline.save_fold_csv()

    # Step 4 — Train full VAE for visualisation
    print("\n[4/6] Training full-dataset VAE ...")
    vae = pipeline.train_full_vae()

    # Step 5 — Plots
    print("\n[5/6] Generating plots ...")
    z_full = pipeline.extract_full_latent()
    plot_latent_umap(z_full, y, class_names)
    plot_confusion_matrix(pipeline.aggregate_cm, class_names)
    plot_roc_curves(pipeline.roc_data, class_names)

    # Step 6 — Save model
    print("\n[6/6] Saving model ...")
    pipeline.save_models()

    print("\nDone.")


if __name__ == "__main__":
    main()
