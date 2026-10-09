"""Funções auxiliares: reprodutibilidade, dispositivo, dados, métricas e gráficos."""

import random

import matplotlib

matplotlib.use("Agg")  # backend sem interface gráfica (só salva arquivos)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.io import arff
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Nomes das classes após o mapeamento do rótulo (0 = legítimo, 1 = phishing)
CLASS_NAMES = ["Legítimo", "Phishing"]


def set_seed(seed=42):
    """Fixa as sementes aleatórias para garantir reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Retorna o melhor dispositivo disponível: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_arff_dataset(path):
    """Lê o arquivo .arff e retorna (X, y) como arrays numpy.

    Os atributos do dataset são nominais, então o loadarff os retorna como
    bytes (ex.: b'-1'); por isso convertemos tudo para inteiro.
    No atributo alvo 'Result', -1 indica phishing e 1 indica site legítimo.
    """
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    df = df.apply(lambda col: col.str.decode("utf-8").astype(int))

    X = df.drop(columns=["Result"]).values.astype(np.float32)
    # Mapeia o alvo para {0, 1}: 1 = phishing (Result == -1), 0 = legítimo
    y = (df["Result"].values == -1).astype(np.int64)
    return X, y


def prepare_data(path, val_size=0.2, test_size=0.2, seed=42):
    """Carrega o dataset, divide em treino/validação/teste e padroniza.

    O StandardScaler é ajustado apenas no treino para evitar vazamento de dados.
    """
    X, y = load_arff_dataset(path)

    # Primeiro separa o teste; depois separa a validação do restante
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    val_ratio = val_size / (1 - test_size)  # 0.2 / 0.8 = 0.25 do restante
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def make_dataloaders(train, val, test, batch_size=32, seed=42):
    """Cria os DataLoaders de treino (embaralhado), validação e teste."""

    def to_dataset(split):
        X, y = split
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    # Gerador com seed para que o embaralhamento seja reprodutível
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        to_dataset(train), batch_size=batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(to_dataset(val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(to_dataset(test), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def compute_metrics(y_true, y_pred):
    """Calcula acurácia, precisão, recall e F1 (classe positiva = phishing)."""
    return {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def plot_loss_curves(history, title, out_path):
    """Salva o gráfico das curvas de loss de treino e validação."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="Treino")
    ax.plot(epochs, history["val_loss"], label="Validação")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss (CrossEntropy)")
    ax.set_title(f"Curvas de loss — {title}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, title, out_path):
    """Salva a matriz de confusão do conjunto de teste."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Matriz de confusão (teste) — {title}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
