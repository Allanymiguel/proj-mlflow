import numpy as np
import torch


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Executa uma época de treino e retorna (loss média, acurácia)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        # Acumula a loss ponderada pelo tamanho do batch
        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += X.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Avalia o modelo em um conjunto e retorna (loss média, acurácia)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)

        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += X.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def predict(model, loader, device):
    """Retorna (rótulos verdadeiros, predições) para todo o conjunto."""
    model.eval()
    all_true, all_pred = [], []

    for X, y in loader:
        logits = model(X.to(device))
        all_pred.append(logits.argmax(dim=1).cpu().numpy())
        all_true.append(y.numpy())

    return np.concatenate(all_true), np.concatenate(all_pred)
