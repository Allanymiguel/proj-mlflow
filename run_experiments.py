"""Script principal: executa os experimentos de classificação de phishing com MLflow.

Uso:
    python run_experiments.py

Os runs são salvos em experiments/ (métricas em mlflow.db, artefatos em mlruns/).
Para visualizar:
    mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db
"""

import tempfile
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn

from src.model import MLP
from src.train import evaluate, predict, train_one_epoch
from src.utils import (
    compute_metrics,
    get_device,
    make_dataloaders,
    plot_confusion_matrix,
    plot_loss_curves,
    prepare_data,
    set_seed,
)

# ============================================================================
# Configurações gerais
# ============================================================================
ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "Training Dataset.arff"
TRACKING_DB = ROOT_DIR / "experiments" / "mlflow.db"  # params e métricas
ARTIFACTS_DIR = ROOT_DIR / "experiments" / "mlruns"  # modelos e gráficos
EXPERIMENT_NAME = "phishing-mlp"
SEED = 42

# ============================================================================
# Configurações dos experimentos (edite aqui para testar novas variações)
# ============================================================================
EXPERIMENTS = [
    {
        "run_name": "baseline",
        "question": "Como o modelo se comporta sem regularização?",
        "hidden_dims": [64, 32],
        "dropout": 0.0,
        "weight_decay": 0.0,
        "learning_rate": 1e-3,
        "epochs": 100,
        "batch_size": 32,
    },
    {
        "run_name": "dropout_l2",
        "question": "A regularização melhora a generalização?",
        "hidden_dims": [64, 32],
        "dropout": 0.3,
        "weight_decay": 1e-3,
        "learning_rate": 1e-3,
        "epochs": 100,
        "batch_size": 32,
    },
]


def setup_mlflow():
    """Configura o MLflow para salvar tudo dentro de experiments/.

    A partir do MLflow 3.x o backend em arquivos (./mlruns) está em modo de
    manutenção, então os metadados (params, métricas) vão para um SQLite em
    experiments/mlflow.db e os artefatos (modelos, gráficos) para experiments/mlruns.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")

    # O local dos artefatos só pode ser definido na criação do experimento
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACTS_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


def run_experiment(config, splits, device):
    """Executa um experimento completo (treino, validação, teste) e registra no MLflow."""
    # Reinicia a seed em cada experimento para que ambos partam do mesmo estado
    set_seed(SEED)

    train, val, test = splits
    train_loader, val_loader, test_loader = make_dataloaders(
        train, val, test, batch_size=config["batch_size"], seed=SEED
    )

    input_dim = train[0].shape[1]
    model = MLP(
        input_dim=input_dim,
        hidden_dims=config["hidden_dims"],
        num_classes=2,
        dropout=config["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    # weight_decay no Adam funciona como regularização L2
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    with mlflow.start_run(run_name=config["run_name"]):
        # --- Hiperparâmetros ---
        mlflow.set_tag("question", config["question"])
        mlflow.log_param("architecture", f"{input_dim} -> {config['hidden_dims']} -> 2")
        mlflow.log_param("hidden_dims", config["hidden_dims"])
        mlflow.log_param("dropout", config["dropout"])
        mlflow.log_param("weight_decay", config["weight_decay"])
        mlflow.log_param("learning_rate", config["learning_rate"])
        mlflow.log_param("epochs", config["epochs"])
        mlflow.log_param("batch_size", config["batch_size"])
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("seed", SEED)
        mlflow.log_param("device", str(device))

        # --- Treino com validação a cada época ---
        history = {"train_loss": [], "val_loss": []}
        for epoch in range(1, config["epochs"] + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"  Época {epoch:3d}/{config['epochs']} | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                )

        # --- Avaliação final no conjunto de teste ---
        y_true, y_pred = predict(model, test_loader, device)
        test_metrics = compute_metrics(y_true, y_pred)
        for name, value in test_metrics.items():
            mlflow.log_metric(f"test_{name}", value)
        print(
            "  Teste: "
            + " ".join(f"{k}={v:.4f}" for k, v in test_metrics.items())
        )

        # --- Modelo ---
        input_example = test[0][:5]
        mlflow.pytorch.log_model(model.cpu(), name="model", input_example=input_example)

        # --- Gráficos (salvos em pasta temporária e enviados como artefatos) ---
        with tempfile.TemporaryDirectory() as tmp:
            loss_path = Path(tmp) / "loss_curves.png"
            cm_path = Path(tmp) / "confusion_matrix.png"
            plot_loss_curves(history, config["run_name"], loss_path)
            plot_confusion_matrix(y_true, y_pred, config["run_name"], cm_path)
            mlflow.log_artifact(str(loss_path), artifact_path="plots")
            mlflow.log_artifact(str(cm_path), artifact_path="plots")

    return test_metrics


def main():
    set_seed(SEED)
    device = get_device()
    print(f"Dispositivo: {device}")

    setup_mlflow()

    # Os dados são carregados uma única vez e reutilizados por todos os experimentos
    splits = prepare_data(DATA_PATH, val_size=0.2, test_size=0.2, seed=SEED)
    train, val, test = splits
    print(f"Treino: {len(train[1])} | Validação: {len(val[1])} | Teste: {len(test[1])}")

    results = {}
    for config in EXPERIMENTS:
        print(f"\n=== Experimento: {config['run_name']} ===")
        print(f"Pergunta: {config['question']}")
        results[config["run_name"]] = run_experiment(config, splits, device)

    # Resumo comparativo no terminal
    print("\n=== Resumo (conjunto de teste) ===")
    for run_name, metrics in results.items():
        print(f"{run_name:>12}: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    print("\nPara ver os resultados: mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db")


if __name__ == "__main__":
    main()
