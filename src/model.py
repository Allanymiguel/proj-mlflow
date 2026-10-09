import torch.nn as nn


class MLP(nn.Module):
    """input_dim: número de features de entrada.
        hidden_dims: lista com o tamanho de cada camada oculta (ex.: [64, 32]).
        num_classes: número de classes de saída.
        dropout: probabilidade de dropout aplicada após cada ReLU (0 desativa).
    """

    def __init__(self, input_dim, hidden_dims, num_classes=2, dropout=0.0):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            # Dropout só é adicionado quando realmente usado
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Camada de saída: logits (a softmax fica implícita na CrossEntropyLoss)
        layers.append(nn.Linear(prev_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
