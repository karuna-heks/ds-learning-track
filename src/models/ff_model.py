# src/models/ff_model.py
"""
Feed-forward нейронная сеть для бинарной классификации.
Параметры (input_size, hidden_sizes, dropout_rate) лучше
передавать из config.yaml, чтобы не хардкодить в коде.
"""

from typing import List
import torch
from torch import nn


class FeedForwardNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = (128, 64),
        dropout_rate: float = 0.0,
        activation: str = "relu",
    ):
        """
        Args:
            input_size:  размер входного вектора признаков
            hidden_sizes: кортеж/список скрытых слоёв
            dropout_rate: вероятность дропа нейронов (0 ⇒ без Dropout)
            activation: 'relu' | 'tanh' | 'gelu' — по вкусу
        """
        super().__init__()

        # --- выбираем функцию активации ---
        act_layer = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
        }[activation]

        layers = []
        in_features = input_size

        # --- строим последовательность полносвязных слоёв ---
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))    # FC слой
            layers.append(act_layer())                  # нелинейность
            if dropout_rate > 0:                        # Dropout по желанию
                layers.append(nn.Dropout(dropout_rate))
            in_features = h

        # --- финальный слой: один логит (без сигмоиды) ---
        layers.append(nn.Linear(in_features, 1))

        self.net = nn.Sequential(*layers)               # собираем в Sequential

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход: возвращаем «сырые» логиты.
        Для BCEWithLogitsLoss не нужна сигмоида.
        """
        return self.net(x)
