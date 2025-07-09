# src/models/train_loop.py
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Tuple


def train_one_run(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device = torch.device("cpu"),
    num_epochs: int = 10,
    # grad_acc_steps: int = 1,
) -> Tuple[Dict[str, list], nn.Module]:
    """
    Простая функция тренировки модели на одну серию гиперпараметров.
    Возвращает словарь с историей метрик и обученную модель.
    """

    # --- инициализация истории ---
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    model.to(device)                                   # переводим модель на CPU/GPU

    for epoch in range(num_epochs):
        model.train()                                  # режим обучения
        running_loss = 0.0

        optimizer.zero_grad()                          # обнуляем градиенты перед эпохой

        for step, (x_batch, y_batch) in enumerate(train_loader, 1):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # --- FORWARD ---
            logits = model(x_batch)                    # прямой проход
            loss = criterion(logits.squeeze(), y_batch.float())  # вычисляем loss
            loss.backward()                            # --- BACKWARD --- вычисляем градиенты

            # # NOTE: пока не обращаем внимание на этот кусок. Если что, пригодится позже
            # # --- градиентный шаг каждые grad_acc_steps батчей ---
            # if step % grad_acc_steps == 0:
            #     optimizer.step()                       # обновляем веса
            #     optimizer.zero_grad()                  # обнуляем градиенты

            running_loss += loss.item()

        # --- оценка на валидации ---
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # сохраняем метрики за эпоху
        history["train_loss"].append(running_loss / len(train_loader))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"train_loss={history['train_loss'][-1]:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

    return history, model


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Валидация модели: возвращает средний loss и accuracy."""
    model.eval()                                       # режим инференса
    total_loss, correct, n_examples = 0.0, 0, 0

    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits.squeeze(), y_batch.float())
        total_loss += loss.item()

        preds = (logits.squeeze() > 0).long()          # бинарный прогноз
        correct += (preds == y_batch).sum().item()
        n_examples += y_batch.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / n_examples
    return avg_loss, accuracy
