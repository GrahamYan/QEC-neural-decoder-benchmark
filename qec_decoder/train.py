"""Training utilities matching the paper appendix."""

from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset


@dataclass
class SchedulerConfig:
    warmup_epochs: int = 5
    decay_end_epoch: int = 80
    min_lr: float = 1e-6


@dataclass
class TrainConfig:
    d: int = 3
    r: Optional[int] = None
    p: float = 0.005
    train_size: Optional[int] = None
    ratio: float = 1.0
    dropout: float = 0.1
    weight_decay: float = 1e-3
    epochs: int = 100
    max_epochs: int = 150
    batch_size: int = 16_384
    num_workers: int = 8
    accumulation_steps: int = 2
    lr: float = 5e-4
    clip_grad: float = 1.0
    patience: int = 10
    min_delta: float = 1e-5
    label_smoothing: float = 0.0
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    save_dir: Optional[str] = "model"
    log_dir: Optional[str] = "logs"

    def __post_init__(self):
        if self.r is None:
            self.r = self.d


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_train_size(size: Optional[int]) -> str:
    if size is None:
        return "all"
    text = f"{float(size):.1e}".replace(".0e", "e").replace("e+0", "e").replace("e+", "e")
    return text


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def format_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def create_dataloaders(
    train_dataset,
    test_dataset,
    batch_size: int,
    num_workers: int,
    train_size: Optional[int] = None,
):
    if train_size is not None and train_size < len(train_dataset):
        train_dataset = Subset(train_dataset, list(range(train_size)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    return train_loader, test_loader


class CosineWarmupSchedule:
    """Linear warmup followed by cosine annealing to ``min_lr``."""

    def __init__(self, peak_lr: float, warmup_epochs: int, decay_end_epoch: int, min_lr: float, steps_per_epoch: int):
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = max(0, warmup_epochs * steps_per_epoch)
        self.decay_steps = max(1, (decay_end_epoch - warmup_epochs) * steps_per_epoch)

    def get_lr(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.peak_lr * (step + 1) / self.warmup_steps
        progress = min(1.0, max(0.0, (step - self.warmup_steps) / self.decay_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine


def _logical_error_rate(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if targets.ndim == 1:
        targets = targets.unsqueeze(1)
    pred = (logits > 0).float()
    correct = (pred == targets).all(dim=1).float().mean().item()
    return 1.0 - correct


def _evaluate_logits_model(model: nn.Module, loader, criterion, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if y.ndim == 1:
                y = y.unsqueeze(1)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = criterion(logits, y)
            pred = (logits > 0).float()
            total_loss += loss.item() * y.size(0)
            total_correct += (pred == y).all(dim=1).sum().item()
            total_samples += y.size(0)
    acc = total_correct / total_samples if total_samples else 0.0
    return total_loss / max(total_samples, 1), acc, 1.0 - acc


def save_history_csv(history: dict, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        keys = list(history.keys())
        writer.writerow(keys)
        for values in zip(*[history[k] for k in keys]):
            writer.writerow(values)


def get_model_filename(model_name: str, config: TrainConfig, suffix: str = "fp32") -> str:
    size = format_train_size(config.train_size)
    return f"{model_name}_{suffix}_d{config.d}_r{config.r}_p{config.p}_ratio{config.ratio}_size{size}.pth"


def train_decoder(
    model: nn.Module,
    train_loader,
    test_loader,
    config: TrainConfig,
    model_name: str = "decoder",
    suffix: str = "fp32",
) -> dict:
    """Trains a decoder with AdamW, BCEWithLogitsLoss, warmup+cosine LR, clipping, and early stopping."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, config.accumulation_steps)))
    lr_schedule = CosineWarmupSchedule(
        peak_lr=config.lr,
        warmup_epochs=config.scheduler.warmup_epochs,
        decay_end_epoch=config.scheduler.decay_end_epoch,
        min_lr=config.scheduler.min_lr,
        steps_per_epoch=steps_per_epoch,
    )

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "ler": [], "lr": []}
    best_acc = -1.0
    best_ler = float("inf")
    best_epoch = 0
    best_buffer = BytesIO()
    es_counter = 0
    optimizer_step = 0

    print(f"\n{'=' * 70}")
    print(f"  {model_name} training | {suffix}")
    print(f"  device={device} params={count_parameters(model):,} size={format_size(get_model_size(model))}")
    print(f"{'=' * 70}\n")

    for epoch in range(config.max_epochs):
        if epoch >= config.epochs and es_counter >= config.patience:
            break

        t0 = time.time()
        model.train()
        optimizer.zero_grad()
        train_loss = 0.0
        num_batches = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            if y.ndim == 1:
                y = y.unsqueeze(1)
            if config.label_smoothing > 0:
                y = y * (1 - config.label_smoothing) + 0.5 * config.label_smoothing

            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = criterion(logits, y) / config.accumulation_steps
            loss.backward()
            train_loss += loss.item() * config.accumulation_steps
            num_batches += 1

            if (batch_idx + 1) % config.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
                lr = lr_schedule.get_lr(optimizer_step)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                optimizer.zero_grad()
                optimizer_step += 1

        if num_batches % config.accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            lr = lr_schedule.get_lr(optimizer_step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad()
            optimizer_step += 1

        val_loss, val_acc, ler = _evaluate_logits_model(model, test_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        avg_train = train_loss / max(num_batches, 1)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["ler"].append(ler)
        history["lr"].append(current_lr)

        marker = ""
        if val_acc >= best_acc + config.min_delta:
            best_acc = val_acc
            best_ler = ler
            best_epoch = epoch + 1
            es_counter = 0
            best_buffer.seek(0)
            best_buffer.truncate()
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, best_buffer)
            marker = " [best]"
        else:
            es_counter += 1

        print(
            f"Epoch {epoch + 1:03d} | train={avg_train:.5f} val={val_loss:.5f} "
            f"acc={val_acc * 100:.2f}% ler={ler * 100:.3f}% lr={current_lr:.2e} "
            f"es={es_counter}/{config.patience} time={time.time() - t0:.1f}s{marker}"
        )

        if epoch + 1 >= config.epochs and es_counter >= config.patience:
            break

    model_path = None
    csv_path = None
    if config.save_dir is not None:
        os.makedirs(config.save_dir, exist_ok=True)
        model_path = os.path.join(config.save_dir, get_model_filename(model_name, config, suffix))
        best_buffer.seek(0)
        with open(model_path, "wb") as f:
            f.write(best_buffer.read())
    if config.log_dir is not None:
        csv_path = os.path.join(config.log_dir, f"{model_name}_{suffix}_{get_timestamp()}.csv")
        save_history_csv(history, csv_path)

    return {
        "best_acc": best_acc,
        "best_ler": best_ler,
        "best_epoch": best_epoch,
        "total_epochs": len(history["epoch"]),
        "model_path": model_path,
        "csv_path": csv_path,
        "history": history,
    }


def _soft_xor_prob(probs: torch.Tensor, logical_matrix_t: torch.Tensor) -> torch.Tensor:
    term = torch.clamp(1.0 - 2.0 * probs, min=-0.9999, max=0.9999)
    log_abs = torch.log(torch.clamp(torch.abs(term), min=1e-6))
    sign = torch.sign(term)
    sum_log = torch.matmul(log_abs, logical_matrix_t)
    neg_count = torch.matmul((sign < 0).float(), logical_matrix_t)
    parity = ((-1.0) ** torch.round(neg_count)) * torch.exp(sum_log)
    return torch.clamp(0.5 * (1.0 - parity), min=0.0, max=1.0)


def train_gnn_decoder(
    model: nn.Module,
    train_loader,
    test_loader,
    logical_matrix_t: torch.Tensor,
    config: TrainConfig,
    model_name: str = "gnn",
    syndrome_loss_weight: float = 0.5,
) -> dict:
    """Trains the neural BP GNN with logical and syndrome-consistency losses."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logical_matrix_t = logical_matrix_t.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.BCELoss()
    lr_schedule = CosineWarmupSchedule(
        peak_lr=config.lr,
        warmup_epochs=config.scheduler.warmup_epochs,
        decay_end_epoch=config.scheduler.decay_end_epoch,
        min_lr=config.scheduler.min_lr,
        steps_per_epoch=max(1, len(train_loader)),
    )
    history = {"epoch": [], "train_loss": [], "logical_loss": [], "syndrome_loss": [], "val_acc": [], "ler": [], "lr": []}

    best_acc = -1.0
    best_ler = float("inf")
    best_epoch = 0
    best_buffer = BytesIO()
    es_counter = 0

    for epoch in range(config.max_epochs):
        model.train()
        total_loss = 0.0
        total_logical = 0.0
        total_syndrome = 0.0
        batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if y.ndim == 1:
                y = y.unsqueeze(1)
            optimizer.zero_grad()
            lr = lr_schedule.get_lr(len(history["epoch"]) * max(1, len(train_loader)) + batches)
            for group in optimizer.param_groups:
                group["lr"] = lr
            physical_logits, syndrome_loss = model(x, compute_syn_loss=True)
            logical_prob = _soft_xor_prob(torch.sigmoid(physical_logits), logical_matrix_t)
            logical_loss = criterion(logical_prob, y)
            loss = logical_loss + syndrome_loss_weight * syndrome_loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            optimizer.step()

            total_loss += loss.item()
            total_logical += logical_loss.item()
            total_syndrome += syndrome_loss.mean().item()
            batches += 1

        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                if y.ndim == 1:
                    y = y.unsqueeze(1)
                physical_logits, _ = model(x, compute_syn_loss=False)
                physical_pred = (physical_logits > 0).float()
                logical_pred = torch.matmul(physical_pred, logical_matrix_t) % 2
                correct += (logical_pred == y).all(dim=1).sum().item()
                total += y.size(0)

        acc = correct / total if total else 0.0
        ler = 1.0 - acc
        current_lr = optimizer.param_groups[0]["lr"]
        avg_loss = total_loss / max(batches, 1)
        avg_logical = total_logical / max(batches, 1)
        avg_syndrome = total_syndrome / max(batches, 1)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["logical_loss"].append(avg_logical)
        history["syndrome_loss"].append(avg_syndrome)
        history["val_acc"].append(acc)
        history["ler"].append(ler)
        history["lr"].append(current_lr)

        if acc >= best_acc + config.min_delta:
            best_acc = acc
            best_ler = ler
            best_epoch = epoch + 1
            es_counter = 0
            best_buffer.seek(0)
            best_buffer.truncate()
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, best_buffer)
        else:
            es_counter += 1

        print(
            f"Epoch {epoch + 1:03d} | loss={avg_loss:.5f} logical={avg_logical:.5f} "
            f"syn={avg_syndrome:.5f} acc={acc * 100:.2f}% ler={ler * 100:.3f}% "
            f"lr={current_lr:.2e} es={es_counter}/{config.patience}"
        )
        if epoch + 1 >= config.epochs and es_counter >= config.patience:
            break

    model_path = None
    csv_path = None
    if config.save_dir is not None:
        os.makedirs(config.save_dir, exist_ok=True)
        model_path = os.path.join(config.save_dir, get_model_filename(model_name, config, "fp32"))
        best_buffer.seek(0)
        with open(model_path, "wb") as f:
            f.write(best_buffer.read())
    if config.log_dir is not None:
        csv_path = os.path.join(config.log_dir, f"{model_name}_fp32_{get_timestamp()}.csv")
        save_history_csv(history, csv_path)

    return {
        "best_acc": best_acc,
        "best_ler": best_ler,
        "best_epoch": best_epoch,
        "total_epochs": len(history["epoch"]),
        "model_path": model_path,
        "csv_path": csv_path,
        "history": history,
    }


def make_qat_config(**overrides) -> TrainConfig:
    """Default fine-tuning setup for W4A4 QAT experiments."""

    config = TrainConfig(
        epochs=50,
        max_epochs=50,
        batch_size=4096,
        accumulation_steps=1,
        lr=1e-4,
        patience=5,
        weight_decay=1e-3,
        scheduler=SchedulerConfig(warmup_epochs=0, decay_end_epoch=50, min_lr=1e-6),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_benchmark_config(model_name: str, **overrides) -> TrainConfig:
    """Default training hyperparameters from the neural decoder benchmark appendix."""

    name = model_name.lower()
    weight_decay = 1e-4 if name in {"mlp", "gnn"} else 1e-3
    config = TrainConfig(weight_decay=weight_decay)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_pruning_finetune_config(**overrides) -> TrainConfig:
    """Default fine-tuning setup after global magnitude pruning."""

    config = TrainConfig(
        epochs=30,
        max_epochs=30,
        batch_size=4096,
        accumulation_steps=1,
        lr=5e-5,
        patience=5,
        weight_decay=1e-3,
        label_smoothing=0.1,
        scheduler=SchedulerConfig(warmup_epochs=0, decay_end_epoch=30, min_lr=1e-6),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


# Short alias used by older local code.
train = train_decoder
