from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from ecg_compression.data import SIGNAL_LENGTH, load_splits, make_loader
from ecg_compression.model import MLPAutoencoder


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build and run on a machine with a visible GPU, "
            "or use --device auto/--device cpu."
        )
    return torch.device(requested_device)


def train_one_model(
    x_train: np.ndarray,
    x_val: np.ndarray,
    bottleneck_dim: int,
    results_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    seed: int,
    device: str = "auto",
    num_workers: int = 0,
    amp: bool = False,
    hidden_dims: tuple[int, ...] = (128, 64),
) -> dict[str, object]:
    torch.manual_seed(seed)
    training_device = resolve_device(device)
    if training_device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    use_amp = amp and training_device.type == "cuda"
    model = MLPAutoencoder(SIGNAL_LENGTH, bottleneck_dim, hidden_dims).to(training_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    train_loader = make_loader(
        x_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=training_device.type == "cuda",
    )
    val_loader = make_loader(
        x_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=training_device.type == "cuda",
    )
    scaler = GradScaler(device=training_device.type, enabled=use_amp)

    best_val_loss = float("inf")
    best_state = None
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in tqdm(range(1, epochs + 1), desc=f"K={bottleneck_dim}", leave=False):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(training_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=training_device.type, enabled=use_amp):
                reconstruction, _ = model(batch_x)
                loss = loss_fn(reconstruction, batch_x)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.item()) * len(batch_x)
            train_count += len(batch_x)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(training_device, non_blocking=True)
                with autocast(device_type=training_device.type, enabled=use_amp):
                    reconstruction, _ = model(batch_x)
                    loss = loss_fn(reconstruction, batch_x)
                val_loss_sum += float(loss.item()) * len(batch_x)
                val_count += len(batch_x)

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model_dir = results_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / f"mlp_autoencoder_k{bottleneck_dim}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "bottleneck_dim": bottleneck_dim,
            "input_dim": SIGNAL_LENGTH,
            "best_val_loss": best_val_loss,
            "history": history,
            "device": str(training_device),
            "amp": use_amp,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "hidden_dims": list(hidden_dims),
        },
        checkpoint_path,
    )

    history_path = results_dir / "history"
    history_path.mkdir(parents=True, exist_ok=True)
    (history_path / f"history_k{bottleneck_dim}.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    return {
        "bottleneck_dim": bottleneck_dim,
        "best_val_loss": best_val_loss,
        "epochs_ran": len(history),
        "checkpoint": str(checkpoint_path),
        "device": str(training_device),
        "amp": use_amp,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "hidden_dims": list(hidden_dims),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MLP ECG autoencoders.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--bottlenecks", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 96])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device. Use 'cuda' to require GPU, 'auto' to use GPU when available.",
    )
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke runs.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    splits = load_splits(args.data_dir, val_size=args.val_size, seed=args.seed, limit=args.limit)
    hidden_dims = tuple(args.hidden_dims)

    summaries = []
    for bottleneck_dim in args.bottlenecks:
        summaries.append(
            train_one_model(
                splits.x_train,
                splits.x_val,
                bottleneck_dim,
                args.results_dir,
                args.epochs,
                args.batch_size,
                args.learning_rate,
                args.patience,
                args.seed,
                args.device,
                args.num_workers,
                args.amp,
                hidden_dims,
            )
        )

    summary_path = args.results_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Saved training summary to {summary_path}")


if __name__ == "__main__":
    main()
