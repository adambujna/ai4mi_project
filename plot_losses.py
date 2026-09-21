#!/usr/bin/env python3

"""Plot validation-loss curves for the available loss functions."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LOSS_FOLDERS = {
    "CE": "ce",
    "Dice": "dice",
    "Dice + CE": "dicece",
    "Dice + TopK CE": "DiceTopK",
    "Dice + Focal": "DiceFocal",
}


def load_validation_losses(results_dir: Path) -> dict[str, np.ndarray]:
    """Load and average validation-batch losses for every epoch."""
    losses = {}
    missing = []

    for label, folder in LOSS_FOLDERS.items():
        loss_file = results_dir / folder / "loss_val.npy"
        if not loss_file.is_file():
            missing.append(str(loss_file))
            continue

        values = np.asarray(np.load(loss_file), dtype=float)
        if values.ndim != 2:
            raise ValueError(f"Expected {loss_file} to have shape (epochs, batches), got {values.shape}")
        losses[label] = values.mean(axis=1)

    if missing:
        missing_files = "\n".join(f"  - {file}" for file in missing)
        raise FileNotFoundError(f"Missing validation-loss files:\n{missing_files}")

    return losses


def plot_validation_losses(losses: dict[str, np.ndarray], destination: Path | None,
                           show: bool = False) -> None:
    """Create the first comparison plot; additional plots can be added here later."""
    figure, axis = plt.subplots(figsize=(9, 5))
    for label, values in losses.items():
        axis.plot(np.arange(1, len(values) + 1), values, linewidth=2, label=label)

    axis.set_title("TOY2 validation loss by loss function")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean validation loss")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()

    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150)
    if show:
        plt.show()
    plt.close(figure)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TOY2 validation losses.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/toy2"),
                        help="Directory containing one result folder per loss function.")
    parser.add_argument("--dest", type=Path, default=Path("results/toy2/loss_comparison.png"),
                        help="Output image path. Use --show to display instead.")
    parser.add_argument("--show", action="store_true", help="Display the plot after saving it.")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    losses = load_validation_losses(args.results_dir)
    plot_validation_losses(losses, args.dest, show=args.show)


if __name__ == "__main__":
    main()