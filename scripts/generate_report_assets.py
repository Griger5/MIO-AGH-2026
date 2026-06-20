from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ecg_compression.data import CLASS_NAMES, nonzero_lengths


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "reports"
ASSETS_DIR = REPORT_DIR / "assets"


def latex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def write_table(path: Path, columns: list[str], rows: list[list[object]]) -> None:
    body = [" & ".join(latex_escape(item) for item in row) + r" \\" for row in rows]
    table = [
        r"\begin{tabular}{" + "l" * len(columns) + r"}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    path.write_text("\n".join(table), encoding="utf-8")


def load_dataset_summary() -> None:
    rows: list[list[object]] = []
    class_rows: list[list[object]] = []
    padding_rows: list[list[object]] = []

    for split, filename in [("treningowy", "mitbih_train.csv"), ("testowy", "mitbih_test.csv")]:
        frame = pd.read_csv(DATA_DIR / filename, header=None)
        x = frame.iloc[:, :187].to_numpy(dtype=np.float32)
        y = frame.iloc[:, 187].to_numpy(dtype=np.int64)
        lengths = nonzero_lengths(x)
        rows.append([split, frame.shape[0], frame.shape[1] - 1, 1])
        padding_rows.append(
            [
                split,
                f"{lengths.mean():.2f}",
                f"{lengths.std():.2f}",
                int(lengths.min()),
                int(np.median(lengths)),
                int(lengths.max()),
            ]
        )
        counts = pd.Series(y).value_counts().sort_index()
        for class_id, count in counts.items():
            percent = 100.0 * count / len(y)
            class_rows.append(
                [
                    split,
                    int(class_id),
                    CLASS_NAMES.get(int(class_id), str(class_id)),
                    int(count),
                    f"{percent:.2f}",
                ]
            )

    write_table(
        ASSETS_DIR / "dataset_shape_table.tex",
        ["Zbior", "Liczba rekordow", "Probki sygnalu", "Etykiety"],
        rows,
    )
    write_table(
        ASSETS_DIR / "class_distribution_table.tex",
        ["Zbior", "Klasa", "Nazwa", "Liczba", "Udzial [\\%]"],
        class_rows,
    )
    write_table(
        ASSETS_DIR / "padding_stats_table.tex",
        ["Zbior", "Srednia", "Odch. std.", "Min", "Mediana", "Max"],
        padding_rows,
    )

    sns.set_theme(style="whitegrid")
    dist = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": split,
                    "class": [CLASS_NAMES.get(int(v), str(int(v))) for v in pd.read_csv(DATA_DIR / filename, header=None).iloc[:, 187]],
                }
            )
            for split, filename in [("treningowy", "mitbih_train.csv"), ("testowy", "mitbih_test.csv")]
        ]
    )
    plt.figure(figsize=(8, 4.8))
    sns.countplot(data=dist, x="class", hue="split", order=["N", "S", "V", "F", "Q"])
    plt.xlabel("Klasa MIT-BIH")
    plt.ylabel("Liczba rekordow")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "class_distribution.png", dpi=180)
    plt.close()


def load_metrics_summary() -> None:
    metrics = pd.read_csv(RESULTS_DIR / "metrics.csv").sort_values(["bottleneck_dim", "method"])
    rows = []
    for _, row in metrics.iterrows():
        rows.append(
            [
                row["method"],
                int(row["bottleneck_dim"]),
                f"{row['compression_ratio']:.3f}",
                f"{row['prd']:.3f}",
                f"{row['prdn']:.3f}",
                f"{row['snr_db']:.3f}",
                f"{row['rmse']:.5f}",
                f"{row['mae']:.5f}",
                f"{row['quality_score']:.4f}",
            ]
        )
    write_table(
        ASSETS_DIR / "metrics_table.tex",
        ["Metoda", "$K$", "$CR$", "PRD", "PRDN", "SNR [dB]", "RMSE", "MAE", "QS"],
        rows,
    )


def load_training_summary() -> None:
    rows = []
    for path in sorted((RESULTS_DIR / "history").glob("history_k*.json"), key=lambda p: int(p.stem.split("_k")[-1])):
        k = int(path.stem.split("_k")[-1])
        history = json.loads(path.read_text(encoding="utf-8"))
        if not history:
            continue
        best = min(history, key=lambda item: item["val_loss"])
        rows.append([k, len(history), f"{best['train_loss']:.6f}", f"{best['val_loss']:.6f}", int(best["epoch"])])

    write_table(
        ASSETS_DIR / "training_table.tex",
        ["$K$", "Epoki", "Najl. MSE train", "Najl. MSE val", "Epoka"],
        rows,
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8.5, 5.2))
    for path in sorted((RESULTS_DIR / "history").glob("history_k*.json"), key=lambda p: int(p.stem.split("_k")[-1])):
        k = int(path.stem.split("_k")[-1])
        history = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
        if not history.empty:
            plt.plot(history["epoch"], history["val_loss"], label=f"K={k}")
    plt.yscale("log")
    plt.xlabel("Epoka")
    plt.ylabel("MSE walidacyjne")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "validation_loss.png", dpi=180)
    plt.close()


def copy_existing_figures() -> None:
    for path in (RESULTS_DIR / "figures").glob("*.png"):
        shutil.copy2(path, ASSETS_DIR / path.name)


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    load_dataset_summary()
    load_metrics_summary()
    load_training_summary()
    copy_existing_figures()
    print(f"Report assets saved to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
