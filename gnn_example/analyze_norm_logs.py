import re
import glob
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


LOG_PATTERN = "*.txt"

# Expected repeated block in each log:
# Average Epoch Time 0.2787
# Train acc 0.5994
# Test acc 0.5893
BLOCK_RE = re.compile(
    r"Average Epoch Time\s+([0-9]*\.?[0-9]+)\s*"
    r"Train acc\s+([0-9]*\.?[0-9]+)\s*"
    r"Test acc\s+([0-9]*\.?[0-9]+)",
    re.MULTILINE
)

# Optional metadata from the Namespace(...) line
NAMESPACE_RE = re.compile(r"Namespace\((.*?)\)", re.DOTALL)


def parse_filename(path: str):
    """
    Expects names like:
      bn_0.txt, gn_2.txt, gsgn_1.txt
    """
    name = Path(path).stem
    m = re.match(r"([A-Za-z0-9]+)_(\d+)$", name)
    if not m:
        raise ValueError(
            f"Filename '{name}' does not match expected pattern like bn_0.txt"
        )
    norm = m.group(1).lower()
    fold = int(m.group(2))
    return norm, fold


def parse_namespace(text: str):
    """
    Very light parser for the Namespace(...) header.
    Only used for a few fields if present.
    """
    m = NAMESPACE_RE.search(text)
    if not m:
        return {}
    raw = m.group(1)

    info = {}
    for key in ["dataset", "model", "epoch", "fold_idx", "norm_type", "exp"]:
        km = re.search(rf"{key}=('.*?'|[^\s,]+)", raw)
        if km:
            val = km.group(1).strip()
            if val.startswith("'") and val.endswith("'"):
                val = val[1:-1]
            info[key] = val
    return info


def parse_log_file(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    norm, fold = parse_filename(path)
    header = parse_namespace(text)
    matches = BLOCK_RE.findall(text)

    if not matches:
        raise ValueError(f"No epoch blocks found in {path}")

    rows = []
    for i, (avg_epoch_time, train_acc, test_acc) in enumerate(matches, start=1):
        rows.append(
            {
                "file": os.path.basename(path),
                "norm": norm,
                "fold": fold,
                "epoch": i,
                "avg_epoch_time": float(avg_epoch_time),
                "train_acc": float(train_acc),
                "test_acc": float(test_acc),
                "dataset": header.get("dataset"),
                "model": header.get("model"),
                "declared_epochs": int(header["epoch"]) if "epoch" in header and str(header["epoch"]).isdigit() else None,
                "norm_type_header": header.get("norm_type"),
                "exp": header.get("exp"),
            }
        )
    return pd.DataFrame(rows)


def load_all_logs(pattern=LOG_PATTERN):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching {pattern}")

    dfs = []
    for file in files:
        dfs.append(parse_log_file(file))
    df = pd.concat(dfs, ignore_index=True)
    return df


def make_mean_curve(df: pd.DataFrame, metric: str):
    """
    Mean/std across folds for each norm and epoch.
    """
    grouped = (
        df.groupby(["norm", "epoch"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["norm", "epoch"])
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["lower"] = grouped["mean"] - grouped["std"]
    grouped["upper"] = grouped["mean"] + grouped["std"]
    return grouped


def save_curve_plot(df: pd.DataFrame, metric: str, ylabel: str, title: str, out_path: str):
    curve_df = make_mean_curve(df, metric)

    plt.figure(figsize=(9, 6))

    for norm in sorted(curve_df["norm"].unique()):
        part = curve_df[curve_df["norm"] == norm]
        plt.plot(part["epoch"], part["mean"], label=norm.upper())
        plt.fill_between(part["epoch"], part["lower"], part["upper"], alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_per_fold_plot(df: pd.DataFrame, metric: str, ylabel: str, title: str, out_path: str):
    plt.figure(figsize=(9, 6))

    for norm in sorted(df["norm"].unique()):
        norm_df = df[df["norm"] == norm]
        for fold in sorted(norm_df["fold"].unique()):
            part = norm_df[norm_df["fold"] == fold].sort_values("epoch")
            plt.plot(part["epoch"], part[metric], alpha=0.35)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_summary(df: pd.DataFrame):
    # final epoch row per file
    final_df = (
        df.sort_values(["file", "epoch"])
        .groupby("file", as_index=False)
        .tail(1)
        .copy()
    )

    # best test epoch per file
    idx = df.groupby("file")["test_acc"].idxmax()
    best_df = df.loc[idx, ["file", "norm", "fold", "epoch", "test_acc"]].copy()
    best_df = best_df.rename(columns={"epoch": "best_test_epoch", "test_acc": "best_test_acc"})

    merged = final_df.merge(best_df, on=["file", "norm", "fold"], how="left")

    # summary per norm
    summary = (
        merged.groupby("norm")
        .agg(
            folds=("fold", "count"),
            epochs_run=("epoch", "mean"),
            final_train_acc_mean=("train_acc", "mean"),
            final_train_acc_std=("train_acc", "std"),
            final_test_acc_mean=("test_acc", "mean"),
            final_test_acc_std=("test_acc", "std"),
            best_test_acc_mean=("best_test_acc", "mean"),
            best_test_acc_std=("best_test_acc", "std"),
            best_test_epoch_mean=("best_test_epoch", "mean"),
            avg_epoch_time_mean=("avg_epoch_time", "mean"),
            avg_epoch_time_std=("avg_epoch_time", "std"),
        )
        .reset_index()
        .sort_values("final_test_acc_mean", ascending=False)
    )

    summary["final_train_acc_std"] = summary["final_train_acc_std"].fillna(0.0)
    summary["final_test_acc_std"] = summary["final_test_acc_std"].fillna(0.0)
    summary["best_test_acc_std"] = summary["best_test_acc_std"].fillna(0.0)
    summary["avg_epoch_time_std"] = summary["avg_epoch_time_std"].fillna(0.0)

    # optional whole-trajectory mean over all epochs and folds
    trajectory = (
        df.groupby("norm")
        .agg(
            mean_train_acc_over_all_epochs=("train_acc", "mean"),
            mean_test_acc_over_all_epochs=("test_acc", "mean"),
        )
        .reset_index()
    )

    summary = summary.merge(trajectory, on="norm", how="left")
    return summary, merged


def generate_comments(summary: pd.DataFrame):
    comments = []

    if summary.empty:
        return comments

    best_final = summary.iloc[0]
    fastest = summary.sort_values("avg_epoch_time_mean").iloc[0]
    best_peak = summary.sort_values("best_test_acc_mean", ascending=False).iloc[0]

    comments.append(
        f"Highest mean final test accuracy: {best_final['norm'].upper()} "
        f"({best_final['final_test_acc_mean']:.4f} ± {best_final['final_test_acc_std']:.4f})."
    )

    comments.append(
        f"Fastest average epoch time: {fastest['norm'].upper()} "
        f"({fastest['avg_epoch_time_mean']:.4f}s ± {fastest['avg_epoch_time_std']:.4f}s)."
    )

    comments.append(
        f"Highest mean best test accuracy during training: {best_peak['norm'].upper()} "
        f"({best_peak['best_test_acc_mean']:.4f} ± {best_peak['best_test_acc_std']:.4f}), "
        f"reached on average around epoch {best_peak['best_test_epoch_mean']:.1f}."
    )

    # overfitting-style comment
    gap_df = summary.copy()
    gap_df["generalization_gap"] = gap_df["final_train_acc_mean"] - gap_df["final_test_acc_mean"]
    worst_gap = gap_df.sort_values("generalization_gap", ascending=False).iloc[0]
    comments.append(
        f"Largest mean train-test gap at the final epoch: {worst_gap['norm'].upper()} "
        f"({worst_gap['generalization_gap']:.4f}), which may suggest relatively more overfitting."
    )

    return comments


def main():
    df = load_all_logs(LOG_PATTERN)

    print("\nParsed files:")
    for f in sorted(df["file"].unique()):
        part = df[df["file"] == f]
        print(f"  {f}: {len(part)} epochs")

    # Save parsed epoch-wise data
    df.to_csv("parsed_epoch_metrics.csv", index=False)

    # Save plots
    save_curve_plot(
        df,
        metric="train_acc",
        ylabel="Train Accuracy",
        title="GIN on PROTEINS: Mean Train Accuracy vs Epoch (across folds)",
        out_path="train_accuracy_mean.png",
    )

    save_curve_plot(
        df,
        metric="test_acc",
        ylabel="Test Accuracy",
        title="GIN on PROTEINS: Mean Test Accuracy vs Epoch (across folds)",
        out_path="test_accuracy_mean.png",
    )

    save_per_fold_plot(
        df,
        metric="train_acc",
        ylabel="Train Accuracy",
        title="GIN on PROTEINS: Per-Fold Train Accuracy Curves",
        out_path="train_accuracy_per_fold.png",
    )

    save_per_fold_plot(
        df,
        metric="test_acc",
        ylabel="Test Accuracy",
        title="GIN on PROTEINS: Per-Fold Test Accuracy Curves",
        out_path="test_accuracy_per_fold.png",
    )

    summary, final_rows = build_summary(df)
    summary.to_csv("norm_summary.csv", index=False)
    final_rows.to_csv("final_and_best_per_run.csv", index=False)

    print("\nSummary by norm:")
    print(summary.to_string(index=False))

    comments = generate_comments(summary)
    print("\nAutomatic comments:")
    for c in comments:
        print(f"- {c}")

    # Also save comments to a text file
    with open("analysis_comments.txt", "w", encoding="utf-8") as f:
        for c in comments:
            f.write(f"- {c}\n")

    print("\nSaved files:")
    print("  parsed_epoch_metrics.csv")
    print("  final_and_best_per_run.csv")
    print("  norm_summary.csv")
    print("  train_accuracy_mean.png")
    print("  test_accuracy_mean.png")
    print("  train_accuracy_per_fold.png")
    print("  test_accuracy_per_fold.png")
    print("  analysis_comments.txt")


if __name__ == "__main__":
    main()
