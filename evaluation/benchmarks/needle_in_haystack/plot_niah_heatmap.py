# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from fire import Fire


@dataclass
class RunRecord:
    run_dir: Path
    context_length: int
    model: str
    press_name: str
    compression_ratio: float


def _discover_runs(
    results_root: str,
    model: str | None,
    press_name: str | None,
    compression_ratio: float | None,
) -> list[RunRecord]:
    root = Path(results_root)
    if not root.exists():
        raise FileNotFoundError(f"Results directory does not exist: {root}")

    runs: list[RunRecord] = []
    for config_path in root.glob("**/config.yaml"):
        run_dir = config_path.parent
        predictions_path = run_dir / "predictions.csv"
        if not predictions_path.exists():
            continue

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if config.get("dataset") != "needle_in_haystack":
            continue

        ctx_len = config.get("max_context_length")
        if ctx_len is None:
            continue

        rec = RunRecord(
            run_dir=run_dir,
            context_length=int(ctx_len),
            model=str(config.get("model")),
            press_name=str(config.get("press_name")),
            compression_ratio=float(config.get("compression_ratio")),
        )

        if model is not None and rec.model != model:
            continue
        if press_name is not None and rec.press_name != press_name:
            continue
        if compression_ratio is not None and abs(rec.compression_ratio - compression_ratio) > 1e-9:
            continue

        runs.append(rec)

    runs.sort(key=lambda x: x.context_length)
    return runs


def _score_predictions(predictions_path: Path, metric: str) -> pd.DataFrame:
    from rouge import Rouge

    df = pd.read_csv(predictions_path)
    needed_cols = {"needle", "predicted_answer", "needle_depth"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {predictions_path}: {sorted(missing)}")

    if metric not in {"rouge-1-f", "rouge-2-f", "rouge-l-f"}:
        raise ValueError("metric must be one of: rouge-1-f, rouge-2-f, rouge-l-f")

    rouge_key = metric.replace("-f", "")
    scorer = Rouge()
    scores = []
    for _, row in df.iterrows():
        needle = str(row["needle"]).strip()
        predicted = str(row["predicted_answer"]).strip()
        score = scorer.get_scores(needle, predicted)[0][rouge_key]["f"]
        scores.append(score)

    scored_df = df.copy()
    scored_df["score"] = scores
    scored_df["needle_depth"] = scored_df["needle_depth"].astype(float)
    return scored_df


def plot(
    results_root: str = "evaluation/results",
    model: str | None = None,
    press_name: str | None = None,
    compression_ratio: float | None = None,
    metric: str = "rouge-l-f",
    aggregation: str = "mean",
    title: str | None = None,
    output_png: str = "evaluation/benchmarks/needle_in_haystack/niah_heatmap.png",
):
    """Create a NIAH heatmap from KVPress evaluation output folders."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install it with `uv sync --extra dev` or `pip install matplotlib`."
        ) from exc

    runs = _discover_runs(results_root, model, press_name, compression_ratio)
    if not runs:
        raise ValueError(
            "No matching needle_in_haystack runs found. "
            "Please run evaluation first and/or relax filters (model, press_name, compression_ratio)."
        )

    frames = []
    for run in runs:
        scored_df = _score_predictions(run.run_dir / "predictions.csv", metric)
        scored_df["context_length"] = run.context_length
        frames.append(scored_df[["context_length", "needle_depth", "score"]])

    all_scores = pd.concat(frames, axis=0, ignore_index=True)

    if aggregation not in {"mean", "median", "min", "max"}:
        raise ValueError("aggregation must be one of: mean, median, min, max")

    pivot = (
        all_scores.groupby(["needle_depth", "context_length"])["score"]
        .agg(aggregation)
        .reset_index()
        .pivot(index="needle_depth", columns="context_length", values="score")
        .sort_index(ascending=True)
    )

    x_vals = list(pivot.columns.astype(int))
    y_vals = list(pivot.index.astype(float))
    z_vals = pivot.to_numpy(dtype=float)

    plt.figure(figsize=(11, 6))
    im = plt.imshow(z_vals, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0, origin="upper")
    plt.colorbar(im, label=f"{metric} ({aggregation})")

    plt.xticks(range(len(x_vals)), [str(v) for v in x_vals], rotation=35, ha="right")
    plt.yticks(range(len(y_vals)), [f"{v:.1f}" for v in y_vals])
    plt.xlabel("Token Limit (max_context_length)")
    plt.ylabel("Depth Percent")

    if title is None:
        title = "Needle in a Haystack: Retrieval Across Context Lengths"
    plt.title(title)

    out = Path(output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()

    print(f"Saved heatmap to: {out}")
    print(f"Included {len(runs)} run(s):")
    for run in runs:
        print(f"  - {run.run_dir} (context={run.context_length})")


if __name__ == "__main__":
    Fire({"plot": plot})
