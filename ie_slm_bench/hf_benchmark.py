from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ie_slm_bench.config import BENCHMARK
from ie_slm_bench.metrics import safe_model_filename


def build_results_payload(summary: pd.DataFrame) -> dict:
    results: dict[str, dict[str, float]] = {}
    for _, row in summary.iterrows():
        model_key = safe_model_filename(row["model_id"])
        results[model_key] = {
            f"strict_exact_match,{BENCHMARK}": float(row["strict_exact_match"]),
            f"field_precision,{BENCHMARK}": float(row["field_precision"]),
            f"field_recall,{BENCHMARK}": float(row["field_recall"]),
            f"field_f1,{BENCHMARK}": float(row["field_f1"]),
            f"null_field_accuracy,{BENCHMARK}": float(row["null_field_accuracy"]),
            f"hallucination_rate,{BENCHMARK}": float(row["hallucination_rate"]),
            f"schema_validity_rate,{BENCHMARK}": float(row["schema_validity_rate"]),
            f"entity_f1,{BENCHMARK}": float(row["entity_f1"]),
        }
    return {
        "benchmark": BENCHMARK,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "summary": summary.to_dict(orient="records"),
    }


def load_summary_frame(run_dir: Path) -> pd.DataFrame:
    summary_frames = []
    for path in run_dir.glob("metrics_summary_*.csv"):
        summary_frames.append(pd.read_csv(path))
    if not summary_frames:
        raise FileNotFoundError(f"No metrics_summary_*.csv in {run_dir}")
    return pd.concat(summary_frames, ignore_index=True)


def stage_benchmark_tree(staging_root: Path, run_dir: Path, results_root: Path) -> None:
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)

    benchmark_dir = staging_root / "benchmark"
    runs_dir = benchmark_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for path in run_dir.glob("metrics_*"):
        (runs_dir / path.name).write_bytes(path.read_bytes())

    metrics_path = results_root / "metrics.json"
    if metrics_path.exists():
        (benchmark_dir / "metrics.json").write_bytes(metrics_path.read_bytes())

    analysis_path = results_root / "analysis.json"
    if analysis_path.exists():
        (benchmark_dir / "analysis.json").write_bytes(analysis_path.read_bytes())

    summary = load_summary_frame(run_dir)
    summary.to_csv(benchmark_dir / "summary.csv", index=False)
    (benchmark_dir / "results.json").write_text(
        json.dumps(build_results_payload(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    assets_src = results_root / "assets"
    assets_dest = staging_root / "benchmark_assets"
    assets_dest.mkdir(parents=True, exist_ok=True)
    if assets_src.exists():
        for png in assets_src.glob("*.png"):
            (assets_dest / png.name).write_bytes(png.read_bytes())


def iter_staged_files(staging_root: Path) -> list[tuple[str, bytes]]:
    upload_paths: list[tuple[str, bytes]] = []
    for path in sorted(staging_root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(staging_root).as_posix()
            upload_paths.append((rel, path.read_bytes()))
    return upload_paths
