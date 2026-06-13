from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import subprocess
from pathlib import Path

from ie_slm_bench.config import RUN_DIR


def git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git"] + args, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=RUN_DIR,
    )
    parser.add_argument(
        "--message",
        type=str,
        default="Colab: IE SLM benchmark results",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()

    repo_root = args.repo_root
    results_root = repo_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    run_dest = results_root / "run"
    assets_dest = results_root / "assets"
    run_dest.mkdir(parents=True, exist_ok=True)
    assets_dest.mkdir(parents=True, exist_ok=True)

    for src in args.run_dir.rglob("*.csv"):
        relative = src.relative_to(args.run_dir)
        destination = run_dest / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(src.read_bytes())

    assets_src = args.run_dir.parent / "assets"
    if assets_src.exists():
        for src in assets_src.glob("*"):
            if src.is_file():
                (assets_dest / src.name).write_bytes(src.read_bytes())

    metrics_src = args.run_dir.parent / "metrics.json"
    if metrics_src.exists():
        (results_root / "metrics.json").write_bytes(metrics_src.read_bytes())

    git(["add", "results/"], repo_root)
    git(["commit", "-m", args.message], repo_root)
    git(["push", "origin", "HEAD"], repo_root)
    print("Pushed results to GitHub")


if __name__ == "__main__":
    main()
