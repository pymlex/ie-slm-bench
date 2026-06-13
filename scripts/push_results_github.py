from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
import subprocess
from pathlib import Path

from ie_slm_bench.config import RUN_DIR


def git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git"] + args, cwd=cwd, check=check, text=True, capture_output=True)


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
    github_name = os.environ.get("GITHUB_NAME")
    github_email = os.environ.get("GITHUB_EMAIL")
    if github_name:
        git(["config", "user.name", github_name], repo_root)
    if github_email:
        git(["config", "user.email", github_email], repo_root)

    results_root = repo_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    run_dir = args.run_dir.resolve()
    run_dest = (results_root / "run").resolve()
    assets_dest = results_root / "assets"
    run_dest.mkdir(parents=True, exist_ok=True)
    assets_dest.mkdir(parents=True, exist_ok=True)

    if run_dir != run_dest:
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
    else:
        print(f"Results already in {run_dest}, skipping copy")

    git(["add", "results/"], repo_root)
    status = git(["status", "--porcelain", "results/"], repo_root, check=False)
    if not status.stdout.strip():
        print("No changes in results/ to commit")
        return

    commit = git(["commit", "-m", args.message], repo_root, check=False)
    if commit.returncode != 0:
        print(commit.stderr.strip() or commit.stdout.strip())
        return

    git(["push", "origin", "HEAD"], repo_root)
    print("Pushed results to GitHub")


if __name__ == "__main__":
    main()
