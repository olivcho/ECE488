"""Run classical ADMM once per grid point, each as its own run folder, then tabulate every finished run."""
import argparse
import csv
import itertools
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_admm import apply_overrides, run  # noqa: E402

SUMMARY_NAME = "summary.csv"


def grid_points(sweep: dict) -> list[dict[str, object]]:
    """One setting per (tau, mu). The 4-image probe tied mu1 = mu2 = mu3, so the sweep keeps them tied."""
    return [
        {"admm.tau": tau, "admm.mu1": mu, "admm.mu2": mu, "admm.mu3": mu}
        for tau, mu in itertools.product(sweep["tau"], sweep["mu"])
    ]


def finished_runs(experiment_dir: Path) -> list[Path]:
    if not experiment_dir.exists():
        return []
    return sorted(path for path in experiment_dir.iterdir() if (path / "metrics.csv").exists())


def already_done(config: dict, experiment_dir: Path) -> bool:
    """A finished run with the same solver and data settings counts, so a Colab reconnect resumes the sweep."""
    for run_dir in finished_runs(experiment_dir):
        previous = yaml.safe_load((run_dir / "config.yaml").read_text())
        if previous["admm"] == config["admm"] and previous["data"] == config["data"]:
            return True
    return False


def summarize(experiment_dir: Path) -> list[dict[str, object]]:
    """Every scored iteration of every finished run, labelled with the setting that produced it."""
    rows = []
    for run_dir in finished_runs(experiment_dir):
        admm = yaml.safe_load((run_dir / "config.yaml").read_text())["admm"]
        for metrics in csv.DictReader((run_dir / "metrics.csv").open()):
            rows.append({
                "run": run_dir.name,
                "tau": admm["tau"], "mu1": admm["mu1"], "mu2": admm["mu2"], "mu3": admm["mu3"],
                "iteration": int(metrics.pop("iteration")),
                **{key: float(value) for key, value in metrics.items()},
            })
    return rows


def write_summary(experiment_dir: Path) -> Path:
    """Regenerated from the run folders each time, so it is derived data rather than a run of its own."""
    rows = summarize(experiment_dir)
    path = experiment_dir / SUMMARY_NAME
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def sweep(config: dict, experiments_root: str | None = None) -> Path:
    config = dict(config)
    points = grid_points(config.pop("sweep"))
    experiment_dir = Path(experiments_root or config["experiment"]["root"]) / config["experiment"]["name"]
    for index, point in enumerate(points, start=1):
        setting = apply_overrides(config, point)
        label = ", ".join(f"{key.split('.')[-1]}={value}" for key, value in point.items())
        if already_done(setting, experiment_dir):
            print(f"[{index}/{len(points)}] skip, already finished: {label}", flush=True)
            continue
        print(f"[{index}/{len(points)}] {label}", flush=True)
        run(setting, experiments_root)
    return write_summary(experiment_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiments-root", help="Create run folders here, such as a Google Drive folder")
    parser.add_argument("--summary-only", action="store_true", help="Only rebuild summary.csv from the finished runs")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.summary_only:
        experiment_dir = Path(args.experiments_root or config["experiment"]["root"]) / config["experiment"]["name"]
        print(write_summary(experiment_dir))
    else:
        print(sweep(config, args.experiments_root))


if __name__ == "__main__":
    main()
