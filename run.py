"""Command-line entry point. Run ``python run.py --help`` for the list of commands.

Checks that use the saved models and results (minutes):
    verify     rebuild the features, re-score the test set with the saved models, compare with Table 1
    figures    redraw Figs 2-4, Extended Data Figs 1-2 and Table 1 from the saved results
    prenatal   apply the saved scorer to the public prenatal cohort and draw Fig. 5
    score      score your own HPO profiles with the saved scorer

Full retraining (hours):
    train-benchmark   rerun the whole benchmark (about 18 h on one workstation)
    train-scorer      retrain the disease-level scorer
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
INPUTS = DATA / "inputs"
PUBLISHED = ROOT / "published_run"
CHECKSUMS = DATA / "checksums.json"


def prepare_inputs() -> Path:
    """Unpack the training tables and ontologies into data/inputs and check their SHA-256."""
    expected = json.loads(CHECKSUMS.read_text())
    for archive in ("training_tables.zip", "ontologies.zip"):
        with zipfile.ZipFile(DATA / archive) as bundle:
            for name in bundle.namelist():
                if not (INPUTS / name).exists():
                    bundle.extract(name, INPUTS)
    for name, checksum in expected.items():
        digest = hashlib.sha256((INPUTS / name).read_bytes()).hexdigest()
        if digest != checksum:
            raise SystemExit(f"Checksum mismatch for data/inputs/{name}")
    return INPUTS


def published_models() -> Path:
    """Decompress the saved model bundles (models/*.pkl.xz) once."""
    models = PUBLISHED / "benchmark" / "models"
    for packed in sorted(models.glob("*.pkl.xz")):
        target = packed.with_suffix("")
        if not target.exists():
            with lzma.open(packed) as source, open(target, "wb") as sink:
                shutil.copyfileobj(source, sink)
    return models


def fresh(path: str) -> str:
    if Path(path).exists() and any(Path(path).iterdir()):
        raise SystemExit(f"{path} is not empty. Choose a new output folder.")
    return path


def cmd_verify(args) -> None:
    from severity.benchmark.verify import run_verification

    inputs = prepare_inputs()
    published_models()
    report = run_verification(str(inputs), str(PUBLISHED / "benchmark"), fresh(args.output))
    for key, model in report["models"].items():
        print(f"{key}: probabilities match saved predictions: {model['probabilities_match_saved']}")
    print(f"feature tables identical: {all(report['feature_tables_identical'].values())}")
    print(
        f"split matches: {all(v['matches_saved_split'] for v in report['split'].values())}; "
        f"Table 1 metrics matching: {report['metrics_matching']}/{report['metrics_checked']}"
    )
    print("PASSED" if report["passed"] else "FAILED")
    if not report["passed"]:
        sys.exit(1)


def cmd_figures(args) -> None:
    from severity.figures.render import render_saved_run

    inputs = prepare_inputs()
    out = render_saved_run(
        str(PUBLISHED / "benchmark"), str(inputs / "gene_disease_mapping_HPO_rebuilt.csv"), fresh(args.output)
    )
    print(f"Figures written to {out / 'main'}; panel data in {out / 'figure_data'}")


def cmd_prenatal(args) -> None:
    from severity.prenatal.analysis import run

    statistics = run(
        args.cohort,
        str(PUBLISHED / "scorer" / "model.pkl"),
        str(PUBLISHED / "scorer" / "held_out_test_predictions.csv"),
        fresh(args.output),
        n_boot=args.n_boot,
        n_perm=args.n_perm,
    )
    print(f"Families analysed: {statistics['audit']['n_analyzed']}")


def cmd_score(args) -> None:
    import pandas as pd

    from severity.scorer import load_scorer

    scorer = load_scorer(str(PUBLISHED / "scorer" / "model.pkl"))
    profiles = pd.read_csv(args.input)
    rows = []
    for sample, terms in zip(profiles["sample_id"], profiles["hpo_terms"]):
        result = scorer.score_profile([t.strip() for t in str(terms).split(";") if t.strip()])
        expected = result.get("phenotype_only_expected_severity_0_3")
        probabilities = result.get("phenotype_only_class_probabilities") or {}
        rows.append(
            {
                "sample_id": sample,
                "status": result["status"],
                "severity_score_0_1": None if expected is None else expected / 3.0,
                "predicted_class": max(probabilities, key=probabilities.get) if probabilities else None,
                **{f"probability_{name.lower()}": value for name, value in probabilities.items()},
                "warnings": " | ".join(result.get("warnings", [])),
            }
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Scores written to {args.output}")


def cmd_train_benchmark(args) -> None:
    from severity.benchmark import pipeline
    from severity.benchmark.config import Config

    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit(
            "Set PYTHONHASHSEED=0 before training, for example: PYTHONHASHSEED=0 python run.py train-benchmark"
        )
    Config.configure(str(prepare_inputs()), fresh(args.output))
    pipeline.main()


def cmd_train_scorer(args) -> None:
    from severity.scorer.model import main

    inputs = prepare_inputs()
    main(
        [
            "--severity-csv",
            str(inputs / "final_output_with_severity.csv"),
            "--hpo-obo",
            str(inputs / "hp.obo"),
            "--output",
            fresh(args.output),
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("verify", help="re-score the test set with the saved models and compare with Table 1")
    p.add_argument("--output", default="outputs/verify")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("figures", help="redraw the benchmark figures from the saved results")
    p.add_argument("--output", default="outputs/figures")
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("prenatal", help="prenatal analysis with the saved scorer (Fig. 5)")
    p.add_argument("--cohort", default=str(DATA / "prenatal_cohort_public.csv"))
    p.add_argument("--output", default="outputs/prenatal")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--n-perm", type=int, default=2000)
    p.set_defaults(func=cmd_prenatal)

    p = sub.add_parser("score", help="score HPO profiles (columns sample_id, hpo_terms separated by ';')")
    p.add_argument("--input", default=str(DATA / "example_profiles.csv"))
    p.add_argument("--output", default="outputs/scores.csv")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("train-benchmark", help="rerun the full benchmark (about 18 h)")
    p.add_argument("--output", default="outputs/benchmark")
    p.set_defaults(func=cmd_train_benchmark)

    p = sub.add_parser("train-scorer", help="retrain the disease-level scorer")
    p.add_argument("--output", default="outputs/scorer")
    p.set_defaults(func=cmd_train_scorer)
    return parser


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(ROOT))
    arguments = build_parser().parse_args()
    arguments.func(arguments)
