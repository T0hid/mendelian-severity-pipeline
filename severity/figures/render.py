"""Redraw the benchmark figures from a completed run without retraining.

The run directory must hold what ``severity.benchmark.pipeline`` writes: ``results/``,
``per_disease_explanations/severity_shap_*.npz``, ``figures/publication/figure_data``
(validation metrics of all candidates and the S3 PCA projection), the feature label
dictionary and ``pipeline_manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import panels  # noqa: E402

MOI_KEYS = (
    ("feat_is_dominant", "MOI_DOMINANT"),
    ("feat_is_recessive", "MOI_RECESSIVE"),
    ("feat_is_xlinked", "MOI_XLINKED"),
)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def load_cohort(run: Path, mapping_path: Path, config: dict) -> Tuple[SimpleNamespace, Dict]:
    """Rebuild the pair table and the primary split from the saved split identifiers."""
    split = _read_csv(run / "results" / "split_identifiers_primary.csv").sort_values("row_index")
    split = split.reset_index(drop=True)
    keys = ["gene_symbol", "mondo_id"]
    mapping = pd.read_csv(
        mapping_path, usecols=keys + ["severity", "moi_title", "disease_name"]
    ).drop_duplicates()
    cohort = split.merge(mapping, on=keys, how="left", validate="one_to_one")
    class_ids = cohort["severity"].str.title().map(panels.CLASS_INDICES)
    if class_ids.isna().any():
        raise ValueError("The gene-disease mapping does not cover the saved split.")
    for column, key in MOI_KEYS:
        cohort[column] = cohort["moi_title"].isin(config[key])

    inventory = _read_csv(run / "results" / "feature_provenance_table.csv")
    active = inventory.loc[inventory["included_in_main_clean_run"]]
    memberships = active["used_in_scenarios"].str.split("|")
    container = SimpleNamespace(
        y=class_ids.to_numpy(dtype=int),
        full_df=cohort,
        groups=cohort["gene_symbol"].to_numpy(),
        scenario_cols={
            scenario: active.loc[memberships.map(lambda names: scenario in names), "feature"].tolist()
            for scenario in panels.SCENARIO_ORDER
        },
        feature_family_lookup=dict(zip(active["feature"], active["family"])),
    )
    indices = {
        key: cohort.index[cohort["partition"] == partition].to_numpy()
        for key, partition in (("train_idx", "train"), ("val_idx", "validation"), ("test_idx", "test"))
    }
    return container, {scenario: indices for scenario in panels.SCENARIO_ORDER}


def load_saved_results(run: Path, test_labels: np.ndarray) -> Tuple[Dict, Dict, SimpleNamespace]:
    """Model results, validation metrics and selected models of a saved run."""
    results = run / "results"
    figure_data = run / "figures" / "publication" / "figure_data"
    selection = _read_csv(results / "model_selection_validation.csv")
    primary = selection.loc[selection["Analysis"] == "primary"]
    winners = primary.loc[primary["selected_on_validation"]]
    selected = dict(zip(winners["Scenario"], winners["Model"]))
    if set(selected) != set(panels.SCENARIO_ORDER) or len(winners) != len(selected):
        raise ValueError("Expected one validation-selected model per feature set.")

    validation = _read_csv(figure_data / "figure_2_candidate_validation_metrics.csv").set_index(
        ["Scenario", "Model"]
    )
    scores = _read_csv(results / "publication_table.csv").set_index(["Scenario", "Model"])
    with (results / "all_results.pkl").open("rb") as source:
        collector = pickle.load(source)

    all_results: Dict = {scenario: {"all": {}} for scenario in panels.SCENARIO_ORDER}
    for row in primary.itertuples(index=False):
        key = row.Scenario, row.Model
        all_results[row.Scenario]["all"][row.Model] = {
            "selected_on_validation": selected[row.Scenario] == row.Model,
            "val_metrics": validation.loc[key]
            .drop(["selected_on_validation", "evaluation_partition"])
            .to_dict(),
        }
    for scenario, model in selected.items():
        predictions = collector["predictions"][f"{scenario}_{model}"]
        if not np.array_equal(predictions["y_true"], test_labels):
            raise ValueError(f"Saved test labels do not match the split for {scenario}.")
        all_results[scenario]["all"][model].update(
            metrics=collector["results"][scenario]["all"][model],
            y_true=predictions["y_true"],
            predictions=predictions,
            test_metrics_uncalibrated={
                "ranked_probability_score": scores.loc[(scenario, model), "Uncalibrated_RPS"],
                "multiclass_brier": scores.loc[(scenario, model), "Uncalibrated_Multiclass_Brier"],
            },
        )
    pub_outputs = {
        "selected_models": selected,
        "bootstrap": _read_csv(results / "bootstrap_cis.csv"),
        "permutation_test": _read_csv(results / "permutation_test.csv"),
        "s3_locked_test_pca": _read_csv(figure_data / "figure_5e_s3_true_predicted_pca.csv"),
    }
    return all_results, pub_outputs, SimpleNamespace(**collector)


def render_saved_run(run_dir: str, mapping_path: str, out_dir: str) -> Path:
    """Draw every benchmark figure of a saved run into ``out_dir``."""
    run = Path(run_dir).resolve()
    out = Path(out_dir).resolve()
    if out == run or (run / "figures" / "publication") in (out, *out.parents):
        raise ValueError("Write the redrawn figures outside the run's own figures/publication folder.")
    manifest = json.loads((run / "pipeline_manifest.json").read_text(encoding="utf-8-sig"))
    container, split = load_cohort(run, Path(mapping_path), manifest["config"])
    test_idx = split[panels.SCENARIO_ORDER[0]]["test_idx"]
    all_results, pub_outputs, collector = load_saved_results(run, container.y[test_idx])
    panels.generate_publication_figures(
        all_results,
        container,
        pub_outputs,
        collector,
        output_dir=str(run),
        fold_artifacts=split,
        figure_output_dir=str(out),
        random_seed=manifest["random_seeds"]["primary"],
    )
    return out


def render_in_pipeline(
    all_results: Dict, container, results_collector, fold_artifacts: Dict, run_dir: str, random_seed: int
) -> None:
    """Draw the figures at the end of a pipeline run, into ``<run_dir>/figures/publication``."""
    results = Path(run_dir) / "results"
    selected = {
        scenario: next(name for name, result in strata["all"].items() if result.get("selected_on_validation"))
        for scenario, strata in all_results.items()
        if scenario in panels.SCENARIO_ORDER
    }
    pub_outputs = {
        "selected_models": selected,
        "bootstrap": _read_csv(results / "bootstrap_cis.csv"),
        "permutation_test": _read_csv(results / "permutation_test.csv"),
    }
    panels.generate_publication_figures(
        all_results,
        container,
        pub_outputs,
        results_collector,
        output_dir=run_dir,
        fold_artifacts=fold_artifacts,
        random_seed=random_seed,
    )


def main(argv: Optional[Sequence[str]] = None) -> Path:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", required=True, help="completed benchmark run directory")
    parser.add_argument("--mapping", required=True, help="gene_disease_mapping_HPO_rebuilt.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    return render_saved_run(args.run, args.mapping, args.output)
