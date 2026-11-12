"""Prenatal cohort analysis (Results, "Application to an independent prenatal cohort"; Fig. 5;
Supplementary Tables 20-23).

The phenotype-only scorer is applied, without retraining, to the HPO terms recorded for each
pregnancy. Rows of the same family are combined into one profile. Outcomes are ordered

    live birth without NICU (0) < live birth with NICU or special care (1)
    < termination of pregnancy (2) < fetal or infant death (3)

and the score is tested against this order with Spearman's rho, Kendall's tau-b, the ordinal
concordance index (equation 12), a proportional-odds model (equation 13), mean score per group,
the separation of neighbouring groups and three prespecified ROC contrasts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from statsmodels.miscmodels.ordinal_model import OrderedModel  # noqa: E402

from ..scorer import load_scorer  # noqa: E402

LIVE_BIRTH = {
    "livebirth",
    "live birth",
    "live born child - no nicu/special care required",
    "live born child - nicu/special care required",
    "live born child - special care nursery at birth",
    "live born child - nicu 1-7 days",
    "live born child - nicu 2-4 weeks",
    "live born child - nicu 4+ weeks",
}
NICU_SET = {
    "live born child - special care nursery at birth",
    "live born child - nicu/special care required",
    "live born child - nicu 1-7 days",
    "live born child - nicu 2-4 weeks",
    "live born child - nicu 4+ weeks",
}
TOP_SET = {"top", "top following genomic result", "top before interview", "termination of pregnancy"}
DEATH_SET = {
    "stillbirth",
    "stillbirth (fdiu from 20 weeks to birth)",
    "neonatal death",
    "neonatal death (following birth)",
    "miscarriage",
    "live born child - death after 28 days",
}
LIVE_BIRTH_NO_NICU = LIVE_BIRTH - NICU_SET

OUTCOME_LABELS = {0: "Live birth (no NICU)", 1: "NICU / special care", 2: "TOP", 3: "Death"}
OUTCOME_PLOT_LABELS = {0: "Live birth\n(no NICU)", 1: "NICU /\nspecial care", 2: "TOP", 3: "Death"}
CLASS_NAMES = ["Mild", "Moderate", "Severe", "Profound"]
SCORE_LABEL = "Phenotype-derived severity score"
HPO_RE = re.compile("HP[:_]?([0-9]{7})", re.IGNORECASE)

FAMILY_COL = "family_id"
OUTCOME_COL = "pregnancy_outcome"
HPO_COL = "hpo_ids"
STUDY_COL = "study"


# Outcome mapping and family records (Supplementary Table 20)


def _normalise_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _is_death_outcome(text: str) -> bool:
    tokens = (
        "miscarriage",
        "stillbirth",
        "still birth",
        "fdiu",
        "fetal death",
        "foetal death",
        "neonatal death",
        "postnatal death",
        "death after",
        "died",
    )
    return text in DEATH_SET or any(token in text for token in tokens)


def _is_top_outcome(text: str) -> bool:
    return text in TOP_SET or text.startswith("top ") or "termination of pregnancy" in text


def map_outcome(value: Any) -> float:
    """Four-level outcome (0-3), or NaN when the label is missing or does not fit.

    Live births with no information on NICU admission go to level 0.
    """
    text = _normalise_text(value)
    if not text:
        return np.nan
    if _is_death_outcome(text):
        return 3.0
    if _is_top_outcome(text):
        return 2.0
    if text in LIVE_BIRTH_NO_NICU or "no nicu/special care required" in text:
        return 0.0
    if text in NICU_SET:
        return 1.0
    if text.startswith("live born child") and (
        "nicu" in text or "special care nursery" in text or "special care required" in text
    ):
        return 1.0
    if text in LIVE_BIRTH or "live birth" in text or "livebirth" in text:
        return 0.0
    return np.nan


def parse_hpo(cell: Any) -> List[str]:
    """HPO identifiers in a cell, as HP:NNNNNNN, in order of first appearance."""
    if pd.isna(cell):
        return []
    terms: List[str] = []
    for match in HPO_RE.finditer(str(cell)):
        term = f"HP:{match.group(1)}"
        if term not in terms:
            terms.append(term)
    return terms


def _unique_nonempty(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = "" if pd.isna(value) else re.sub(r"\s+", " ", str(value).strip())
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            out.append(text)
    return out


def _union_hpo(values: Sequence[Any]) -> List[str]:
    terms: List[str] = []
    for value in values:
        for term in parse_hpo(value):
            if term not in terms:
                terms.append(term)
    return terms


def _resolve_family_outcome(values: Sequence[Any]) -> Dict[str, Any]:
    """Outcome of one family. Families whose rows disagree are flagged and left unmapped."""
    labels = _unique_nonempty(values)
    mapped = [map_outcome(label) for label in labels]
    levels = sorted({int(v) for v in mapped if np.isfinite(v)})
    n_unmapped = sum(1 for v in mapped if not np.isfinite(v))
    conflict = len(levels) > 1 or (n_unmapped > 0 and bool(levels))
    return {
        "selected_label": labels[0] if labels and not conflict else "",
        "all_labels": " | ".join(labels),
        "outcome": float(levels[0]) if len(levels) == 1 and not conflict else np.nan,
        "outcome_conflict": bool(conflict),
        "n_unmapped_labels": int(n_unmapped),
    }


def _aggregate_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """One record per family: HPO terms are unioned and other fields keep their distinct values."""
    records: List[Dict[str, Any]] = []
    for family_id, group in df.groupby(FAMILY_COL, sort=False, dropna=False):
        record: Dict[str, Any] = {FAMILY_COL: str(family_id)}
        for column in df.columns:
            if column not in (FAMILY_COL, OUTCOME_COL, HPO_COL):
                record[column] = " | ".join(_unique_nonempty(group[column].tolist()))
        resolved = _resolve_family_outcome(group[OUTCOME_COL].tolist())
        record[HPO_COL] = ";".join(_union_hpo(group[HPO_COL].tolist()))
        record[OUTCOME_COL] = resolved["selected_label"]
        record["pregnancy_outcome_all"] = resolved["all_labels"]
        record["outcome"] = resolved["outcome"]
        record["outcome_conflict"] = resolved["outcome_conflict"]
        record["n_unmapped_outcome_labels"] = resolved["n_unmapped_labels"]
        record["source_row_count"] = int(len(group))
        record["source_studies"] = (
            " | ".join(_unique_nonempty(group[STUDY_COL].tolist())) if STUDY_COL in group.columns else ""
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def load_cohort(path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Read the cohort (columns family_id, pregnancy_outcome, hpo_ids, study) and combine rows by family."""
    df = pd.read_csv(path, low_memory=False)
    missing = [column for column in (OUTCOME_COL, HPO_COL) if column not in df.columns]
    if missing:
        raise ValueError(f"Cohort is missing required columns: {missing}")
    audit: Dict[str, Any] = {"n_start_rows": int(len(df))}
    if FAMILY_COL not in df.columns:
        df[FAMILY_COL] = [f"ROW_{i + 1:06d}" for i in range(len(df))]
        audit["family_id_generated_from_rows"] = True
    else:
        audit["family_id_generated_from_rows"] = False
        raw = df[FAMILY_COL]
        absent = raw.isna() | raw.astype(str).str.strip().str.casefold().isin({"", "nan", "none", "null"})
        df[FAMILY_COL] = raw.astype(str).str.strip()
        df.loc[absent, FAMILY_COL] = [f"ROW_MISSING_{i + 1:06d}" for i in range(int(absent.sum()))]
    audit["n_missing_family_ids_replaced"] = int(
        df[FAMILY_COL].astype(str).str.startswith("ROW_MISSING_").sum()
    )
    audit["n_exact_duplicate_rows_observed"] = int(df.duplicated().sum())
    sizes = df.groupby(FAMILY_COL, sort=False).size()
    audit["n_multirow_families"] = int((sizes > 1).sum())
    audit["n_rows_in_multirow_families"] = int(sizes[sizes > 1].sum())

    families = _aggregate_by_family(df)
    families["hpo_list"] = families[HPO_COL].apply(parse_hpo)
    families["n_hpo"] = families["hpo_list"].apply(len)
    families["has_mapped_outcome"] = families["outcome"].notna()
    families["has_hpo"] = families["n_hpo"] > 0

    audit["n_family_profiles"] = int(len(families))
    audit["n_outcome_conflict_families"] = int(families["outcome_conflict"].sum())
    audit["n_with_hpo"] = int(families["has_hpo"].sum())
    audit["n_without_hpo"] = int((~families["has_hpo"]).sum())
    audit["n_mapped_outcome"] = int(families["has_mapped_outcome"].sum())
    audit["n_missing_or_unmapped_outcome"] = int((~families["has_mapped_outcome"]).sum())
    audit["outcome_counts"] = {
        label: int((families["outcome"] == level).sum()) for level, label in OUTCOME_LABELS.items()
    }
    return families.reset_index(drop=True), audit


# Scoring


def score_families(families: pd.DataFrame, scorer) -> pd.DataFrame:
    """Add the phenotype-only severity score (expected severity / 3) to each family.

    ``direct_term_effects_json`` holds, for the multichannel version, the change in expected
    severity when each recorded term is removed (Supplementary Table 23).
    """
    rows: List[Dict[str, Any]] = []
    for terms in families["hpo_list"]:
        row: Dict[str, Any] = {
            "severity_score": np.nan,
            "phenotype_expected_severity_0_3": np.nan,
            "combined_expected_severity_0_3": np.nan,
        }
        row.update({f"phenotype_probability_{name.lower()}": np.nan for name in CLASS_NAMES})
        if not terms:
            row.update(
                score_status="no_hpo_terms",
                score_warnings="No HPO terms.",
                n_terms_supported=0,
                direct_term_effects_json="[]",
            )
            rows.append(row)
            continue
        result = scorer.score_profile(list(terms), absent_hpo_terms=[], allow_low_information=False)
        phenotype_expected = result.get("phenotype_only_expected_severity_0_3")
        if phenotype_expected is not None:
            row["phenotype_expected_severity_0_3"] = float(phenotype_expected)
            row["severity_score"] = float(phenotype_expected) / 3.0
        if result.get("combined_expected_severity_0_3") is not None:
            row["combined_expected_severity_0_3"] = float(result["combined_expected_severity_0_3"])
        for name, value in (result.get("phenotype_only_class_probabilities") or {}).items():
            row[f"phenotype_probability_{name.lower()}"] = float(value)
        row.update(
            score_status=str(result["status"]),
            score_warnings=" | ".join(map(str, result.get("warnings", []))),
            n_terms_supported=int(result.get("n_supported_phenotype_direct_terms", 0)),
            direct_term_effects_json=json.dumps(
                result.get("direct_term_leave_one_out_effects", []), sort_keys=True
            ),
        )
        rows.append(row)
    out = pd.concat([families, pd.DataFrame.from_records(rows, index=families.index)], axis=1)
    out["score_available"] = np.isfinite(out["severity_score"])
    out["included_in_analysis"] = out["score_available"] & out["outcome"].notna()
    return out


# Association statistics (Supplementary Tables 21 and 22)


def _pairwise_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a value from ``pos`` exceeds one from ``neg``), ties counted as 0.5."""
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    u = ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def ordinal_concordance(score: np.ndarray, outcome: np.ndarray) -> float:
    """Equation (12): pairwise AUCs over all pairs of outcome levels, weighted by pair counts."""
    s = np.asarray(score, float)
    y = np.asarray(outcome, float)
    levels = np.unique(y[np.isfinite(y)])
    num = den = 0.0
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            lo = s[y == levels[i]]
            hi = s[y == levels[j]]
            if lo.size == 0 or hi.size == 0:
                continue
            weight = float(lo.size * hi.size)
            num += weight * _pairwise_auc(hi, lo)
            den += weight
    return float("nan") if den == 0 else num / den


def format_permutation_p(perm_p: float, n_perm_used: int, unicode_times: bool = True) -> str:
    """Permutation P as text. At the floor 1/(B + 1) the floor itself is shown, not 0.000."""
    if perm_p is None or not np.isfinite(perm_p):
        return "not available"
    floor = 1.0 / (int(n_perm_used) + 1) if int(n_perm_used) > 0 else 0.0
    times = " × " if unicode_times else " x "

    def scientific(value: float) -> str:
        exponent = int(np.floor(np.log10(value)))
        mantissa = value / 10.0**exponent
        if unicode_times:
            superscripts = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
            return f"{mantissa:.1f}{times}10{str(exponent).translate(superscripts)}"
        return f"{mantissa:.1f}{times}10^{exponent}"

    if floor > 0 and perm_p <= floor * (1.0 + 1e-9):
        return scientific(floor)
    if perm_p < 0.001:
        return scientific(perm_p)
    return f"{perm_p:.3f}"


def correlation_analysis(
    df: pd.DataFrame,
    n_boot: int,
    n_perm: int,
    seed: int = 1,
    score_col: str = "severity_score",
    outcome_col: str = "outcome",
) -> Dict[str, Any]:
    """Spearman's rho, Kendall's tau-b and ordinal concordance, with family bootstrap 95% CIs
    and two-sided label-permutation P values."""
    rng = np.random.RandomState(seed)
    s = df[score_col].to_numpy(float)
    y = df[outcome_col].to_numpy(float)
    n = len(s)
    rho, _ = stats.spearmanr(s, y)
    tau, _ = stats.kendalltau(s, y, variant="b")
    conc = ordinal_concordance(s, y)
    boot, conc_boot = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        r, _ = stats.spearmanr(s[idx], y[idx])
        if np.isfinite(r):
            boot.append(r)
        c = ordinal_concordance(s[idx], y[idx])
        if np.isfinite(c):
            conc_boot.append(c)
    null, conc_null = [], []
    for _ in range(n_perm):
        shuffled = rng.permutation(y)
        r, _ = stats.spearmanr(s, shuffled)
        if np.isfinite(r):
            null.append(r)
        c = ordinal_concordance(s, shuffled)
        if np.isfinite(c):
            conc_null.append(c)
    boot, conc_boot, null, conc_null = map(np.array, (boot, conc_boot, null, conc_null))

    def interval(values: np.ndarray) -> Tuple[float, float]:
        if not values.size:
            return float("nan"), float("nan")
        return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

    ci = interval(boot)
    conc_ci = interval(conc_boot)
    perm_p = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (null.size + 1)) if null.size else float("nan")
    conc_perm_p = (
        float((np.sum(np.abs(conc_null - 0.5) >= abs(conc - 0.5)) + 1) / (conc_null.size + 1))
        if conc_null.size
        else float("nan")
    )
    return {
        "spearman_rho": float(rho),
        "kendall_tau_b": float(tau),
        "ordinal_concordance": float(conc),
        "ordinal_concordance_ci95_low": conc_ci[0],
        "ordinal_concordance_ci95_high": conc_ci[1],
        "ordinal_concordance_perm_p": conc_perm_p,
        "ci95_low": ci[0],
        "ci95_high": ci[1],
        "perm_p": perm_p,
        "n_boot_used": int(boot.size),
        "n_perm_used": int(null.size),
        "n": int(n),
        "score_col": score_col,
        "outcome_col": outcome_col,
    }


def proportional_odds(
    df: pd.DataFrame, outcome_col: str = "outcome", score_col: str = "severity_score"
) -> Dict[str, Any]:
    """Equation (13): ordered logit of outcome on the standardised score. OR per +1 SD, Wald CI."""
    s = df[score_col].to_numpy(float)
    sd = float(s.std())
    z = (s - s.mean()) / (sd if sd > 0 else 1.0)
    y = df[outcome_col].to_numpy(int)
    res = OrderedModel(y, z.reshape(-1, 1), distr="logit").fit(method="bfgs", disp=False)
    beta, se = float(res.params[0]), float(res.bse[0])
    return {
        "available": True,
        "method": "statsmodels_ordered_logit",
        "score_col": score_col,
        "or_per_sd": float(np.exp(beta)),
        "ci95": [float(np.exp(beta - 1.96 * se)), float(np.exp(beta + 1.96 * se))],
        "p_value": float(2 * (1 - stats.norm.cdf(abs(beta / se)))) if se > 0 else float("nan"),
        "beta": beta,
        "se": se,
        "outcome_col": outcome_col,
    }


def mean_ci_by_outcome(
    df: pd.DataFrame, outcome_col: str = "outcome", score_col: str = "severity_score"
) -> pd.DataFrame:
    """Mean score per outcome group with mean +/- 1.96 standard errors."""
    rows = []
    for level, label in OUTCOME_LABELS.items():
        sub = df[df[outcome_col] == level][score_col].to_numpy(float)
        if sub.size == 0:
            continue
        mean = float(sub.mean())
        se = float(sub.std(ddof=1) / np.sqrt(len(sub))) if len(sub) > 1 else 0.0
        rows.append(
            {
                "outcome": label,
                "level": int(level),
                "n": int(len(sub)),
                "mean": mean,
                "ci_low": mean - 1.96 * se,
                "ci_high": mean + 1.96 * se,
            }
        )
    return pd.DataFrame(rows)


ROC_CONTRASTS: List[Tuple[str, Callable, Callable]] = [
    ("Death vs live birth (no NICU)", lambda y: y == 3, lambda y: y == 0),
    ("TOP vs live birth (no NICU)", lambda y: y == 2, lambda y: y == 0),
    ("Death vs NICU / special care", lambda y: y == 3, lambda y: y == 1),
]


def roc_contrasts(
    df: pd.DataFrame, outcome_col: str = "outcome", score_col: str = "severity_score"
) -> List[Dict[str, Any]]:
    """AUC for the prespecified contrasts; the more severe group is the positive class."""
    score = df[score_col].to_numpy(float)
    outcome = df[outcome_col].to_numpy(int)
    results: List[Dict[str, Any]] = []
    for name, positive_rule, negative_rule in ROC_CONTRASTS:
        positive = np.asarray(positive_rule(outcome), dtype=bool)
        negative = np.asarray(negative_rule(outcome), dtype=bool)
        if positive.sum() == 0 or negative.sum() == 0:
            continue
        selected = positive | negative
        labels = positive[selected].astype(int)
        fpr, tpr, _ = roc_curve(labels, score[selected])
        results.append(
            {
                "name": name,
                "score_col": score_col,
                "auc": float(roc_auc_score(labels, score[selected])),
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "n_pos": int(positive.sum()),
                "n_neg": int(negative.sum()),
            }
        )
    return results


def adjacent_level_separation(
    df: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 7,
    outcome_col: str = "outcome",
    score_col: str = "severity_score",
) -> pd.DataFrame:
    """For each pair of neighbouring outcome groups, P(the more severe group scores higher),
    with a bootstrap 95% CI. A CI that covers 0.5 means that boundary is not resolved."""
    rng = np.random.RandomState(seed)
    score = df[score_col].to_numpy(float)
    outcome = df[outcome_col].to_numpy(float)
    levels = sorted(int(v) for v in np.unique(outcome[np.isfinite(outcome)]))
    rows = []
    for lower, upper in zip(levels[:-1], levels[1:]):
        lo = score[outcome == lower]
        hi = score[outcome == upper]
        if lo.size == 0 or hi.size == 0:
            continue
        draws = []
        for _ in range(n_boot):
            lo_b = lo[rng.randint(0, lo.size, lo.size)]
            hi_b = hi[rng.randint(0, hi.size, hi.size)]
            value = _pairwise_auc(hi_b, lo_b)
            if np.isfinite(value):
                draws.append(value)
        draws = np.array(draws)
        ci_low, ci_high = (
            (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
            if draws.size
            else (float("nan"), float("nan"))
        )
        rows.append(
            {
                "boundary": f"{OUTCOME_LABELS.get(lower, lower)} vs {OUTCOME_LABELS.get(upper, upper)}",
                "lower_level": lower,
                "upper_level": upper,
                "n_lower": int(lo.size),
                "n_upper": int(hi.size),
                "mean_lower": float(lo.mean()),
                "mean_upper": float(hi.mean()),
                "mean_difference": float(hi.mean() - lo.mean()),
                "prob_more_severe_scores_higher": float(_pairwise_auc(hi, lo)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "resolved_at_95pct": bool(np.isfinite(ci_low) and ci_low > 0.5),
                "n_boot_used": int(draws.size),
                "score_col": score_col,
            }
        )
    return pd.DataFrame(rows)


def phenotype_contributions(analysis: pd.DataFrame) -> pd.DataFrame:
    """Supplementary Table 23: leave-one-term-out changes in expected severity (multichannel
    version), summed over the analysed families."""
    records = []
    for payload in analysis["direct_term_effects_json"]:
        for effect in json.loads(payload):
            records.append(
                {
                    "hpo_id": effect["hpo_id"],
                    "name": effect.get("name", ""),
                    "delta": float(effect["delta_expected_severity_when_present"]),
                }
            )
    if not records:
        return pd.DataFrame(
            columns=[
                "hpo_id",
                "hpo_term",
                "families",
                "mean_change_expected_severity",
                "total_absolute_contribution",
                "share_of_total_absolute_contribution",
            ]
        )
    effects = pd.DataFrame(records)
    effects["abs_delta"] = effects["delta"].abs()
    table = (
        effects.groupby("hpo_id")
        .agg(
            hpo_term=("name", "first"),
            families=("delta", "size"),
            mean_change_expected_severity=("delta", "mean"),
            total_absolute_contribution=("abs_delta", "sum"),
        )
        .reset_index()
    )
    table["share_of_total_absolute_contribution"] = (
        table["total_absolute_contribution"] / table["total_absolute_contribution"].sum()
    )
    return table.sort_values("total_absolute_contribution", ascending=False).reset_index(drop=True)


# Figure 5


def _draw_scorer_confusion(
    ax, heldout: pd.DataFrame, title: str = "a  Scorer held-out confusion matrix"
) -> None:
    """Phenotype-only scorer on the 1,241 held-out diseases: counts and within-row proportions."""
    observed = heldout["observed"].astype(int).to_numpy()
    predicted = np.clip(np.round(heldout["phenotype_expected"].to_numpy()), 0, len(CLASS_NAMES) - 1).astype(
        int
    )
    n_classes = len(CLASS_NAMES)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_value, pred_value in zip(observed, predicted):
        matrix[true_value, pred_value] += 1
    row_totals = matrix.sum(axis=1, keepdims=True)
    proportions = np.divide(matrix, np.where(row_totals == 0, 1, row_totals), dtype=float)
    image = ax.imshow(proportions, cmap="Blues", vmin=0.0, vmax=1.0)
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j,
                i,
                "{:,}\n{:.0f}%".format(matrix[i, j], proportions[i, j] * 100),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if proportions[i, j] > 0.55 else "#222222",
            )
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right")
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels([f"{name}\n(n={int(row_totals[i, 0])})" for i, name in enumerate(CLASS_NAMES)])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Observed class")
    ax.set_title(title, loc="left", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("Within-observed-class proportion", fontsize=8)
    bar.ax.tick_params(labelsize=8)


def _draw_scorer_expected_severity(
    ax, heldout: pd.DataFrame, title: str = "b  Scorer held-out expected severity by observed class"
) -> None:
    """Expected severity (0-3) by observed class for the phenotype-only and multichannel versions."""
    rng = np.random.RandomState(0)
    channels = [
        ("phenotype_expected", "Phenotype-only scorer", "#1f77b4", -0.16, "o"),
        ("combined_expected", "Multichannel version", "#ff7f0e", 0.16, "x"),
    ]
    for column, label, color, offset, marker in channels:
        for class_index in range(len(CLASS_NAMES)):
            values = heldout.loc[heldout["observed"].astype(int) == class_index, column].to_numpy(dtype=float)
            if values.size == 0:
                continue
            jitter = rng.uniform(-0.055, 0.055, size=values.size)
            ax.scatter(
                np.full(values.size, class_index + offset) + jitter,
                values,
                s=9,
                alpha=0.35,
                color=color,
                marker=marker,
                linewidths=0.6,
                label=label if class_index == 0 else None,
            )
            ax.hlines(
                float(np.mean(values)),
                class_index + offset - 0.13,
                class_index + offset + 0.13,
                color=color,
                linewidth=2.4,
                zorder=5,
            )
    ax.plot(
        range(len(CLASS_NAMES)),
        range(len(CLASS_NAMES)),
        "--",
        color="gray",
        linewidth=1.2,
        label="Reference class score",
    )
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_xlim(-0.55, len(CLASS_NAMES) - 0.45)
    ax.set_ylim(-0.12, 3.12)
    ax.set_xlabel("Observed severity class")
    ax.set_ylabel("Expected severity (0-3)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)


def _draw_group_distribution(ax, df: pd.DataFrame, title: str) -> None:
    levels = list(OUTCOME_LABELS)
    colors = ["#1f77b4", "#17becf", "#ff7f0e", "#d62728"]
    data = [df[df["outcome"] == level]["severity_score"].to_numpy(float) for level in levels]
    nonempty = [(level, values) for level, values in zip(levels, data) if len(values)]
    violin_ready = [(level, values) for level, values in nonempty if len(values) >= 2]
    if violin_ready:
        parts = ax.violinplot(
            [values for _, values in violin_ready],
            positions=[level for level, _ in violin_ready],
            widths=0.8,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor("#a6cee3")
            body.set_alpha(0.45)
    if nonempty:
        ax.boxplot(
            [values for _, values in nonempty],
            positions=[level for level, _ in nonempty],
            widths=0.25,
            showfliers=False,
            medianprops=dict(color="black"),
        )
    rng = np.random.RandomState(0)
    for level, values, color in zip(levels, data, colors):
        if len(values):
            ax.scatter(
                level + rng.uniform(-0.09, 0.09, len(values)),
                values,
                s=14,
                alpha=0.55,
                color=color,
                edgecolors="none",
                zorder=3,
            )
    means = np.array([values.mean() if len(values) else np.nan for values in data], dtype=float)
    valid = np.isfinite(means)
    if valid.any():
        level_array = np.asarray(levels)
        ax.plot(level_array[valid], means[valid], "--", color="#b22222", zorder=4)
        ax.scatter(
            level_array[valid],
            means[valid],
            marker="D",
            s=90,
            color="#b22222",
            edgecolors="white",
            zorder=5,
            label="Group mean",
        )
    ax.set_xticks(levels)
    ax.set_xticklabels(
        [f"{OUTCOME_PLOT_LABELS[level]}\n(n={len(values)})" for level, values in zip(levels, data)]
    )
    ax.set_ylabel(f"{SCORE_LABEL} (0-1)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylim(-0.02, 1.02)
    if valid.any():
        ax.legend(loc="upper left")


def _draw_mean_panel(ax, mean_tbl: pd.DataFrame, corr: Dict[str, Any], title: str) -> None:
    ax.errorbar(
        mean_tbl["level"],
        mean_tbl["mean"],
        yerr=[mean_tbl["mean"] - mean_tbl["ci_low"], mean_tbl["ci_high"] - mean_tbl["mean"]],
        fmt="o-",
        color="#1f77b4",
        capsize=5,
        markersize=9,
        linewidth=2,
    )
    for _, row in mean_tbl.iterrows():
        ax.annotate(
            f"n={int(row['n'])}",
            (row["level"], row["ci_high"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
        )
    ax.set_xticks(list(OUTCOME_LABELS))
    ax.set_xticklabels([OUTCOME_PLOT_LABELS[level] for level in OUTCOME_LABELS])
    ax.set_ylabel("Mean severity score (95% CI)")
    ax.set_title(title, loc="left", fontweight="bold")
    text = (
        f"Spearman rho = {corr['spearman_rho']:+.3f}\n"
        f"(95% CI {corr['ci95_low']:+.3f}, {corr['ci95_high']:+.3f})\n"
        f"permutation p = {format_permutation_p(corr['perm_p'], corr.get('n_perm_used', 0))}"
    )
    ax.text(
        0.97,
        0.04,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="gray"),
    )


def _draw_class_probability_panel(ax, df: pd.DataFrame, title: str) -> None:
    """Mean calibrated class probabilities of the phenotype-only scorer per outcome group."""
    columns = [f"phenotype_probability_{name.lower()}" for name in CLASS_NAMES]
    levels = list(OUTCOME_LABELS)
    matrix, counts = [], []
    for level in levels:
        complete = df[df["outcome"] == level][columns].apply(pd.to_numeric, errors="coerce").dropna(how="all")
        counts.append(int(len(complete)))
        matrix.append([float(complete[column].mean()) if len(complete) else 0.0 for column in columns])
    matrix_array = np.asarray(matrix, dtype=float)
    totals = matrix_array.sum(axis=1, keepdims=True)
    matrix_array = np.divide(matrix_array, totals, out=np.zeros_like(matrix_array), where=totals > 0)
    bottom = np.zeros(len(levels), dtype=float)
    for index, (label, color) in enumerate(zip(CLASS_NAMES, ["#0072B2", "#56B4E9", "#E69F00", "#CC79A7"])):
        ax.bar(levels, matrix_array[:, index], bottom=bottom, width=0.68, label=label, color=color)
        bottom += matrix_array[:, index]
    for level, count in zip(levels, counts):
        ax.annotate(
            f"n={count}",
            (level, 1.0),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(levels)
    ax.set_xticklabels([OUTCOME_PLOT_LABELS[level] for level in levels])
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Mean calibrated class probability")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(ncol=4, loc="upper center", fontsize=8, frameon=False)


def build_figure(
    analysis: pd.DataFrame,
    mean_tbl: pd.DataFrame,
    corr: Dict[str, Any],
    contrasts: List[Dict[str, Any]],
    heldout: pd.DataFrame,
    out_png: str,
) -> None:
    """Fig. 5: scorer held-out validation (a, b) and the prenatal cohort (c-f). Writes PNG and PDF."""
    fig, axes = plt.subplots(3, 2, figsize=(15.5, 17.0), constrained_layout=True)
    _draw_scorer_confusion(axes[0, 0], heldout)
    _draw_scorer_expected_severity(axes[0, 1], heldout)
    _draw_group_distribution(axes[1, 0], analysis, "c  Severity score by pregnancy outcome")
    _draw_mean_panel(axes[1, 1], mean_tbl, corr, "d  Mean score by pregnancy outcome")
    ax = axes[2, 0]
    for contrast, color in zip(contrasts, ["#1f77b4", "#ff7f0e", "#d62728"]):
        ax.plot(
            contrast["fpr"],
            contrast["tpr"],
            color=color,
            linewidth=2,
            label=f"{contrast['name']}  AUC={contrast['auc']:.3f} (n={contrast['n_pos']} vs {contrast['n_neg']})",
        )
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("e  ROC for clinical contrasts", loc="left", fontweight="bold")
    if contrasts:
        ax.legend(loc="lower right", fontsize=8)
    _draw_class_probability_panel(
        axes[2, 1], analysis, "f  Mean severity-class probabilities by pregnancy outcome"
    )
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_png)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)


# Command line


def _records(table: pd.DataFrame) -> List[Dict[str, Any]]:
    return table.to_dict(orient="records")


def run(
    cohort: str, model: str, heldout: str, output: str, n_boot: int = 2000, n_perm: int = 2000
) -> Dict[str, Any]:
    """Run the full prenatal analysis and write tables, ``statistics.json`` and Fig. 5 to ``output``."""
    os.makedirs(output, exist_ok=True)
    families, audit = load_cohort(cohort)
    scorer = load_scorer(model)
    families = score_families(families, scorer)
    analysis = families[families["included_in_analysis"]].reset_index(drop=True)
    audit["n_scoreable"] = int(families["score_available"].sum())
    audit["n_not_scoreable"] = int((~families["score_available"]).sum())
    audit["n_analyzed"] = int(len(analysis))
    audit["scoring_status_counts"] = {
        str(k): int(v) for k, v in families["score_status"].value_counts().items()
    }
    audit["outcome_counts_analyzed"] = {
        label: int((analysis["outcome"] == level).sum()) for level, label in OUTCOME_LABELS.items()
    }
    if len(analysis) < 4 or analysis["outcome"].nunique() < 2:
        raise ValueError("Not enough scored families with a mapped outcome.")

    corr = correlation_analysis(analysis, n_boot, n_perm, seed=1)
    po = proportional_odds(analysis)
    means = mean_ci_by_outcome(analysis)
    contrasts = roc_contrasts(analysis)
    adjacent = adjacent_level_separation(analysis, n_boot=n_boot)
    contributions = phenotype_contributions(analysis)

    export = families.drop(columns=["hpo_list"])
    export.to_csv(os.path.join(output, "severity_by_family.csv"), index=False)
    export[export["included_in_analysis"]].to_csv(os.path.join(output, "severity_analysis.csv"), index=False)
    means.to_csv(os.path.join(output, "mean_by_outcome.csv"), index=False)
    roc_table = pd.DataFrame([{k: c[k] for k in ("name", "auc", "n_pos", "n_neg")} for c in contrasts])
    roc_table.rename(columns={"name": "contrast"}).to_csv(
        os.path.join(output, "roc_contrasts.csv"), index=False
    )
    adjacent.to_csv(os.path.join(output, "adjacent_level_separation.csv"), index=False)
    contributions.to_csv(os.path.join(output, "phenotype_contributions.csv"), index=False)

    statistics = {
        "audit": audit,
        "correlation": corr,
        "proportional_odds": po,
        "roc_contrasts": _records(roc_table),
        "adjacent_level_separation": _records(adjacent),
        "score_label": SCORE_LABEL,
    }
    with open(os.path.join(output, "statistics.json"), "w") as handle:
        json.dump(statistics, handle, indent=2)

    build_figure(
        analysis,
        means,
        corr,
        contrasts,
        pd.read_csv(heldout),
        os.path.join(output, "prenatal_severity_figure.png"),
    )
    print(
        f"families={audit['n_family_profiles']} analysed={len(analysis)} "
        f"rho={corr['spearman_rho']:+.3f} (95% CI {corr['ci95_low']:+.3f}, {corr['ci95_high']:+.3f}) "
        f"p={format_permutation_p(corr['perm_p'], corr['n_perm_used'], unicode_times=False)}"
    )
    return statistics


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cohort", required=True, help="CSV with family_id, pregnancy_outcome, hpo_ids, study"
    )
    parser.add_argument("--model", required=True, help="model.pkl of the disease-level scorer")
    parser.add_argument(
        "--heldout", required=True, help="held_out_test_predictions.csv of the scorer (Fig. 5a, b)"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=2000)
    args = parser.parse_args(argv)
    return run(args.cohort, args.model, args.heldout, args.output, args.n_boot, args.n_perm)
