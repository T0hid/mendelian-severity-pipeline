#!/usr/bin/env python3
"""Train and apply the disease-level phenotype scorer."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import pickle
import re
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

RANDOM_SEED = 42
SEVERITY_MAP = {"Mild": 0, "Moderate": 1, "Severe": 2, "Profound": 3}
CLASS_NAMES = ["Mild", "Moderate", "Severe", "Profound"]
NUM_CLASSES = 4
HPO_RE = re.compile(r"^HP:\d{7}$")


EXPLICIT_SEVERITY_HPO_LEVELS: Dict[str, int] = {
    # Lethal / death endpoints.
    "HP:0003824": 3,  # Neonatal death
}
SEVERITY_HPO_IDS = set(EXPLICIT_SEVERITY_HPO_LEVELS)
SEVERITY_HPO_KEYWORDS = ("profound", "severe", "moderate", "mild")

# Non-ordinal clinical modifiers should not be treated as phenotype evidence or
# forced into an ordinal severity level.
EXCLUDED_CLINICAL_MODIFIER_HPO_IDS = {
    "HP:0011420",  # Age of death (no age/value by itself)
    "HP:0003828",  # Variable expressivity
}



LEGACY_ORGAN_SYSTEM_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "nervous_system": ("nervous system", "neurolog", "brain", "spinal cord"),
    "cardiovascular": ("cardiovascular", "heart", "cardiac", "blood vessel", "vascular"),
    "respiratory": ("respiratory system", "lung", "pulmonary", "airway", "breathing"),
    "renal_urinary": ("urinary system", "kidney", "renal", "ureter", "bladder"),
    "digestive_hepatic": ("digestive system", "gastrointestinal", "intestin", "liver", "hepatic"),
    "musculoskeletal": ("skeletal system", "musculature", "muscle", "bone", "joint"),
    "immune": ("immune system", "immun", "inflammation"),
    "hematologic": ("blood and blood-forming", "hematolog", "haematolog", "coagulation"),
    "endocrine_metabolic": ("endocrine", "metabolism/homeostasis", "metabolic", "hormone"),
    "sensory": ("eye", "visual", "ear", "hearing", "vestibular"),
    "integument": ("integument", "skin", "hair", "nail"),
    "reproductive": ("genital", "reproductive system", "gonad", "uterus", "testis", "ovary"),
}

# IDs are stable anchors when present. Exact-name aliases allow the same code to
# adapt to an HPO release in which a branch ID/name has moved or been renamed.
# Fallback patterns are token/boundary aware and are used only when no branch root
# can be resolved for that system.
ORGAN_SYSTEM_BRANCHES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "nervous_system": {
        "ids": ("HP:0000707",),
        "names": ("abnormality of the nervous system",),
        "patterns": (r"nervous system", r"neurolog(?:ic|ical|ically|y)?", r"brain", r"spinal cord"),
    },
    "cardiovascular": {
        "ids": ("HP:0001626",),
        "names": ("abnormality of the cardiovascular system",),
        "patterns": (r"cardiovascular", r"heart", r"cardiac", r"blood vessels?", r"vascular"),
    },
    "respiratory": {
        "ids": ("HP:0002086",),
        "names": ("abnormality of the respiratory system",),
        "patterns": (r"respiratory system", r"lungs?", r"pulmonary", r"airways?", r"breathing"),
    },
    "renal_urinary": {
        "ids": ("HP:0000079",),
        "names": ("abnormality of the urinary system",),
        "patterns": (r"urinary system", r"kidneys?", r"renal", r"ureters?", r"bladder"),
    },
    "digestive_hepatic": {
        "ids": ("HP:0025031",),
        "names": ("abnormality of the digestive system",),
        "patterns": (r"digestive system", r"gastrointestinal", r"intestin(?:e|al)?", r"liver", r"hepatic"),
    },
    "musculoskeletal": {
        "ids": ("HP:0033127",),
        "names": ("abnormality of the musculoskeletal system",),
        "patterns": (r"musculoskeletal", r"skeletal system", r"musculature", r"muscles?", r"bones?", r"joints?"),
    },
    "immune": {
        "ids": ("HP:0002715",),
        "names": ("abnormality of the immune system",),
        "patterns": (r"immune system", r"immun(?:e|ologic|ological|ity)", r"inflamm(?:ation|atory)"),
    },
    "hematologic": {
        "ids": ("HP:0001871",),
        "names": ("abnormality of blood and blood-forming tissues",),
        "patterns": (r"blood and blood-forming tissues", r"hematolog(?:ic|ical|y)?", r"haematolog(?:ic|ical|y)?", r"coagulation"),
    },
    "endocrine_metabolic": {
        "ids": ("HP:0000818", "HP:0001939"),
        "names": ("abnormality of the endocrine system", "abnormality of metabolism/homeostasis"),
        "patterns": (r"endocrine", r"metabolism/homeostasis", r"metabolic", r"hormones?"),
    },
    "eye": {
        "ids": ("HP:0000478",),
        "names": ("abnormality of the eye",),
        "patterns": (r"eyes?", r"visual", r"ocular", r"retina", r"cornea"),
    },
    "ear": {
        "ids": ("HP:0000598",),
        "names": ("abnormality of the ear",),
        "patterns": (r"ears?", r"hearing", r"vestibular", r"deafness", r"auditory"),
    },
    "integument": {
        "ids": ("HP:0001574",),
        "names": ("abnormality of the integument",),
        "patterns": (r"integument", r"skin", r"hair", r"nails?"),
    },
    "reproductive": {
        "ids": ("HP:0012243", "HP:0000080"),
        "names": (
            "abnormal reproductive system morphology",
            "abnormality of reproductive system physiology",
            "abnormality of the genital system",
        ),
        "patterns": (r"genital", r"reproductive system", r"gonads?", r"uterus", r"test(?:is|es|icular)", r"ovar(?:y|ies|ian)"),
    },


    "growth": {
        "ids": (),
        "names": ("growth abnormality",),
        "patterns": (r"growth abnormality", r"short stature", r"tall stature", r"failure to thrive"),
    },
    "prenatal_and_birth": {
        "ids": (),
        "names": ("abnormality of prenatal development or birth",),
        "patterns": (r"prenatal development or birth", r"prenatal", r"fetal", r"foetal", r"perinatal"),
    },
    "head_or_neck": {
        "ids": (),
        "names": ("abnormality of head or neck",),
        "patterns": (r"head or neck", r"craniofacial", r"facial", r"skull", r"neck"),
    },
    "limbs": {
        "ids": (),
        "names": ("abnormality of limbs",),
        "patterns": (r"limbs?", r"digits?", r"hands?", r"feet", r"foot"),
    },
    "thoracic_cavity": {
        "ids": (),
        "names": ("abnormality of the thoracic cavity",),
        "patterns": (r"thoracic cavity", r"thorax", r"mediastin(?:um|al)", r"pleural?"),
    },
    "voice": {
        "ids": (),
        "names": ("abnormality of the voice",),
        "patterns": (r"voice", r"phonation", r"hoarse(?:ness)?", r"dysphonia"),
    },
    "breast": {
        "ids": (),
        "names": ("abnormality of the breast",),
        "patterns": (r"breast", r"mammary", r"nipples?"),
    },
    "neoplasm": {
        "ids": (),
        "names": ("neoplasm",),
        "patterns": (r"neoplasm", r"tumou?rs?", r"carcinoma", r"malignan(?:t|cy)"),
    },
    "constitutional": {
        "ids": (),
        "names": ("constitutional symptom",),
        "patterns": (r"constitutional symptom", r"fever", r"fatigue", r"malaise"),
    },
    "cellular": {
        "ids": (),
        "names": ("abnormal cellular phenotype",),
        "patterns": (r"cellular phenotype", r"abnormal cell", r"cell death", r"apoptosis"),
    },
}


def _sha256(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_gene(value: Any) -> str:
    return str(value).strip().upper()


def _normalise_mondo(value: Any) -> str:
    text = str(value).strip().upper().replace("_", ":")
    match = re.search(r"MONDO:(\d+)", text)
    return f"MONDO:{int(match.group(1)):07d}" if match else text


def _normalise_hpo(value: Any) -> str:
    text = str(value).strip().upper().replace("_", ":")
    match = re.search(r"HP:(\d{7})", text)
    return f"HP:{match.group(1)}" if match else text


def _extract_hpo_ids(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).upper().replace("_", ":")
    return list(dict.fromkeys(re.findall(r"HP:\d{7}", text)))


def _parse_bool(value: Any) -> Optional[bool]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "present", "positive", "observed", "included"}:
        return True
    if text in {"0", "false", "no", "n", "absent", "negative", "negated", "excluded"}:
        return False
    return None




def _is_severity_term(hpo_id: str, name: str = "") -> bool:
    if hpo_id in SEVERITY_HPO_IDS:
        return True
    normalised_name = str(name).lower().replace("_", " ")
    return any(
        re.search(rf"(^|[^a-z]){re.escape(keyword)}([^a-z]|$)", normalised_name)
        for keyword in SEVERITY_HPO_KEYWORDS
    )


def _explicit_severity_value(name: str) -> Optional[float]:
    low = str(name).lower().replace("_", " ")
    for keyword, value in (("profound", 3.0), ("severe", 2.0), ("moderate", 1.0), ("mild", 0.0)):
        if re.search(rf"(^|[^a-z]){keyword}([^a-z]|$)", low):
            return value
    return None


def _explicit_severity_level(hpo_id: str, name: str) -> Optional[float]:
    canonical = _normalise_hpo(hpo_id)
    if canonical in EXPLICIT_SEVERITY_HPO_LEVELS:
        return float(EXPLICIT_SEVERITY_HPO_LEVELS[canonical])
    return _explicit_severity_value(name)


def _normalise_class_weight(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "off", "unweighted"}:
        return None
    if text == "balanced":
        return "balanced"
    raise ValueError(f"Unsupported class-weight setting: {value!r}")


def _l2_logistic(C: float, max_iter: int = 5000, class_weight: Optional[str] = "balanced") -> LogisticRegression:
    return LogisticRegression(
        C=float(C), solver="lbfgs", class_weight=_normalise_class_weight(class_weight),
        max_iter=int(max_iter)
    )


def _sparse_hstack(parts: Sequence[Any]) -> sparse.csr_matrix:
    converted: List[sparse.csr_matrix] = []
    n_rows: Optional[int] = None
    for part in parts:
        if sparse.issparse(part):
            matrix = part.tocsr()
        else:
            array = np.asarray(part, dtype=float)
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            matrix = sparse.csr_matrix(array)
        if n_rows is None:
            n_rows = matrix.shape[0]
        elif matrix.shape[0] != n_rows:
            raise ValueError("Feature blocks have inconsistent row counts.")
        converted.append(matrix)
    if not converted:
        return sparse.csr_matrix((0, 0), dtype=float)
    return sparse.hstack(converted, format="csr")


@dataclass(frozen=True)
class ModelConfig:
    C: float = 1.0
    mild_C: float = 0.25
    mild_class_weight: str = "none"
    other_class_weight: str = "balanced"
    mild_calibration: str = "isotonic"
    min_count: int = 5
    max_freq: float = 0.95
    max_hops: int = 2
    ancestor_decay: float = 0.75
    ic_weighting: bool = False
    explicit_channel_scale: float = 0.35
    explicit_min_abs_logit: float = 0.15
    absent_channel_scale: float = 0.50
    organ_channel_scale: float = 0.50
    interaction_channel_scale: float = 0.50
    low_burden_channel_scale: float = 0.50
    interaction_min_count: int = 10
    max_interactions: int = 250
    use_organ_burden: bool = True
    use_interactions: bool = True
    use_explicit_channel: bool = True
    use_absent_channel: bool = True
    use_low_burden_channel: bool = True
    severity_term_policy: str = "phenotype"

    @property
    def label(self) -> str:
        return (
            f"C={self.C:g}|mildC={self.mild_C:g}|mildW={self.mild_class_weight}|"
            f"otherW={self.other_class_weight}|mildCal={self.mild_calibration}|"
            f"min={self.min_count}|maxfreq={self.max_freq:g}|"
            f"hops={self.max_hops}|decay={self.ancestor_decay:g}|ic={int(self.ic_weighting)}|"
            f"explicit={self.explicit_channel_scale:g}|expMinLogit={self.explicit_min_abs_logit:g}|"
            f"absent={self.absent_channel_scale:g}|"
            f"organScale={self.organ_channel_scale:g}|interScale={self.interaction_channel_scale:g}|"
            f"lowScale={self.low_burden_channel_scale:g}|"
            f"imin={self.interaction_min_count}|imax={self.max_interactions}|"
            f"organ={int(self.use_organ_burden)}|inter={int(self.use_interactions)}|"
            f"exp={int(self.use_explicit_channel)}|neg={int(self.use_absent_channel)}|"
            f"low={int(self.use_low_burden_channel)}|sevTerms={self.severity_term_policy}"
        )


class OntologyExpander:
    """Parse HPO, canonicalise terms, and provide ancestor information."""

    def __init__(self, excluded: Sequence[str] = ("HP:0000001", "HP:0000118")):
        self.excluded = set(excluded)
        self.parents: Dict[str, List[str]] = {}
        self.alt_to_id: Dict[str, str] = {}
        self.names: Dict[str, str] = {}
        self.enabled = False
        self._ancestor_id_cache: Dict[str, Tuple[str, ...]] = {}
        self._normalised_name_to_ids: Dict[str, List[str]] = {}

    def load(self, obo_path: Optional[str]) -> "OntologyExpander":
        if not obo_path or not os.path.isfile(obo_path):
            print(f"[ontology] no usable .obo at {obo_path!r}; ontology expansion disabled.")
            return self
        records: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None

        def commit() -> None:
            nonlocal current
            if current and current.get("id"):
                records.append(current)
            current = None

        with open(obo_path, encoding="utf-8") as handle:
            for raw in handle:
                line = raw.rstrip("\n")
                if line == "[Term]":
                    commit()
                    current = {"id": None, "name": "", "parents": [], "alt": [], "obsolete": False, "replaced_by": None}
                elif line.startswith("["):
                    commit()
                elif current is not None:
                    if line.startswith("id: HP:"):
                        current["id"] = line[4:].strip()
                    elif line.startswith("name: "):
                        current["name"] = line[6:].strip()
                    elif line.startswith("is_a: HP:"):
                        current["parents"].append(line[6:].split("!")[0].strip())
                    elif line.startswith("alt_id: HP:"):
                        current["alt"].append(line[8:].strip())
                    elif line.startswith("is_obsolete: true"):
                        current["obsolete"] = True
                    elif line.startswith("replaced_by: HP:"):
                        current["replaced_by"] = line[13:].strip()
            commit()

        for record in records:
            term_id = record["id"]
            if record["obsolete"]:
                replacement = record.get("replaced_by")
                if replacement:
                    self.alt_to_id[term_id] = replacement
                    for alt_id in record["alt"]:
                        self.alt_to_id[alt_id] = replacement
                continue
            self.parents[term_id] = list(record["parents"])
            self.names[term_id] = str(record.get("name", ""))
            for alt_id in record["alt"]:
                self.alt_to_id[alt_id] = term_id
        self.enabled = bool(self.parents)
        self._ancestor_id_cache = {}
        self._normalised_name_to_ids = {}
        for term_id, name in self.names.items():
            key = re.sub(r"\s+", " ", str(name).strip().casefold())
            if key:
                self._normalised_name_to_ids.setdefault(key, []).append(term_id)
        print(f"[ontology] parsed {len(self.parents)} terms and {len(self.alt_to_id)} alternate/replaced IDs.")
        return self

    def canonical(self, hpo_id: str) -> str:
        term = _normalise_hpo(hpo_id)
        seen = set()
        while term in self.alt_to_id and term not in seen:
            seen.add(term)
            term = self.alt_to_id[term]
        return term

    def is_known(self, hpo_id: str) -> bool:
        canonical = self.canonical(hpo_id)
        return not self.enabled or canonical in self.parents or canonical in self.names

    def clean_terms(
        self,
        terms: Sequence[Any],
        fallback_names: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        fallback_names = fallback_names or {}
        accepted: List[str] = []
        audit: Dict[str, List[str]] = {
            "invalid_format": [], "unknown_ontology": [], "severity_coded": [],
            "clinical_modifier_excluded": [], "duplicates": []
        }
        seen = set()
        for raw in terms:
            normalised = _normalise_hpo(raw)
            if not HPO_RE.fullmatch(normalised):
                audit["invalid_format"].append(str(raw))
                continue
            canonical = self.canonical(normalised)
            if self.enabled and not self.is_known(canonical):
                audit["unknown_ontology"].append(canonical)
                continue
            if canonical in EXCLUDED_CLINICAL_MODIFIER_HPO_IDS:
                audit["clinical_modifier_excluded"].append(canonical)
                continue
            if canonical in seen:
                audit["duplicates"].append(canonical)
                continue
            seen.add(canonical)
            name = self.names.get(canonical, fallback_names.get(canonical, ""))
            if _is_severity_term(canonical, name):
                audit["severity_coded"].append(canonical)
            accepted.append(canonical)
        return sorted(accepted), audit

    def detect_severity_terms(self, terms, fallback_names=None):
        fallback_names = fallback_names or {}
        return sorted(term for term in terms
                      if _is_severity_term(term, self.names.get(term, fallback_names.get(term, ""))))

    def expand_with_distance(
        self, direct_terms: Sequence[str], max_hops: int, include_severity_ancestors: bool = False
    ) -> Dict[str, int]:
        direct = {self.canonical(term) for term in direct_terms}
        distances: Dict[str, int] = {term: 0 for term in direct}
        if not self.enabled or max_hops <= 0:
            return distances
        for start in direct:
            frontier = {start}
            visited = {start}
            for distance in range(1, max_hops + 1):
                next_frontier = set()
                for node in frontier:
                    for parent in self.parents.get(node, []):
                        parent = self.canonical(parent)
                        if parent in visited or parent in self.excluded:
                            continue
                        visited.add(parent)
                        if not include_severity_ancestors and _is_severity_term(parent, self.names.get(parent, "")):
                            continue
                        next_frontier.add(parent)
                        previous = distances.get(parent)
                        if previous is None or distance < previous:
                            distances[parent] = distance
                frontier = next_frontier
                if not frontier:
                    break
        return distances

    def ids_for_exact_names(self, names: Sequence[str]) -> List[str]:
        found: List[str] = []
        seen = set()
        for name in names:
            key = re.sub(r"\s+", " ", str(name).strip().casefold())
            for term_id in self._normalised_name_to_ids.get(key, []):
                canonical = self.canonical(term_id)
                if canonical not in seen:
                    seen.add(canonical)
                    found.append(canonical)
        return found

    def ancestor_ids(self, term: str, max_hops: int = 80) -> Tuple[str, ...]:
        canonical = self.canonical(term)
        if canonical in self._ancestor_id_cache:
            return self._ancestor_id_cache[canonical]
        if not self.enabled:
            return (canonical,)
        visited = {canonical}
        frontier = {canonical}
        for _ in range(max_hops):
            next_frontier = set()
            for node in frontier:
                for parent in self.parents.get(node, []):
                    parent = self.canonical(parent)
                    if parent not in visited:
                        visited.add(parent)
                        next_frontier.add(parent)
            frontier = next_frontier
            if not frontier:
                break
        result = tuple(sorted(visited))
        self._ancestor_id_cache[canonical] = result
        return result

    def ancestor_names(self, term: str, max_hops: int = 40) -> List[str]:
        canonical = self.canonical(term)
        names = [self.names.get(canonical, "")]
        if not self.enabled:
            return names
        frontier = {canonical}
        visited = {canonical}
        for _ in range(max_hops):
            next_frontier = set()
            for node in frontier:
                for parent in self.parents.get(node, []):
                    parent = self.canonical(parent)
                    if parent in visited:
                        continue
                    visited.add(parent)
                    next_frontier.add(parent)
                    names.append(self.names.get(parent, ""))
            frontier = next_frontier
            if not frontier:
                break
        return names


class PhenotypeVectorizer:
    """Sparse ontology-expanded phenotype representation."""

    def __init__(
        self,
        min_count: int,
        max_freq: float,
        ic_weighting: bool,
        ancestor_decay: float,
        exclude_severity_terms: bool = True,
    ):
        self.min_count = int(min_count)
        self.max_freq = float(max_freq)
        self.ic_weighting = bool(ic_weighting)
        self.ancestor_decay = float(ancestor_decay)
        self.exclude_severity_terms = bool(exclude_severity_terms)
        self.vocab: List[str] = []
        self.index: Dict[str, int] = {}
        self.ic = np.array([], dtype=float)
        self.id_to_name: Dict[str, str] = {}
        self.training_counts: Dict[str, int] = {}

    def fit(self, profiles: Sequence[Dict[str, int]], id_to_name: Dict[str, str]) -> "PhenotypeVectorizer":
        self.id_to_name = dict(id_to_name)
        counts: Counter = Counter()
        for profile in profiles:
            counts.update(profile.keys())
        self.training_counts = dict(counts)
        n_rows = max(len(profiles), 1)
        self.vocab = sorted(
            term for term, count in counts.items()
            if count >= self.min_count
            and count / n_rows <= self.max_freq
            and (
                not self.exclude_severity_terms
                or not _is_severity_term(term, self.id_to_name.get(term, ""))
            )
        )
        self.index = {term: index for index, term in enumerate(self.vocab)}
        frequencies = np.array([counts[term] / n_rows for term in self.vocab], dtype=float)
        frequencies = np.clip(frequencies, 1.0 / n_rows, 1.0)
        self.ic = -np.log2(frequencies) if len(frequencies) else np.array([], dtype=float)
        print(f"[vectorizer] phenotype vocabulary={len(self.vocab)}")
        return self

    def _value(self, index: int, distance: int) -> float:
        value = self.ancestor_decay ** max(int(distance), 0)
        if self.ic_weighting:
            value *= self.ic[index]
        return float(value)

    def transform_one(self, profile: Dict[str, int]) -> sparse.csr_matrix:
        return self.transform([profile])

    def transform(self, profiles: Sequence[Dict[str, int]]) -> sparse.csr_matrix:
        data: List[float] = []
        indices: List[int] = []
        indptr = [0]
        for profile in profiles:
            row_items = sorted(
                (self.index[term], self._value(self.index[term], distance))
                for term, distance in profile.items() if term in self.index
            )
            indices.extend(index for index, _ in row_items)
            data.extend(value for _, value in row_items)
            indptr.append(len(indices))
        return sparse.csr_matrix(
            (np.asarray(data, dtype=float), np.asarray(indices, dtype=int), np.asarray(indptr, dtype=int)),
            shape=(len(profiles), len(self.vocab)), dtype=float,
        )


class DirectTermVectorizer:

    def __init__(self, min_count: int, max_freq: float, scale: float, label: str):
        self.min_count = max(1, int(min_count))
        self.max_freq = float(max_freq)
        self.scale = float(scale)
        self.label = label
        self.vocab: List[str] = []
        self.index: Dict[str, int] = {}

    def fit(self, term_sets: Sequence[Sequence[str]]) -> "DirectTermVectorizer":
        counts: Counter = Counter()
        for terms in term_sets:
            counts.update(set(terms))
        n_rows = max(len(term_sets), 1)
        self.vocab = sorted(
            term for term, count in counts.items()
            if count >= self.min_count and count / n_rows <= self.max_freq
        )
        self.index = {term: index for index, term in enumerate(self.vocab)}
        print(f"[vectorizer] {self.label} vocabulary={len(self.vocab)}")
        return self

    def transform_one(self, terms: Sequence[str]) -> sparse.csr_matrix:
        return self.transform([terms])

    def transform(self, term_sets: Sequence[Sequence[str]]) -> sparse.csr_matrix:
        data: List[float] = []
        indices: List[int] = []
        indptr = [0]
        for terms in term_sets:
            row = sorted({self.index[term] for term in terms if term in self.index})
            indices.extend(row)
            data.extend([self.scale] * len(row))
            indptr.append(len(indices))
        return sparse.csr_matrix(
            (np.asarray(data, dtype=float), np.asarray(indices, dtype=int), np.asarray(indptr, dtype=int)),
            shape=(len(term_sets), len(self.vocab)), dtype=float,
        )


class ExplicitSeverityVectorizer:

    def __init__(self, expander: OntologyExpander, scale: float = 0.35, enabled: bool = True):
        self.expander = expander
        self.scale = float(scale)
        self.enabled = bool(enabled)
        self.id_to_name: Dict[str, str] = {}
        self.vocab: List[str] = list(CLASS_NAMES) if self.enabled else []
        self.index: Dict[str, int] = {name: i for i, name in enumerate(self.vocab)}
        self.feature_names: List[str] = [f"explicit_level:{name}" for name in self.vocab]
        self.training_counts: Dict[str, int] = {}
        self.unmapped_training_terms: Dict[str, int] = {}

    def _level_for_term(self, term: str) -> Optional[int]:
        name = self.id_to_name.get(term, self.expander.names.get(term, ""))
        value = _explicit_severity_level(term, name)
        return int(value) if value is not None else None

    def levels_for_terms(self, terms: Sequence[str]) -> List[int]:
        levels = []
        for term in terms:
            level = self._level_for_term(term)
            if level is not None:
                levels.append(level)
        return sorted(set(levels))

    def fit(self, term_sets: Sequence[Sequence[str]]) -> "ExplicitSeverityVectorizer":
        counts: Counter = Counter()
        unmapped: Counter = Counter()
        for terms in term_sets:
            levels = self.levels_for_terms(terms)
            if levels:
                counts[CLASS_NAMES[max(levels)]] += 1
            for term in terms:
                if self._level_for_term(term) is None:
                    unmapped[term] += 1
        self.training_counts = dict(counts)
        self.unmapped_training_terms = dict(unmapped)
        print(
            "[vectorizer] explicit ordinal levels="
            + ", ".join(f"{name}:{counts.get(name, 0)}" for name in CLASS_NAMES)
            + f"; unmapped severity-coded terms={sum(unmapped.values())}"
        )
        return self

    def transform_one(self, terms: Sequence[str]) -> sparse.csr_matrix:
        return self.transform([terms])

    def transform(self, term_sets: Sequence[Sequence[str]]) -> sparse.csr_matrix:
        if not self.enabled:
            return sparse.csr_matrix((len(term_sets), 0), dtype=float)
        data: List[float] = []
        indices: List[int] = []
        indptr = [0]
        for terms in term_sets:
            levels = self.levels_for_terms(terms)
            if levels:
                level_name = CLASS_NAMES[max(levels)]
                indices.append(self.index[level_name])
                data.append(self.scale)
            indptr.append(len(indices))
        return sparse.csr_matrix(
            (np.asarray(data, dtype=float), np.asarray(indices, dtype=int), np.asarray(indptr, dtype=int)),
            shape=(len(term_sets), len(self.vocab)), dtype=float,
        )


class OrganSystemVectorizer:

    def __init__(self, expander: OntologyExpander, enabled: bool = True, scale: float = 1.0):
        self.expander = expander
        self.enabled = bool(enabled)
        self.scale = float(scale)
        self.id_to_name: Dict[str, str] = {}
        self.mapping_version = "hpo_branch_v1"
        self.systems = list(ORGAN_SYSTEM_BRANCHES)
        self.system_pairs = list(itertools.combinations(self.systems, 2))
        self.feature_names = (
            [f"organ:{system}" for system in self.systems]
            + ["organ:n_systems_fraction", "organ:multisystem"]
            + [f"organ_interaction:{a}&{b}" for a, b in self.system_pairs]
        ) if self.enabled else []
        self._resolved_branch_roots: Optional[Dict[str, Tuple[str, ...]]] = None
        self._compiled_fallback_patterns: Dict[str, Tuple[Any, ...]] = {}

    def _systems_from_names(self, terms: Sequence[str]) -> List[str]:
        found = set()
        for term in terms:
            names = self.expander.ancestor_names(term)
            fallback = self.id_to_name.get(term, "")
            if fallback:
                names = [fallback, *names]
            text = " | ".join(names).lower()
            for system, keywords in LEGACY_ORGAN_SYSTEM_KEYWORDS.items():
                if any(keyword in text for keyword in keywords):
                    found.add(system)
        return sorted(found)

    def _resolve_branch_roots(self) -> Dict[str, Tuple[str, ...]]:
        if self._resolved_branch_roots is not None:
            return self._resolved_branch_roots
        resolved: Dict[str, Tuple[str, ...]] = {}
        for system, specification in ORGAN_SYSTEM_BRANCHES.items():
            roots = set()
            for term_id in specification.get("ids", ()):
                canonical = self.expander.canonical(term_id)
                if not self.expander.enabled or self.expander.is_known(canonical):
                    roots.add(canonical)
            roots.update(self.expander.ids_for_exact_names(specification.get("names", ())))
            resolved[system] = tuple(sorted(roots))
            self._compiled_fallback_patterns[system] = tuple(
                re.compile(pattern, flags=re.IGNORECASE)
                for pattern in specification.get("patterns", ())
            )
        self._resolved_branch_roots = resolved
        summary = ", ".join(f"{system}:{len(roots)}" for system, roots in resolved.items())
        print(f"[organ] mapping=hpo_branch_v1 resolved roots: {summary}")
        return resolved

    def mapping_audit(self) -> Dict[str, Any]:
        if not hasattr(self, "mapping_version"):
            return {"mapping_version": "legacy_substring", "resolved_roots": {}}
        roots = self._resolve_branch_roots()
        return {
            "mapping_version": self.mapping_version,
            "resolved_roots": {system: list(values) for system, values in roots.items()},
            "fallback_systems": [system for system, values in roots.items() if not values],
        }

    def systems_for_terms(self, terms: Sequence[str]) -> List[str]:
        if not hasattr(self, "mapping_version"):
            return self._systems_from_names(terms)
        roots_by_system = self._resolve_branch_roots()
        found = set()
        for term in terms:
            ancestors = set(self.expander.ancestor_ids(term))
            names = self.expander.ancestor_names(term)
            fallback = self.id_to_name.get(term, "")
            if fallback:
                names = [fallback, *names]
            text = " | ".join(name for name in names if name)
            for system in self.systems:
                roots = roots_by_system.get(system, ())
                if roots:
                    if ancestors.intersection(roots):
                        found.add(system)
                elif any(pattern.search(text) for pattern in self._compiled_fallback_patterns.get(system, ())):
                    found.add(system)
        return sorted(found)

    def burden_score(self, terms: Sequence[str]) -> float:
        count = len(self.systems_for_terms(terms))
        if count == 0:
            return 0.0
        if count == 1:
            return 1.0
        if count <= 3:
            return 2.0
        return 3.0

    def transform_one(self, terms: Sequence[str]) -> sparse.csr_matrix:
        return self.transform([terms])

    def transform(self, term_sets: Sequence[Sequence[str]]) -> sparse.csr_matrix:
        if not self.enabled:
            return sparse.csr_matrix((len(term_sets), 0), dtype=float)
        rows: List[List[float]] = []
        for terms in term_sets:
            present = set(self.systems_for_terms(terms))
            values = [1.0 if system in present else 0.0 for system in self.systems]
            values.extend([
                len(present) / max(len(self.systems), 1),
                1.0 if len(present) >= 2 else 0.0,
            ])
            values.extend([1.0 if a in present and b in present else 0.0 for a, b in self.system_pairs])
            rows.append([self.scale * value for value in values])
        return sparse.csr_matrix(np.asarray(rows, dtype=float))


class ObservedLowBurdenVectorizer:

    def __init__(
        self,
        organ: OrganSystemVectorizer,
        expander: OntologyExpander,
        enabled: bool = True,
        scale: float = 0.5,
    ):
        self.organ = organ
        self.expander = expander
        self.enabled = bool(enabled)
        self.scale = float(scale)
        self.id_to_name: Dict[str, str] = {}
        self.feature_names = [
            "low_burden:single_mapped_system",
            "low_burden:observed_only_noncritical_systems",
        ] if self.enabled else []

    def _explicit_values(self, terms: Sequence[str]) -> List[float]:
        values: List[float] = []
        for term in terms:
            name = self.id_to_name.get(term, self.expander.names.get(term, ""))
            value = _explicit_severity_level(term, name)
            if value is not None:
                values.append(float(value))
        return values

    def transform(
        self,
        phenotype_term_sets: Sequence[Sequence[str]],
        explicit_term_sets: Sequence[Sequence[str]],
    ) -> sparse.csr_matrix:
        if not self.enabled:
            return sparse.csr_matrix((len(phenotype_term_sets), 0), dtype=float)
        rows: List[List[float]] = []
        legacy_layout = len(getattr(self, "feature_names", [])) >= 4
        for phenotype_terms, explicit_terms in zip(phenotype_term_sets, explicit_term_sets):
            systems = self.organ.systems_for_terms(phenotype_terms)
            n_systems = len(systems)
            has_mapped_system = n_systems > 0
            critical_systems = {
                "nervous_system", "cardiovascular", "respiratory", "renal_urinary",
                "digestive_hepatic", "immune", "hematologic", "endocrine_metabolic",
            }
            values = [
                1.0 if n_systems == 1 else 0.0,
                1.0 if has_mapped_system and not (set(systems) & critical_systems) else 0.0,
            ]
            if legacy_layout:
                explicit_values = self._explicit_values(explicit_terms)
                values.extend([
                    1.0 if any(value == 0.0 for value in explicit_values) else 0.0,
                    1.0 if explicit_values and max(explicit_values) <= 1.0 else 0.0,
                ])
            rows.append([self.scale * value for value in values])
        return sparse.csr_matrix(np.asarray(rows, dtype=float))

    def transform_one(
        self,
        phenotype_terms: Sequence[str],
        explicit_terms: Sequence[str],
    ) -> sparse.csr_matrix:
        return self.transform([phenotype_terms], [explicit_terms])


class InteractionVectorizer:
    """Select frequent pairwise interactions among supported direct phenotypes."""

    def __init__(self, min_count: int, max_interactions: int, enabled: bool = True, scale: float = 1.0):
        self.min_count = max(1, int(min_count))
        self.max_interactions = max(0, int(max_interactions))
        self.enabled = bool(enabled and self.max_interactions > 0)
        self.scale = float(scale)
        self.pairs: List[Tuple[str, str]] = []
        self.index: Dict[Tuple[str, str], int] = {}

    def fit(self, term_sets: Sequence[Sequence[str]], allowed_terms: Iterable[str]) -> "InteractionVectorizer":
        if not self.enabled:
            return self
        allowed = set(allowed_terms)
        counts: Counter = Counter()
        for terms in term_sets:
            supported = sorted(set(terms) & allowed)
            counts.update(itertools.combinations(supported, 2))
        eligible = [(pair, count) for pair, count in counts.items() if count >= self.min_count]
        eligible.sort(key=lambda item: (-item[1], item[0]))
        self.pairs = [pair for pair, _ in eligible[:self.max_interactions]]
        self.index = {pair: index for index, pair in enumerate(self.pairs)}
        print(f"[vectorizer] phenotype interactions={len(self.pairs)}")
        return self

    @property
    def feature_names(self) -> List[str]:
        return [f"interaction:{a}&{b}" for a, b in self.pairs]

    def transform_one(self, terms: Sequence[str]) -> sparse.csr_matrix:
        return self.transform([terms])

    def transform(self, term_sets: Sequence[Sequence[str]]) -> sparse.csr_matrix:
        data: List[float] = []
        indices: List[int] = []
        indptr = [0]
        for terms in term_sets:
            present = set(terms)
            row = [index for index, (a, b) in enumerate(self.pairs) if a in present and b in present]
            indices.extend(row)
            data.extend([self.scale] * len(row))
            indptr.append(len(indices))
        return sparse.csr_matrix(
            (np.asarray(data, dtype=float), np.asarray(indices, dtype=int), np.asarray(indptr, dtype=int)),
            shape=(len(term_sets), len(self.pairs)), dtype=float,
        )


class ClinicalFeatureBuilder:
    """Fit and transform all requested evidence channels without target leakage."""

    def __init__(self, expander: OntologyExpander, config: ModelConfig):
        self.expander = expander
        self.config = config
        self.severity_term_policy = config.severity_term_policy
        self.explicit_channel_enabled = bool(
            False
        )
        self.phenotype = PhenotypeVectorizer(
            config.min_count,
            config.max_freq,
            config.ic_weighting,
            config.ancestor_decay,
            exclude_severity_terms=False,
        )
        direct_min = max(2, min(config.min_count, 5))
        self.explicit = ExplicitSeverityVectorizer(
            expander, scale=config.explicit_channel_scale,
            enabled=self.explicit_channel_enabled,
        )
        self.absent = DirectTermVectorizer(
            direct_min, 0.995, config.absent_channel_scale, "absent-term"
        )
        self.organ = OrganSystemVectorizer(
            expander, config.use_organ_burden, scale=config.organ_channel_scale
        )
        self.low_burden = ObservedLowBurdenVectorizer(
            self.organ, expander, config.use_low_burden_channel,
            scale=config.low_burden_channel_scale,
        )
        self.interactions = InteractionVectorizer(
            config.interaction_min_count, config.max_interactions,
            config.use_interactions, scale=config.interaction_channel_scale,
        )
        self.feature_names: List[str] = []
        self.slices: Dict[str, Tuple[int, int]] = {}

    def fit(
        self,
        pairs: pd.DataFrame,
        profiles: Sequence[Dict[str, int]],
        id_to_name: Dict[str, str],
    ) -> "ClinicalFeatureBuilder":
        self.phenotype.fit(profiles, id_to_name)
        self.organ.id_to_name = dict(id_to_name)
        self.low_burden.id_to_name = dict(id_to_name)
        self.explicit.id_to_name = dict(id_to_name)
        if self.explicit_channel_enabled:
            self.explicit.fit(pairs["explicit_severity_terms"].tolist())
        if self.config.use_absent_channel:
            self.absent.fit(pairs["absent_terms"].tolist())
        self.interactions.fit(pairs["phenotype_direct_terms"].tolist(), self.phenotype.vocab)

        groups = [
            ("phenotype", [f"phenotype:{term}" for term in self.phenotype.vocab]),
            ("explicit", self.explicit.feature_names),
            ("organ", self.organ.feature_names),
            ("low_burden", self.low_burden.feature_names),
            ("interactions", self.interactions.feature_names),
            ("absent", [f"absent:{term}" for term in self.absent.vocab]),
        ]
        self.feature_names = []
        self.slices = {}
        start = 0
        for name, features in groups:
            end = start + len(features)
            self.slices[name] = (start, end)
            self.feature_names.extend(features)
            start = end
        return self

    def transform(
        self, pairs: pd.DataFrame, profiles: Sequence[Dict[str, int]]
    ) -> Tuple[sparse.csr_matrix, Dict[str, sparse.csr_matrix]]:
        blocks = {
            "phenotype": self.phenotype.transform(profiles),
            "explicit": self.explicit.transform(pairs["explicit_severity_terms"].tolist()),
            "organ": self.organ.transform(pairs["phenotype_direct_terms"].tolist()),
            "low_burden": self.low_burden.transform(
                pairs["phenotype_direct_terms"].tolist(),
                pairs["explicit_severity_terms"].tolist(),
            ),
            "interactions": self.interactions.transform(pairs["phenotype_direct_terms"].tolist()),
            "absent": self.absent.transform(pairs["absent_terms"].tolist()),
        }
        return _sparse_hstack(list(blocks.values())), blocks

    def transform_one(
        self,
        phenotype_direct: Sequence[str],
        explicit_terms: Sequence[str],
        absent_terms: Sequence[str],
        profile: Dict[str, int],
    ) -> Tuple[sparse.csr_matrix, Dict[str, sparse.csr_matrix]]:
        blocks = {
            "phenotype": self.phenotype.transform_one(profile),
            "explicit": self.explicit.transform_one(explicit_terms),
            "organ": self.organ.transform_one(phenotype_direct),
            "low_burden": self.low_burden.transform_one(phenotype_direct, explicit_terms),
            "interactions": self.interactions.transform_one(phenotype_direct),
            "absent": self.absent.transform_one(absent_terms),
        }
        return _sparse_hstack(list(blocks.values())), blocks


class _ConstantBinary:
    def __init__(self, value: int):
        self.value = int(value)

    def predict_proba(self, X: Any) -> np.ndarray:
        result = np.zeros((X.shape[0], 2), dtype=float)
        result[:, self.value] = 1.0
        return result


class _ConstantProbability:
    def __init__(self, probability: float):
        self.probability = float(np.clip(probability, 0.0, 1.0))

    def predict(self, X: Any) -> np.ndarray:
        return np.full(X.shape[0], self.probability, dtype=float)


def _proba1(model: Any, X: Any) -> np.ndarray:
    probabilities = model.predict_proba(X)
    return probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities[:, 0]


def _project_nonincreasing_row(values: Sequence[float]) -> np.ndarray:
    blocks = [[float(value), 1, 1] for value in values]
    index = 0
    while index < len(blocks) - 1:
        if blocks[index][0] + 1e-15 >= blocks[index + 1][0]:
            index += 1
            continue
        left, right = blocks[index], blocks[index + 1]
        weight = left[1] + right[1]
        mean = (left[0] * left[1] + right[0] * right[1]) / weight
        blocks[index:index + 2] = [[mean, weight, left[2] + right[2]]]
        index = max(index - 1, 0)
    projected: List[float] = []
    for mean, _, length in blocks:
        projected.extend([mean] * int(length))
    return np.clip(np.asarray(projected, dtype=float), 0.0, 1.0)


def project_nonincreasing(matrix: np.ndarray) -> np.ndarray:
    return np.vstack([_project_nonincreasing_row(row) for row in np.asarray(matrix, dtype=float)])


class ThresholdCalibrator:
    """Calibrate each ordinal boundary from out-of-fold predictions."""

    def __init__(self, methods: Optional[Sequence[str]] = None):
        supplied = list(methods or ["platt"] * (NUM_CLASSES - 1))
        if len(supplied) != NUM_CLASSES - 1:
            raise ValueError("One calibration method is required per ordinal boundary.")
        self.methods = [str(method).lower() for method in supplied]
        self.models: List[Any] = []

    def fit(self, raw_cumulative: np.ndarray, y: np.ndarray) -> "ThresholdCalibrator":
        self.models = []
        eps = 1e-6
        for threshold in range(NUM_CLASSES - 1):
            target = (np.asarray(y) > threshold).astype(int)
            if np.unique(target).size < 2:
                self.models.append(("constant", _ConstantProbability(float(target.mean()))))
                continue
            probability = np.clip(raw_cumulative[:, threshold], eps, 1.0 - eps)
            requested = self.methods[threshold]
            if (
                requested == "isotonic"
                and len(target) >= 40
                and np.unique(probability).size >= 8
                and min(int(target.sum()), int((1 - target).sum())) >= 10
            ):
                model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                model.fit(probability, target)
                self.models.append(("isotonic", model))
            else:
                logit = np.log(probability / (1.0 - probability)).reshape(-1, 1)
                model = _l2_logistic(1e6, class_weight=None)
                model.fit(logit, target)
                self.models.append(("platt", model))
        return self

    def transform(self, raw_cumulative: np.ndarray) -> np.ndarray:
        eps = 1e-6
        calibrated = []
        for threshold, entry in enumerate(self.models):
            probability = np.clip(raw_cumulative[:, threshold], eps, 1.0 - eps)

            if isinstance(entry, tuple) and len(entry) == 2:
                method, model = entry
            else:
                method, model = "legacy", entry
            if method == "constant" or isinstance(model, _ConstantProbability):
                calibrated.append(model.predict(probability.reshape(-1, 1)))
            elif method == "isotonic":
                calibrated.append(np.asarray(model.predict(probability), dtype=float))
            else:
                logit = np.log(probability / (1.0 - probability)).reshape(-1, 1)
                calibrated.append(model.predict_proba(logit)[:, 1])
        return np.column_stack(calibrated)


class OrdinalThresholdModel:
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 5000,
        mild_C: Optional[float] = None,
        mild_class_weight: str = "none",
        other_class_weight: str = "balanced",
        mild_calibration: str = "isotonic",
    ):
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.mild_C = float(mild_C if mild_C is not None else C)
        self.mild_class_weight = str(mild_class_weight)
        self.other_class_weight = str(other_class_weight)
        self.mild_calibration = str(mild_calibration).lower()
        self.models: List[Any] = []
        self.n_features = 0
        self.calibrator: Optional[ThresholdCalibrator] = None

    def fit(self, X: Any, y: np.ndarray) -> "OrdinalThresholdModel":
        self.n_features = X.shape[1]
        self.models = []
        y = np.asarray(y, dtype=int)
        for threshold in range(NUM_CLASSES - 1):
            target = (y > threshold).astype(int)
            if np.unique(target).size < 2:
                model: Any = _ConstantBinary(int(target.mean() >= 0.5))
            else:
                current_C = self.mild_C if threshold == 0 else self.C
                current_weight = self.mild_class_weight if threshold == 0 else self.other_class_weight
                model = _l2_logistic(current_C, self.max_iter, current_weight)
                model.fit(X, target)
            self.models.append(model)
        return self

    def raw_cumulative_proba(self, X: Any) -> np.ndarray:
        return np.column_stack([_proba1(model, X) for model in self.models])

    def cumulative_proba(self, X: Any, calibrated: bool = True) -> np.ndarray:
        cumulative = self.raw_cumulative_proba(X)
        if calibrated and self.calibrator is not None:
            cumulative = self.calibrator.transform(cumulative)
        return project_nonincreasing(cumulative)

    def class_proba(self, X: Any, calibrated: bool = True) -> np.ndarray:
        cumulative = self.cumulative_proba(X, calibrated)
        probabilities = np.column_stack([
            1.0 - cumulative[:, 0],
            cumulative[:, 0] - cumulative[:, 1],
            cumulative[:, 1] - cumulative[:, 2],
            cumulative[:, 2],
        ])
        probabilities = np.clip(probabilities, 0.0, 1.0)
        totals = probabilities.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        return probabilities / totals

    def expected_severity(self, X: Any, calibrated: bool = True) -> np.ndarray:
        return self.cumulative_proba(X, calibrated).sum(axis=1)

    def fit_calibrator(self, raw_oof: np.ndarray, y: np.ndarray) -> "OrdinalThresholdModel":
        methods = [self.mild_calibration, "platt", "platt"]
        self.calibrator = ThresholdCalibrator(methods).fit(raw_oof, y)
        return self

    def coef_matrix(self) -> np.ndarray:
        rows = []
        for model in self.models:
            if isinstance(model, _ConstantBinary):
                rows.append(np.zeros(self.n_features, dtype=float))
            else:
                rows.append(np.asarray(model.coef_).ravel())
        return np.vstack(rows)




def _ordinal_model_from_config(config: ModelConfig, max_iter: int) -> OrdinalThresholdModel:
    return OrdinalThresholdModel(
        C=config.C,
        max_iter=max_iter,
        mild_C=config.mild_C,
        mild_class_weight=config.mild_class_weight,
        other_class_weight=config.other_class_weight,
        mild_calibration=config.mild_calibration,
    )

def _first_column(columns: Sequence[str], candidates: Sequence[str], required_name: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"severity CSV missing {required_name}; tried {list(candidates)}")


def _row_is_present(source: pd.DataFrame) -> pd.Series:
    result = pd.Series(True, index=source.index, dtype=bool)
    if "is_present" in source.columns:
        parsed = source["is_present"].map(_parse_bool)
        return parsed.fillna(True).astype(bool)
    if "present" in source.columns:
        parsed = source["present"].map(_parse_bool)
        return parsed.fillna(True).astype(bool)
    for column in ("negated", "is_absent", "excluded"):
        if column in source.columns:
            parsed = source[column].map(_parse_bool)
            # For a negation column, true means absent.
            return (~parsed.fillna(False).astype(bool)).astype(bool)
    for column in ("status", "evidence_status", "phenotype_status"):
        if column in source.columns:
            parsed = source[column].map(_parse_bool)
            return parsed.fillna(True).astype(bool)
    return result


import os as _gdm_os, sys as _gdm_sys
_gdm_dir = _gdm_os.path.dirname(_gdm_os.path.abspath(__file__))
if _gdm_dir not in _gdm_sys.path:
    _gdm_sys.path.insert(0, _gdm_dir)
from gene_disease_mapping_filter import (
    apply_rebuilt_mapping as _apply_rebuilt_gene_disease_mapping,
    resolve_mapping_path as _resolve_gene_disease_mapping_path,
)

# HPO-verified gene-disease mapping. See Script/gene_disease_mapping_filter.py.
REBUILT_GENE_DISEASE_MAPPING = "gene_disease_mapping_HPO_rebuilt.csv"
APPLY_REBUILT_GENE_DISEASE_MAPPING = True
INCLUDE_HPO_ADDED_PAIRS = False


def load_pairs(
    severity_csv: str,
    output_dir: Optional[str] = None,
    conflict_policy: str = "exclude",
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, Any]]:
    source = pd.read_csv(severity_csv, low_memory=False)
    gene_col = _first_column(source.columns, ("gene_symbol", "Gene Name", "Gene"), "gene symbol")
    mondo_col = _first_column(source.columns, ("mondo_id", "MONDO_ID", "MONDO ID"), "MONDO ID")
    if APPLY_REBUILT_GENE_DISEASE_MAPPING:
        _mapping_path = _resolve_gene_disease_mapping_path(
            REBUILT_GENE_DISEASE_MAPPING,
            os.path.dirname(os.path.abspath(severity_csv)),
            os.path.dirname(os.path.abspath(__file__)),
        )
        source = _apply_rebuilt_gene_disease_mapping(
            source,
            gene_col=gene_col,
            mondo_col=mondo_col,
            mapping_path=_mapping_path,
            log=None,
            audit_dir=output_dir,
            audit_name="gene_disease_mapping_filter_audit.csv",
            include_added=INCLUDE_HPO_ADDED_PAIRS,
        ).reset_index(drop=True)
    severity_col = _first_column(source.columns, ("severity", "Severity", "target_severity"), "severity")
    hpo_col = _first_column(source.columns, ("hpo_id", "HPO_ID", "Phenotypes_HPO", "HPO_Term"), "HPO identifiers")
    name_col = next((column for column in ("hpo_name", "HPO_Name", "HPO_Term_Name", "HPO_Term") if column in source.columns and column != hpo_col), None)
    absent_col = next((column for column in ("absent_hpo_id", "absent_hpo_terms", "excluded_hpo_id", "negated_hpo_id") if column in source.columns), None)
    dimension_columns = {}

    records: List[Dict[str, Any]] = []
    present_flags = _row_is_present(source)
    for row_index, row in source.iterrows():
        base = {
            "gene_symbol": _normalise_gene(row[gene_col]),
            "mondo_id": _normalise_mondo(row[mondo_col]),
            "severity": str(row[severity_col]).strip().title(),
        }
        for term in _extract_hpo_ids(row[hpo_col]):
            records.append({
                **base, "hpo_id": _normalise_hpo(term),
                "hpo_name": str(row[name_col]).strip() if name_col else "",
                "is_present": bool(present_flags.loc[row_index]),
            })
        if absent_col:
            for term in _extract_hpo_ids(row[absent_col]):
                records.append({
                    **base, "hpo_id": _normalise_hpo(term),
                    "hpo_name": "", "is_present": False,
                })

    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No HPO identifiers were found in the severity CSV.")
    frame = frame[frame["severity"].isin(SEVERITY_MAP)].copy()
    frame["sev_num"] = frame["severity"].map(SEVERITY_MAP).astype(int)

    id_to_name: Dict[str, str] = {}
    for term, name in zip(frame["hpo_id"], frame["hpo_name"]):
        if HPO_RE.fullmatch(str(term)) and str(name).strip() and term not in id_to_name:
            id_to_name[str(term)] = str(name).strip()

    conflict = frame.groupby("mondo_id")["sev_num"].nunique().reset_index(name="n_unique_labels")
    conflicting_diseases = set(
        conflict.loc[conflict["n_unique_labels"] > 1, "mondo_id"].astype(str)
    )
    if conflicting_diseases and conflict_policy == "exclude":
        frame = frame.loc[~frame["mondo_id"].isin(conflicting_diseases)].copy()
    elif conflicting_diseases and conflict_policy != "mode":
        raise ValueError(f"Unknown conflict policy: {conflict_policy}")

    rows: List[Dict[str, Any]] = []
    polarity_conflicts = 0
    gene_disease_pairs = frame[["gene_symbol", "mondo_id"]].drop_duplicates()
    for mondo, group in frame.groupby("mondo_id", sort=False):
        severity_mode = group["sev_num"].mode()
        severity = int(severity_mode.iloc[0])
        associated_genes = sorted(set(group["gene_symbol"].astype(str)))
        present = set(group.loc[group["is_present"], "hpo_id"].astype(str))
        absent = set(group.loc[~group["is_present"], "hpo_id"].astype(str))
        both = present & absent
        if both:
            polarity_conflicts += len(both)
            present -= both
            absent -= both
        if not present:
            continue
        row: Dict[str, Any] = {
            "mondo_id": mondo,
            "associated_genes": associated_genes,
            "n_associated_genes": len(associated_genes),
            "sev_num": severity,
            "raw_present_terms": sorted(term for term in present if HPO_RE.fullmatch(term)),
            "raw_absent_terms": sorted(term for term in absent if HPO_RE.fullmatch(term)),
        }
        rows.append(row)

    pairs = pd.DataFrame(rows).reset_index(drop=True)
    if pairs.empty:
        raise ValueError("No eligible disease records remained after curation.")
    audit = {
        "n_source_rows": int(len(source)),
        "n_hpo_records": int(len(frame)),
        "n_gene_disease_pairs": int(len(gene_disease_pairs)),
        "n_diseases": int(len(pairs)),
        "n_pairs": int(len(pairs)),  # compatibility alias; one row now equals one disease
        "n_conflicting_label_diseases": int(len(conflicting_diseases)),
        "n_conflicting_label_pairs": int(len(conflicting_diseases)),  # compatibility alias
        "n_present_absent_term_conflicts_removed": int(polarity_conflicts),
        "negative_annotation_column_detected": bool(absent_col or any(column in source.columns for column in ("is_present", "present", "negated", "is_absent", "excluded", "status"))),
        "dimension_columns": dimension_columns,
    }
    if output_dir:
        pd.DataFrame([audit]).to_csv(os.path.join(output_dir, "input_curation_summary.csv"), index=False)
    return pairs, id_to_name, audit


def prepare_terms(
    pairs: pd.DataFrame,
    expander: OntologyExpander,
    id_to_name: Dict[str, str],
    severity_term_policy: str = "phenotype",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    prepared = pairs.copy()
    phenotype_sets: List[List[str]] = []
    explicit_sets: List[List[str]] = []
    detected_severity_sets: List[List[str]] = []
    absent_sets: List[List[str]] = []
    all_present_sets: List[List[str]] = []
    audit_counts: Counter = Counter()

    for present_raw, absent_raw in zip(prepared["raw_present_terms"], prepared["raw_absent_terms"]):
        present, present_audit = expander.clean_terms(present_raw, id_to_name)
        absent, absent_audit = expander.clean_terms(absent_raw, id_to_name)
        conflicts = set(present) & set(absent)
        if conflicts:
            present = [term for term in present if term not in conflicts]
            absent = [term for term in absent if term not in conflicts]
            audit_counts["present_absent_conflicts_removed"] += len(conflicts)
        detected_severity = expander.detect_severity_terms(present, id_to_name)
        phenotype, explicit = sorted(set(present)), []
        phenotype_sets.append(phenotype)
        explicit_sets.append(explicit)
        detected_severity_sets.append(detected_severity)
        absent_sets.append(sorted(absent))
        all_present_sets.append(sorted(present))
        for key, values in present_audit.items():
            audit_counts[f"present_{key}"] += len(values)
        for key, values in absent_audit.items():
            audit_counts[f"absent_{key}"] += len(values)

    prepared["phenotype_direct_terms"] = phenotype_sets
    prepared["explicit_severity_terms"] = explicit_sets
    prepared["detected_severity_terms"] = detected_severity_sets
    prepared["absent_terms"] = absent_sets
    prepared["all_present_terms"] = all_present_sets
    prepared["direct_terms"] = phenotype_sets  # compatibility alias
    prepared["n_direct"] = prepared["phenotype_direct_terms"].apply(len)
    prepared["n_all_present"] = prepared["all_present_terms"].apply(len)
    before = len(prepared)
    prepared = prepared[prepared["n_all_present"] > 0].reset_index(drop=True)
    audit_counts["pairs_removed_no_usable_present_terms"] = before - len(prepared)
    return prepared, dict(audit_counts)


def expansion_profiles(
    pairs: pd.DataFrame,
    expander: OntologyExpander,
    max_hops: int,
    severity_term_policy: str = "phenotype",
) -> List[Dict[str, int]]:
    return [
        expander.expand_with_distance(
            terms,
            max_hops=max_hops,
            include_severity_ancestors=severity_term_policy == "phenotype",
        )
        for terms in pairs["phenotype_direct_terms"]
    ]


def _split_score(all_y: np.ndarray, train_y: np.ndarray, test_y: np.ndarray, target_fraction: float) -> float:
    overall = np.bincount(all_y, minlength=NUM_CLASSES) / max(len(all_y), 1)
    train_prop = np.bincount(train_y, minlength=NUM_CLASSES) / max(len(train_y), 1)
    test_prop = np.bincount(test_y, minlength=NUM_CLASSES) / max(len(test_y), 1)
    return float(np.abs(train_prop - overall).sum() + np.abs(test_prop - overall).sum() + 3 * abs(len(test_y) / len(all_y) - target_fraction))


def optimised_disease_disjoint_split(
    pairs: pd.DataFrame,
    test_fraction: float,
    seed: int,
    attempts: int = 150,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split disease-level records while preserving approximate class balance."""
    y = pairs["sev_num"].to_numpy(dtype=int)
    groups = pairs["mondo_id"].to_numpy()
    best: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
    for offset in range(attempts):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed + offset)
        train_idx, test_idx = next(splitter.split(pairs, y, groups))
        if np.unique(y[train_idx]).size < NUM_CLASSES or np.unique(y[test_idx]).size < NUM_CLASSES:
            continue
        score = _split_score(y, y[train_idx], y[test_idx], test_fraction)
        if best is None or score < best[0]:
            best = (score, train_idx, test_idx)
    if best is None:
        raise ValueError("Could not create a disease-disjoint split containing all four classes.")
    _, train_idx, test_idx = best
    development = pairs.iloc[train_idx].reset_index(drop=True)
    test = pairs.iloc[test_idx].reset_index(drop=True)
    overlap = set(development["mondo_id"]) & set(test["mondo_id"])
    if overlap:
        raise RuntimeError(
            f"Disease-disjoint split audit failed: {len(overlap)} MONDO IDs overlap."
        )
    return development, test


def make_grouped_cv_splits(pairs: pd.DataFrame, n_splits: int, repeats: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    y = pairs["sev_num"].to_numpy(dtype=int)
    groups = pairs["mondo_id"].to_numpy()
    requested = max(2, min(int(n_splits), np.unique(groups).size))
    for split_count in range(requested, 1, -1):
        all_splits: List[Tuple[np.ndarray, np.ndarray]] = []
        feasible = True
        for repeat in range(repeats):
            accepted = None
            for attempt in range(60):
                splitter = StratifiedGroupKFold(
                    n_splits=split_count, shuffle=True,
                    random_state=seed + 1009 * repeat + attempt,
                )
                candidate = list(splitter.split(np.zeros(len(pairs)), y, groups))
                if all(
                    np.unique(y[train_idx]).size == NUM_CLASSES
                    and np.unique(y[val_idx]).size == NUM_CLASSES
                    and not (set(groups[train_idx]) & set(groups[val_idx]))
                    for train_idx, val_idx in candidate
                ):
                    accepted = candidate
                    break
            if accepted is None:
                feasible = False
                break
            all_splits.extend(accepted)
        if feasible:
            return all_splits
    raise ValueError("Could not create grouped CV folds containing all four classes.")


def evaluate_model(model: OrdinalThresholdModel, X: Any, y: np.ndarray, calibrated: bool = True) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    expected = model.expected_severity(X, calibrated)
    prediction = np.clip(np.rint(expected), 0, NUM_CLASSES - 1).astype(int)
    probabilities = model.class_proba(X, calibrated)
    cumulative = model.cumulative_proba(X, calibrated)
    mild_mask = y == 0
    mild_expected = expected[mild_mask]
    mild_probabilities = probabilities[mild_mask, 0]
    return {
        "qwk": float(cohen_kappa_score(y, prediction, weights="quadratic")),
        "mcc": float(matthews_corrcoef(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "mae_expected": float(mean_absolute_error(y, expected)),
        "multiclass_log_loss": float(log_loss(y, probabilities, labels=list(range(NUM_CLASSES)))),
        "ordinal_brier": float(np.mean([
            np.mean((cumulative[:, threshold] - (y > threshold)) ** 2)
            for threshold in range(NUM_CLASSES - 1)
        ])),
        "mild_mae_expected": float(np.mean(mild_expected)) if mild_expected.size else float("nan"),
        "mild_median_expected": float(np.median(mild_expected)) if mild_expected.size else float("nan"),
        "mild_p95_expected": float(np.percentile(mild_expected, 95)) if mild_expected.size else float("nan"),
        "mild_mean_probability": float(np.mean(mild_probabilities)) if mild_probabilities.size else float("nan"),
        "mild_class_recall": float(np.mean(prediction[mild_mask] == 0)) if mild_mask.any() else float("nan"),
        "mild_boundary_brier": float(np.mean(cumulative[mild_mask, 0] ** 2)) if mild_mask.any() else float("nan"),
        "n": int(len(y)),
    }


def _fit_fold(
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    train_profiles: Sequence[Dict[str, int]],
    val_profiles: Sequence[Dict[str, int]],
    expander: OntologyExpander,
    id_to_name: Dict[str, str],
    config: ModelConfig,
    max_iter: int,
    channel: str,
) -> Tuple[OrdinalThresholdModel, ClinicalFeatureBuilder, sparse.csr_matrix, sparse.csr_matrix]:
    builder = ClinicalFeatureBuilder(expander, config).fit(train_pairs, train_profiles, id_to_name)
    X_train_combined, train_blocks = builder.transform(train_pairs, train_profiles)
    X_val_combined, val_blocks = builder.transform(val_pairs, val_profiles)
    if channel == "phenotype":
        X_train, X_val = train_blocks["phenotype"], val_blocks["phenotype"]
    else:
        X_train, X_val = X_train_combined, X_val_combined
    if X_train.shape[1] == 0:
        raise ValueError(f"No features survived for {channel} channel.")
    model = _ordinal_model_from_config(config, max_iter).fit(
        X_train, train_pairs["sev_num"].to_numpy(dtype=int)
    )
    return model, builder, X_train, X_val


def generate_oof_raw(
    pairs: pd.DataFrame,
    profiles: Sequence[Dict[str, int]],
    expander: OntologyExpander,
    id_to_name: Dict[str, str],
    config: ModelConfig,
    max_iter: int,
    cv_folds: int,
    channel: str,
) -> np.ndarray:
    splits = make_grouped_cv_splits(pairs, cv_folds, 1, RANDOM_SEED + (7001 if channel == "combined" else 8001))
    raw = np.full((len(pairs), NUM_CLASSES - 1), np.nan, dtype=float)
    for train_idx, val_idx in splits:
        train_pairs = pairs.iloc[train_idx]
        val_pairs = pairs.iloc[val_idx]
        train_profiles = [profiles[int(index)] for index in train_idx]
        val_profiles = [profiles[int(index)] for index in val_idx]
        model, _, _, X_val = _fit_fold(
            train_pairs, val_pairs, train_profiles, val_profiles,
            expander, id_to_name, config, max_iter, channel,
        )
        raw[val_idx] = model.raw_cumulative_proba(X_val)
    if np.isnan(raw).any():
        raise RuntimeError(f"OOF predictions did not cover all rows for {channel} model.")
    return raw


def _build_candidate_configs(args: argparse.Namespace) -> List[ModelConfig]:
    use_explicit_channel = bool(
        False
    )
    baseline = ModelConfig(
        C=args.C,
        mild_C=args.mild_C,
        mild_class_weight=args.mild_class_weight,
        other_class_weight=args.other_class_weight,
        mild_calibration=args.mild_calibration,
        min_count=args.min_count,
        max_freq=args.max_freq,
        max_hops=0 if args.no_expand else args.max_hops,
        ancestor_decay=args.ancestor_decay,
        ic_weighting=args.ic_weighting,
        explicit_channel_scale=args.explicit_channel_scale,
        explicit_min_abs_logit=args.explicit_min_abs_logit,
        absent_channel_scale=args.absent_channel_scale,
        organ_channel_scale=args.organ_channel_scale,
        interaction_channel_scale=args.interaction_channel_scale,
        low_burden_channel_scale=args.low_burden_channel_scale,
        interaction_min_count=args.interaction_min_count,
        max_interactions=args.max_interactions,
        use_organ_burden=not args.disable_organ_burden,
        use_interactions=not args.disable_interactions,
        use_explicit_channel=use_explicit_channel,
        use_absent_channel=not args.disable_absent_channel,
        use_low_burden_channel=not args.disable_low_burden_channel,
        severity_term_policy=args.severity_term_policy,
    )
    if args.no_tune:
        return [baseline]
    ic_values = [False, True] if args.compare_ic else [args.ic_weighting]
    mild_calibration_values = [args.mild_calibration]
    all_configs = [
        ModelConfig(
            C=float(C), mild_C=float(mild_C), mild_class_weight=str(mild_weight),
            other_class_weight=args.other_class_weight,
            mild_calibration=str(mild_calibration),
            min_count=int(min_count), max_freq=float(max_freq),
            max_hops=int(max_hops), ancestor_decay=float(decay), ic_weighting=bool(ic),
            explicit_channel_scale=args.explicit_channel_scale,
            explicit_min_abs_logit=args.explicit_min_abs_logit,
            absent_channel_scale=args.absent_channel_scale,
            organ_channel_scale=args.organ_channel_scale,
            interaction_channel_scale=args.interaction_channel_scale,
            low_burden_channel_scale=args.low_burden_channel_scale,
            interaction_min_count=args.interaction_min_count,
            max_interactions=args.max_interactions,
            use_organ_burden=not args.disable_organ_burden,
            use_interactions=not args.disable_interactions,
            use_explicit_channel=use_explicit_channel,
            use_absent_channel=not args.disable_absent_channel,
            use_low_burden_channel=not args.disable_low_burden_channel,
            severity_term_policy=args.severity_term_policy,
        )
        for C, mild_C, mild_weight, mild_calibration, min_count, max_freq, max_hops, decay, ic in itertools.product(
            args.C_grid, args.mild_C_grid, args.mild_class_weight_grid,
            mild_calibration_values, args.min_count_grid, args.max_freq_grid,
            ([0] if args.no_expand else args.max_hops_grid),
            args.ancestor_decay_grid, ic_values,
        )
    ]
    unique = {config.label: config for config in all_configs}
    unique[baseline.label] = baseline
    configs = list(unique.values())
    if len(configs) <= args.max_configs:
        return sorted(configs, key=lambda config: config.label)
    rng = np.random.RandomState(RANDOM_SEED)
    others = [config for config in configs if config != baseline]
    chosen = [baseline] + [others[int(index)] for index in rng.choice(len(others), args.max_configs - 1, replace=False)]
    return chosen


def select_model_config(
    development: pd.DataFrame,
    expander: OntologyExpander,
    id_to_name: Dict[str, str],
    args: argparse.Namespace,
) -> Tuple[ModelConfig, pd.DataFrame]:
    configs = _build_candidate_configs(args)
    splits = make_grouped_cv_splits(development, args.cv_folds, args.cv_repeats, RANDOM_SEED)
    profile_cache = {
        hops: expansion_profiles(
            development, expander, hops, severity_term_policy=args.severity_term_policy
        )
        for hops in sorted({config.max_hops for config in configs})
    }
    rows: List[Dict[str, Any]] = []
    for config_index, config in enumerate(configs, 1):
        profiles = profile_cache[config.max_hops]
        fold_metrics = []
        failure = ""
        for fold_index, (train_idx, val_idx) in enumerate(splits):
            try:
                model, _, _, X_val = _fit_fold(
                    development.iloc[train_idx], development.iloc[val_idx],
                    [profiles[int(index)] for index in train_idx],
                    [profiles[int(index)] for index in val_idx],
                    expander, id_to_name, config, args.max_iter, "combined",
                )
                fold_metrics.append(evaluate_model(
                    model, X_val, development.iloc[val_idx]["sev_num"].to_numpy(dtype=int), calibrated=False
                ))
            except Exception as exc:
                failure = f"fold {fold_index}: {exc}"
                break
        complete = len(fold_metrics) == len(splits)
        row: Dict[str, Any] = {**asdict(config), "config_label": config.label, "complete": complete, "failure": failure}
        metrics_to_average = (
            "qwk", "mcc", "balanced_accuracy", "mae_expected",
            "multiclass_log_loss", "ordinal_brier", "mild_mae_expected",
            "mild_median_expected", "mild_p95_expected", "mild_mean_probability",
            "mild_class_recall", "mild_boundary_brier",
        )
        for metric in metrics_to_average:
            values = [item[metric] for item in fold_metrics]
            row[f"mean_{metric}"] = float(np.nanmean(values)) if complete else np.nan
        row["selection_score"] = (
            row["mean_qwk"]
            - float(args.low_end_penalty_weight) * row["mean_mild_mae_expected"]
            - 0.05 * row["mean_ordinal_brier"]
        ) if complete else np.nan
        rows.append(row)
        if complete:
            print(
                f"[selection] {config_index}/{len(configs)} "
                f"score={row['selection_score']:.4f} QWK={row['mean_qwk']:.4f} "
                f"MildMAE={row['mean_mild_mae_expected']:.4f} {config.label}"
            )
        else:
            print(f"[selection] rejected {config.label}: {failure}")
    results = pd.DataFrame(rows)
    valid = results[results["complete"]].sort_values(
        ["selection_score", "mean_qwk", "mean_mild_mae_expected", "mean_ordinal_brier"],
        ascending=[False, False, True, True],
    )
    if valid.empty:
        raise RuntimeError("Every model configuration failed grouped cross-validation.")
    winner_label = str(valid.iloc[0]["config_label"])
    winner = next(config for config in configs if config.label == winner_label)
    results.to_csv(os.path.join(args.output, "model_selection_results.csv"), index=False)
    print(f"[selection] selected {winner.label}")
    return winner, results


class SeverityScorer:
    """Saved deployment artifact and scoring interface."""

    def __init__(self, expander: OntologyExpander, builder: ClinicalFeatureBuilder, config: ModelConfig):
        self.expander = expander
        self.builder = builder
        self.vectorizer = builder.phenotype  # compatibility alias
        self.config = config
        self.pheno_model: Optional[OrdinalThresholdModel] = None
        self.combined_model: Optional[OrdinalThresholdModel] = None
        self.dimension_models: Dict[str, OrdinalThresholdModel] = {}
        self.dimension_model_calibrated: Dict[str, bool] = {}
        self.combined_feature_weights: Dict[str, List[float]] = {}
        self.term_weight_by_threshold: Dict[str, List[float]] = {}
        self.min_supported_direct_terms = 1
        self.min_recognised_expanded_terms = 1
        self.metadata: Dict[str, Any] = {}

    def _prepare_profile(self, present_hpo_terms: Sequence[str], absent_hpo_terms: Sequence[str]) -> Dict[str, Any]:
        present, audit = self.expander.clean_terms(present_hpo_terms, self.vectorizer.id_to_name)
        absent, absent_audit = self.expander.clean_terms(absent_hpo_terms, self.vectorizer.id_to_name)
        conflicts = sorted(set(present) & set(absent))
        if conflicts:
            present = [term for term in present if term not in conflicts]
            absent = [term for term in absent if term not in conflicts]
        severity_term_policy = self.config.severity_term_policy
        detected_severity = self.expander.detect_severity_terms(present, self.vectorizer.id_to_name)
        phenotype, explicit = sorted(set(present)), []
        include_severity_ancestors = severity_term_policy == "phenotype"
        profile = self.expander.expand_with_distance(
            phenotype,
            self.config.max_hops,
            include_severity_ancestors=include_severity_ancestors,
        )
        X_combined, blocks = self.builder.transform_one(phenotype, explicit, absent, profile)
        supported_direct = []
        for term in phenotype:
            one = self.expander.expand_with_distance(
                [term],
                self.config.max_hops,
                include_severity_ancestors=include_severity_ancestors,
            )
            if any(candidate in self.vectorizer.index for candidate in one):
                supported_direct.append(term)
        return {
            "accepted_present": sorted(present),
            "phenotype_direct": phenotype,
            "explicit_terms": explicit,
            "detected_severity_terms": detected_severity,
            "accepted_absent": sorted(absent),
            "audit": audit,
            "absent_audit": absent_audit,
            "conflicts": conflicts,
            "profile": profile,
            "X_combined": X_combined,
            "blocks": blocks,
            "supported_direct": sorted(supported_direct),
            "recognised_expanded": sorted(term for term in profile if term in self.vectorizer.index),
        }

    @staticmethod
    def _prediction_dict(model: Optional[OrdinalThresholdModel], X: sparse.csr_matrix, calibrated: bool = True) -> Dict[str, Any]:
        if model is None or X.shape[1] == 0:
            return {"expected": None, "probabilities": None}
        return {
            "expected": float(model.expected_severity(X, calibrated)[0]),
            "probabilities": model.class_proba(X, calibrated)[0],
        }

    def _explicit_signal(self, terms: Sequence[str]) -> Dict[str, Any]:
        values = []
        details = []
        for term in terms:
            name = self.vectorizer.id_to_name.get(term, self.expander.names.get(term, ""))
            value = _explicit_severity_level(term, name)
            details.append({"hpo_id": term, "name": name, "mapped_0_3": value})
            if value is not None:
                values.append(value)
        return {
            "expected_severity_0_3": float(np.mean(values)) if values else None,
            "maximum_severity_0_3": float(max(values)) if values else None,
            "terms": details,
        }

    def score_profile(
        self,
        present_hpo_terms: Sequence[str],
        absent_hpo_terms: Optional[Sequence[str]] = None,
        allow_low_information: bool = False,
    ) -> Dict[str, Any]:
        """Score HPO terms with the fitted disease-level phenotype model."""
        present_hpo_terms = list(present_hpo_terms)
        absent_hpo_terms = list(absent_hpo_terms or [])
        prepared = self._prepare_profile(present_hpo_terms, absent_hpo_terms)
        X_pheno = _sparse_hstack([prepared["blocks"]["phenotype"], prepared["blocks"]["explicit"]])
        X_combined = prepared["X_combined"]
        nonzero_combined = int(X_combined.nnz)
        low_information = nonzero_combined == 0
        conflicts = prepared["conflicts"]
        warnings_list: List[str] = []
        if conflicts:
            warnings_list.append("The same HPO term was supplied as both present and absent; conflicting terms were removed.")
        if absent_hpo_terms and not self.builder.absent.vocab:
            warnings_list.append("Absent terms were retained and audited, but the training data had insufficient negative annotations to learn absent-term effects.")
        if low_information:
            warnings_list.append("No supplied evidence contributes to the trained feature space.")

        status = (
            "insufficient_information" if low_information and not allow_low_information
            else "scored_with_conflicts_removed" if conflicts
            else "scored_with_low_information_override" if low_information
            else "scored"
        )
        output: Dict[str, Any] = {
            "status": status,
            "warnings": warnings_list,
            "severity_term_policy": self.config.severity_term_policy,
            "input_audit": prepared["audit"],
            "absent_input_audit": prepared["absent_audit"],
            "present_absent_conflicts": conflicts,
            "accepted_present_terms": prepared["accepted_present"],
            "phenotype_direct_terms": prepared["phenotype_direct"],
            "explicit_severity_terms": prepared["explicit_terms"],
            "severity_naming_terms": prepared["detected_severity_terms"],
            "accepted_absent_terms": prepared["accepted_absent"],
            "n_supported_phenotype_direct_terms": len(prepared["supported_direct"]),
            "n_recognised_expanded_terms": len(prepared["recognised_expanded"]),
            "learned_absent_channel_available": bool(self.builder.absent.vocab),
            "severity_score_0_1": None,
            "expected_severity_0_3": None,
        }
        if low_information and not allow_low_information:
            output["direct_term_leave_one_out_effects"] = []
            output["absent_term_leave_one_out_effects"] = []
            return output

        phenotype_prediction = self._prediction_dict(self.pheno_model, X_pheno, True)
        combined_prediction = self._prediction_dict(self.combined_model, X_combined, True)
        headline = combined_prediction if combined_prediction["expected"] is not None else phenotype_prediction
        headline_expected = headline["expected"]
        output["expected_severity_0_3"] = headline_expected
        output["severity_score_0_1"] = float(np.clip(headline_expected / 3.0, 0.0, 1.0)) if headline_expected is not None else None
        output["severity_score_source"] = "combined_multichannel" if combined_prediction["expected"] is not None else "phenotype_only"
        output["phenotype_only_expected_severity_0_3"] = phenotype_prediction["expected"]
        output["combined_expected_severity_0_3"] = combined_prediction["expected"]
        if phenotype_prediction["probabilities"] is not None:
            output["phenotype_only_class_probabilities"] = {
                name: float(value) for name, value in zip(CLASS_NAMES, phenotype_prediction["probabilities"])
            }
        if combined_prediction["probabilities"] is not None:
            output["class_probabilities"] = {
                name: float(value) for name, value in zip(CLASS_NAMES, combined_prediction["probabilities"])
            }

        affected_systems = self.builder.organ.systems_for_terms(prepared["phenotype_direct"])
        explicit_signal = self._explicit_signal(prepared["detected_severity_terms"])
        learned_dimensions: Dict[str, Any] = {}
        output["severity_dimensions"] = {
            "phenotype_derived": {
                "expected_severity_0_3": phenotype_prediction["expected"],
                "score_0_1": phenotype_prediction["expected"] / 3.0 if phenotype_prediction["expected"] is not None else None,
            },
            "explicit_annotation_signal": explicit_signal,
            "organ_system_burden": {
                "score_0_3": self.builder.organ.burden_score(prepared["phenotype_direct"]),
                "affected_systems": affected_systems,
                "n_affected_systems": len(affected_systems),
            },
            "observed_low_burden_evidence": {
                "available": bool(self.builder.low_burden.feature_names),
                "active_features": [
                    self.builder.low_burden.feature_names[int(index)]
                    for index in prepared["blocks"]["low_burden"].tocoo().col
                ] if self.builder.low_burden.feature_names else [],
                "note": (
                    "Uses observed evidence only; undocumented findings are not treated as absent."
                ),
            },
            "overall_combined": {
                "expected_severity_0_3": combined_prediction["expected"],
                "score_0_1": combined_prediction["expected"] / 3.0 if combined_prediction["expected"] is not None else None,
            },
            "learned_clinical_dimensions": learned_dimensions,
        }

        # Patient-specific leave-one-out effects on the combined score.
        direct_effects = []
        for term in prepared["accepted_present"]:
            reduced = [candidate for candidate in prepared["accepted_present"] if candidate != term]
            reduced_prepared = self._prepare_profile(reduced, prepared["accepted_absent"])
            reduced_prediction = self._prediction_dict(self.combined_model, reduced_prepared["X_combined"], True)
            if combined_prediction["expected"] is not None and reduced_prediction["expected"] is not None:
                direct_effects.append({
                    "hpo_id": term,
                    "name": self.vectorizer.id_to_name.get(term, self.expander.names.get(term, "")),
                    "channel": "explicit_severity" if term in prepared["explicit_terms"] else "phenotype",
                    "delta_expected_severity_when_present": float(combined_prediction["expected"] - reduced_prediction["expected"]),
                })
        direct_effects.sort(key=lambda item: abs(item["delta_expected_severity_when_present"]), reverse=True)
        output["direct_term_leave_one_out_effects"] = direct_effects[:20]

        absent_effects = []
        for term in prepared["accepted_absent"]:
            reduced_absent = [candidate for candidate in prepared["accepted_absent"] if candidate != term]
            reduced_prepared = self._prepare_profile(prepared["accepted_present"], reduced_absent)
            reduced_prediction = self._prediction_dict(self.combined_model, reduced_prepared["X_combined"], True)
            if combined_prediction["expected"] is not None and reduced_prediction["expected"] is not None:
                absent_effects.append({
                    "hpo_id": term,
                    "name": self.vectorizer.id_to_name.get(term, self.expander.names.get(term, "")),
                    "delta_expected_severity_when_absent": float(combined_prediction["expected"] - reduced_prediction["expected"]),
                })
        absent_effects.sort(key=lambda item: abs(item["delta_expected_severity_when_absent"]), reverse=True)
        output["absent_term_leave_one_out_effects"] = absent_effects[:20]

        contributions = []
        if self.combined_model is not None:
            row = X_combined.tocoo()
            boundary_names = ("severity_gt_0", "severity_gt_1", "severity_gt_2")
            for column, value in zip(row.col, row.data):
                feature = self.builder.feature_names[int(column)]
                weights = self.combined_feature_weights.get(feature, [0.0, 0.0, 0.0])
                contributions.append({
                    "feature": feature,
                    "feature_value": float(value),
                    "ordinal_boundary_contributions": {
                        name: float(value * weight) for name, weight in zip(boundary_names, weights)
                    },
                })
            contributions.sort(
                key=lambda item: max(abs(value) for value in item["ordinal_boundary_contributions"].values()),
                reverse=True,
            )
        output["top_combined_feature_contributions"] = contributions[:30]
        return output

    def score_patient(self, hpo_terms: Sequence[str], absent_hpo_terms: Optional[Sequence[str]] = None, allow_low_information: bool = False) -> Dict[str, Any]:
        return self.score_profile(hpo_terms, absent_hpo_terms, allow_low_information)


def fit_scorer(
    pairs: pd.DataFrame,
    profiles: Sequence[Dict[str, int]],
    expander: OntologyExpander,
    id_to_name: Dict[str, str],
    config: ModelConfig,
    max_iter: int,
    cv_folds: int,
) -> Tuple[SeverityScorer, sparse.csr_matrix, sparse.csr_matrix]:
    builder = ClinicalFeatureBuilder(expander, config).fit(pairs, profiles, id_to_name)
    X_combined, blocks = builder.transform(pairs, profiles)
    X_pheno = blocks["phenotype"]
    y = pairs["sev_num"].to_numpy(dtype=int)
    scorer = SeverityScorer(expander, builder, config)

    if X_pheno.shape[1] > 0:
        pheno_model = _ordinal_model_from_config(config, max_iter).fit(X_pheno, y)
        pheno_raw = generate_oof_raw(pairs, profiles, expander, id_to_name, config, max_iter, cv_folds, "phenotype")
        pheno_model.fit_calibrator(pheno_raw, y)
        scorer.pheno_model = pheno_model
        coef = pheno_model.coef_matrix()
        scorer.term_weight_by_threshold = {
            term: [float(value) for value in coef[:, index]]
            for index, term in enumerate(builder.phenotype.vocab)
        }

    if X_combined.shape[1] == 0:
        raise ValueError("No combined features survived vectorisation.")
    combined_model = _ordinal_model_from_config(config, max_iter).fit(X_combined, y)
    combined_raw = generate_oof_raw(pairs, profiles, expander, id_to_name, config, max_iter, cv_folds, "combined")
    combined_model.fit_calibrator(combined_raw, y)
    scorer.combined_model = combined_model
    combined_coef = combined_model.coef_matrix()
    scorer.combined_feature_weights = {
        feature: [float(value) for value in combined_coef[:, index]]
        for index, feature in enumerate(builder.feature_names)
    }


    return scorer, X_pheno, X_combined


def write_feature_weights(output_dir: str, scorer: SeverityScorer) -> None:
    rows = []
    for feature in scorer.builder.feature_names:
        weights = scorer.combined_feature_weights.get(feature, [0.0, 0.0, 0.0])
        rows.append({
            "feature": feature,
            "weight_severity_gt_0": weights[0],
            "weight_severity_gt_1": weights[1],
            "weight_severity_gt_2": weights[2],
        })
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "combined_feature_weights.csv"), index=False)


def _visualization_dependencies() -> Tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Visualisation requires matplotlib. Install it with `pip install matplotlib`."
        ) from exc
    try:
        import shap
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The SHAP beeswarm requires shap. Install it with `pip install shap`."
        ) from exc
    return plt, shap


def _stratified_sample_indices(y: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    if len(y) <= maximum:
        return np.arange(len(y), dtype=int)
    rng = np.random.default_rng(seed)
    chosen: List[int] = []
    classes = sorted(np.unique(y).tolist())
    per_class = max(1, maximum // max(len(classes), 1))
    for value in classes:
        candidates = np.flatnonzero(y == value)
        take = min(len(candidates), per_class)
        if take:
            chosen.extend(rng.choice(candidates, size=take, replace=False).tolist())
    remaining = maximum - len(chosen)
    if remaining > 0:
        available = np.setdiff1d(np.arange(len(y)), np.asarray(chosen, dtype=int), assume_unique=False)
        if len(available):
            chosen.extend(rng.choice(available, size=min(remaining, len(available)), replace=False).tolist())
    return np.asarray(sorted(chosen[:maximum]), dtype=int)


def _feature_channel(feature: str) -> str:
    if feature.startswith("phenotype:"):
        return "phenotype"
    if feature.startswith("explicit:"):
        return "explicit"
    if feature.startswith("organ_interaction:"):
        return "organ_interaction"
    if feature.startswith("organ:"):
        return "organ"
    if feature.startswith("low_burden:"):
        return "low_burden"
    if feature.startswith("interaction:"):
        return "phenotype_interaction"
    if feature.startswith("absent:"):
        return "absent"
    return "other"


def _pretty_feature_name(feature: str, scorer: SeverityScorer, max_length: int = 72) -> str:
    def term_label(term: str) -> str:
        name = scorer.vectorizer.id_to_name.get(term, scorer.expander.names.get(term, ""))
        return f"{name} [{term}]" if name else term

    label = feature
    if feature.startswith("explicit_level:"):
        label = feature.replace("explicit_level:", "explicit severity level: ")
        return label if len(label) <= max_length else label[: max_length - 1] + "…"
    for prefix in ("phenotype:", "explicit:", "absent:"):
        if feature.startswith(prefix):
            label = f"{prefix[:-1]}: {term_label(feature[len(prefix):])}"
            break
    else:
        if feature.startswith("interaction:"):
            raw = feature[len("interaction:"):]
            left, separator, right = raw.partition("&")
            label = (
                f"interaction: {term_label(left)} × {term_label(right)}"
                if separator else feature
            )
        elif feature.startswith("organ_interaction:"):
            label = feature.replace("organ_interaction:", "organ interaction: ").replace("_", " ")
        elif feature.startswith("organ:"):
            label = feature.replace("organ:", "organ: ").replace("_", " ")
        elif feature.startswith("low_burden:"):
            label = feature.replace("low_burden:", "low-burden evidence: ").replace("_", " ")
    return label if len(label) <= max_length else label[: max_length - 1] + "…"


def compute_ordinal_shap_values(
    scorer: SeverityScorer,
    X_background: sparse.csr_matrix,
    X_explain: sparse.csr_matrix,
    y_explain: np.ndarray,
    max_samples: int = 400,
    background_size: int = 200,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    if scorer.combined_model is None:
        raise ValueError("Combined model is unavailable.")
    _, shap = _visualization_dependencies()
    n_features = X_explain.shape[1]
    if n_features == 0:
        raise ValueError("No features are available for SHAP explanation.")

    # Keep the dense SHAP result within a practical memory envelope.
    max_elements = 20_000_000
    memory_limited_samples = max(20, max_elements // max(n_features, 1))
    effective_max = max(1, min(int(max_samples), int(memory_limited_samples), X_explain.shape[0]))
    explain_indices = _stratified_sample_indices(y_explain, effective_max, seed)
    X_selected = X_explain[explain_indices].tocsr()

    rng = np.random.default_rng(seed + 17)
    if X_background.shape[0] > background_size:
        background_indices = rng.choice(
            X_background.shape[0], size=int(background_size), replace=False
        )
        background = X_background[np.sort(background_indices)].tocsr()
    else:
        background = X_background.tocsr()

    aggregated = np.zeros((X_selected.shape[0], n_features), dtype=np.float32)
    base_value = 0.0
    explained_thresholds = 0
    for threshold_model in scorer.combined_model.models:
        if isinstance(threshold_model, _ConstantBinary):
            continue
        explainer = shap.LinearExplainer(threshold_model, background)
        explanation = explainer(X_selected)
        values = np.asarray(explanation.values, dtype=np.float32)
        if values.ndim == 3:
            values = values[..., -1]
        aggregated += values
        base_values = np.asarray(explanation.base_values, dtype=float)
        base_value += float(np.mean(base_values))
        explained_thresholds += 1

    if explained_thresholds == 0:
        raise ValueError("All ordinal thresholds are constant; SHAP values are undefined.")
    return {
        "values": aggregated,
        "X": X_selected,
        "indices": explain_indices,
        "y": np.asarray(y_explain, dtype=int)[explain_indices],
        "base_value": base_value,
        "explained_thresholds": explained_thresholds,
    }


def _barh_top(
    ax: Any,
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    xlabel: str,
    maximum: int = 15,
) -> None:
    values_array = np.asarray(values, dtype=float)
    if not len(values_array) or not np.isfinite(values_array).any() or np.nanmax(np.abs(values_array)) == 0:
        ax.text(0.5, 0.5, "No supported features", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_axis_off()
        return
    order = np.argsort(np.abs(values_array))[-maximum:]
    ordered_values = values_array[order]
    ordered_labels = [labels[int(index)] for index in order]
    ax.barh(np.arange(len(order)), ordered_values)
    ax.set_yticks(np.arange(len(order)), ordered_labels)
    ax.axvline(0.0, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)


def write_model_visualization_dashboard(
    output_dir: str,
    scorer: SeverityScorer,
    X_background: sparse.csr_matrix,
    X_test_combined: sparse.csr_matrix,
    X_test_pheno: sparse.csr_matrix,
    y_test: np.ndarray,
    max_shap_samples: int = 400,
    shap_background_size: int = 200,
    max_display: int = 20,
    dpi: int = 180,
) -> Dict[str, Any]:
    """Write one multi-panel held-out evaluation dashboard as PNG and PDF."""
    plt, shap = _visualization_dependencies()
    os.makedirs(output_dir, exist_ok=True)
    y_test = np.asarray(y_test, dtype=int)
    combined_probabilities = scorer.combined_model.class_proba(X_test_combined, True)
    cumulative = scorer.combined_model.cumulative_proba(X_test_combined, True)
    combined_expected = scorer.combined_model.expected_severity(X_test_combined, True)
    combined_predicted = np.clip(np.rint(combined_expected), 0, NUM_CLASSES - 1).astype(int)
    phenotype_expected = (
        scorer.pheno_model.expected_severity(X_test_pheno, True)
        if scorer.pheno_model is not None and X_test_pheno.shape[1] else np.full(len(y_test), np.nan)
    )

    shap_result = compute_ordinal_shap_values(
        scorer,
        X_background,
        X_test_combined,
        y_test,
        max_samples=max_shap_samples,
        background_size=shap_background_size,
    )
    shap_values = shap_result["values"]
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    mean_shap = np.mean(shap_values, axis=0)
    feature_names = scorer.builder.feature_names
    pretty_names = [_pretty_feature_name(feature, scorer) for feature in feature_names]

    importance = pd.DataFrame({
        "feature": feature_names,
        "display_name": pretty_names,
        "channel": [_feature_channel(feature) for feature in feature_names],
        "mean_abs_ordinal_shap": mean_abs_shap,
        "mean_ordinal_shap": mean_shap,
    }).sort_values("mean_abs_ordinal_shap", ascending=False)
    importance.to_csv(os.path.join(output_dir, "held_out_ordinal_shap_importance.csv"), index=False)

    figure = plt.figure(figsize=(27, 31), constrained_layout=True)
    grid = figure.add_gridspec(4, 3, height_ratios=[1.35, 1.0, 1.0, 1.0])
    axes = {
        "beeswarm": figure.add_subplot(grid[0, 0:2]),
        "global": figure.add_subplot(grid[0, 2]),
        "confusion": figure.add_subplot(grid[1, 0]),
        "calibration": figure.add_subplot(grid[1, 1]),
        "roc": figure.add_subplot(grid[1, 2]),
        "observed": figure.add_subplot(grid[2, 0]),
        "comparison": figure.add_subplot(grid[2, 1]),
        "residual": figure.add_subplot(grid[2, 2]),
        "channels": figure.add_subplot(grid[3, 0]),
        "organ": figure.add_subplot(grid[3, 1]),
        "interactions": figure.add_subplot(grid[3, 2]),
    }

    top_indices = np.argsort(mean_abs_shap)[-max(1, min(max_display, len(mean_abs_shap))):]
    top_indices = top_indices[np.argsort(mean_abs_shap[top_indices])]
    beeswarm_explanation = shap.Explanation(
        values=shap_values[:, top_indices],
        data=shap_result["X"][:, top_indices].toarray(),
        feature_names=[pretty_names[int(index)] for index in top_indices],
        base_values=np.full(shap_values.shape[0], shap_result["base_value"], dtype=float),
    )
    shap.plots.beeswarm(
        beeswarm_explanation,
        max_display=len(top_indices),
        ax=axes["beeswarm"],
        show=False,
        plot_size=None,
    )
    axes["beeswarm"].set_title(
        "Held-out SHAP beeswarm — summed raw ordinal-threshold log-odds\n"
        "Positive values support higher severity; calibration/projection are downstream"
    )

    _barh_top(
        axes["global"], pretty_names, mean_abs_shap,
        "Global SHAP importance", "Mean |ordinal SHAP|", maximum=max_display,
    )

    matrix = confusion_matrix(y_test, combined_predicted, labels=np.arange(NUM_CLASSES))
    image = axes["confusion"].imshow(matrix, aspect="auto")
    figure.colorbar(image, ax=axes["confusion"], fraction=0.046, pad=0.04)
    axes["confusion"].set_xticks(np.arange(NUM_CLASSES), CLASS_NAMES, rotation=30, ha="right")
    axes["confusion"].set_yticks(np.arange(NUM_CLASSES), CLASS_NAMES)
    axes["confusion"].set_xlabel("Predicted class")
    axes["confusion"].set_ylabel("Observed class")
    axes["confusion"].set_title("Held-out confusion matrix")
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            axes["confusion"].text(
                column, row, str(int(matrix[row, column])), ha="center", va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )

    axes["calibration"].plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Ideal")
    boundary_labels = ("Y > Mild", "Y > Moderate", "Y > Severe")
    for boundary, label in enumerate(boundary_labels):
        target = (y_test > boundary).astype(int)
        if np.unique(target).size < 2:
            continue
        observed_fraction, predicted_mean = calibration_curve(
            target, cumulative[:, boundary], n_bins=min(8, max(3, len(y_test) // 10)),
            strategy="quantile",
        )
        axes["calibration"].plot(predicted_mean, observed_fraction, marker="o", label=label)
    axes["calibration"].set_xlim(0, 1)
    axes["calibration"].set_ylim(0, 1)
    axes["calibration"].set_xlabel("Mean predicted probability")
    axes["calibration"].set_ylabel("Observed frequency")
    axes["calibration"].set_title("Ordinal calibration")
    axes["calibration"].legend(fontsize=8)

    axes["roc"].plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    roc_rows = []
    for boundary, label in enumerate(boundary_labels):
        target = (y_test > boundary).astype(int)
        if np.unique(target).size < 2:
            continue
        false_positive, true_positive, _ = roc_curve(target, cumulative[:, boundary])
        area = auc(false_positive, true_positive)
        roc_rows.append({"boundary": label, "auc": float(area)})
        axes["roc"].plot(false_positive, true_positive, label=f"{label} (AUC={area:.3f})")
    axes["roc"].set_xlim(0, 1)
    axes["roc"].set_ylim(0, 1)
    axes["roc"].set_xlabel("False-positive rate")
    axes["roc"].set_ylabel("True-positive rate")
    axes["roc"].set_title("Ordinal threshold ROC")
    axes["roc"].legend(fontsize=8)

    jitter = np.random.default_rng(RANDOM_SEED).normal(0.0, 0.045, size=len(y_test))
    axes["observed"].scatter(y_test + jitter, combined_expected, alpha=0.75)
    axes["observed"].plot([0, 3], [0, 3], linestyle="--", linewidth=1)
    axes["observed"].set_xticks(np.arange(NUM_CLASSES), CLASS_NAMES, rotation=25, ha="right")
    axes["observed"].set_xlim(-0.25, 3.25)
    axes["observed"].set_ylim(-0.1, 3.1)
    axes["observed"].set_ylabel("Expected severity")
    axes["observed"].set_title("Observed versus expected severity")

    valid_pheno = np.isfinite(phenotype_expected)
    if valid_pheno.any():
        scatter = axes["comparison"].scatter(
            phenotype_expected[valid_pheno], combined_expected[valid_pheno],
            c=y_test[valid_pheno], alpha=0.8,
        )
        figure.colorbar(scatter, ax=axes["comparison"], ticks=np.arange(NUM_CLASSES), label="Observed severity")
        axes["comparison"].plot([0, 3], [0, 3], linestyle="--", linewidth=1)
        axes["comparison"].set_xlim(-0.1, 3.1)
        axes["comparison"].set_ylim(-0.1, 3.1)
        axes["comparison"].set_xlabel("Phenotype-derived expected severity")
        axes["comparison"].set_ylabel("Combined expected severity")
        axes["comparison"].set_title("Phenotype-only versus multichannel")
    else:
        axes["comparison"].text(0.5, 0.5, "Phenotype-only model unavailable", ha="center", va="center", transform=axes["comparison"].transAxes)
        axes["comparison"].set_axis_off()

    residuals = combined_expected - y_test
    axes["residual"].hist(residuals, bins=min(15, max(5, len(residuals) // 4)))
    axes["residual"].axvline(0.0, linestyle="--", linewidth=1)
    axes["residual"].set_xlabel("Expected severity − observed severity")
    axes["residual"].set_ylabel("Profiles")
    axes["residual"].set_title(f"Residual distribution (MAE={mean_absolute_error(y_test, combined_expected):.3f})")

    channel_labels = []
    channel_values = []
    for channel, (start, end) in scorer.builder.slices.items():
        if end > start:
            channel_labels.append(channel.replace("_", " "))
            channel_values.append(float(mean_abs_shap[start:end].sum()))
    _barh_top(
        axes["channels"], channel_labels, channel_values,
        "Evidence-channel importance", "Summed mean |ordinal SHAP|", maximum=10,
    )

    organ_start, organ_end = scorer.builder.slices.get("organ", (0, 0))
    organ_labels = pretty_names[organ_start:organ_end]
    organ_values = mean_abs_shap[organ_start:organ_end]
    _barh_top(
        axes["organ"], organ_labels, organ_values,
        "Organ-system burden importance", "Mean |ordinal SHAP|", maximum=12,
    )

    interaction_start, interaction_end = scorer.builder.slices.get("interactions", (0, 0))
    interaction_labels = pretty_names[interaction_start:interaction_end]
    interaction_values = mean_abs_shap[interaction_start:interaction_end]
    _barh_top(
        axes["interactions"], interaction_labels, interaction_values,
        "Phenotype-interaction importance", "Mean |ordinal SHAP|", maximum=12,
    )

    dashboard_metrics = {
        "n_held_out": int(len(y_test)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, combined_predicted)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_test, combined_predicted, weights="quadratic")),
        "matthews_correlation": float(matthews_corrcoef(y_test, combined_predicted)),
        "mae_expected": float(mean_absolute_error(y_test, combined_expected)),
        "log_loss": float(log_loss(y_test, combined_probabilities, labels=np.arange(NUM_CLASSES))),
        "mild_mean_expected": (
            float(np.mean(combined_expected[y_test == 0])) if np.any(y_test == 0) else None
        ),
        "mild_median_expected": (
            float(np.median(combined_expected[y_test == 0])) if np.any(y_test == 0) else None
        ),
        "mild_mean_probability": (
            float(np.mean(combined_probabilities[y_test == 0, 0])) if np.any(y_test == 0) else None
        ),
        "mild_recall": (
            float(np.mean(combined_predicted[y_test == 0] == 0)) if np.any(y_test == 0) else None
        ),
        "shap_profiles": int(shap_values.shape[0]),
        "shap_features": int(shap_values.shape[1]),
        "shap_explained_thresholds": int(shap_result["explained_thresholds"]),
        "roc": roc_rows,
        "shap_definition": "Sum of exact LinearExplainer SHAP values for the three uncalibrated ordinal-threshold logits.",
    }
    figure.suptitle(
        "Phenotype Severity Scorer — Unified Held-out Evaluation Panel\n"
        f"n={len(y_test)} | balanced accuracy={dashboard_metrics['balanced_accuracy']:.3f} | "
        f"QWK={dashboard_metrics['quadratic_weighted_kappa']:.3f} | "
        f"MAE={dashboard_metrics['mae_expected']:.3f} | "
        f"Mild mean={dashboard_metrics['mild_mean_expected']:.3f}",
        fontsize=18,
    )
    png_path = os.path.join(output_dir, "model_visualization_panel.png")
    pdf_path = os.path.join(output_dir, "model_visualization_panel.pdf")
    figure.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    with open(os.path.join(output_dir, "visualization_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(dashboard_metrics, handle, indent=2)
    print(f"[visualization] wrote {png_path} and {pdf_path}")
    return {"png": png_path, "pdf": pdf_path, "metrics": dashboard_metrics}


def evaluate_external_cohort(scorer: SeverityScorer, external_csv: str, output_dir: str) -> Dict[str, float]:
    external_pairs, names, _ = load_pairs(external_csv, None, "exclude")
    external_pairs, _ = prepare_terms(
        external_pairs,
        scorer.expander,
        names,
        severity_term_policy=scorer.config.severity_term_policy,
    )
    observed: List[int] = []
    expected: List[float] = []
    for _, row in external_pairs.iterrows():
        score = scorer.score_profile(row["all_present_terms"], row["absent_terms"])
        if score["expected_severity_0_3"] is not None:
            observed.append(int(row["sev_num"]))
            expected.append(float(score["expected_severity_0_3"]))
    y = np.asarray(observed, dtype=int)
    exp = np.asarray(expected, dtype=float)
    if not len(y):
        raise ValueError("No external profiles could be scored.")
    pred = np.clip(np.rint(exp), 0, 3).astype(int)
    metrics = {
        "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
        "mcc": float(matthews_corrcoef(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mae_expected": float(mean_absolute_error(y, exp)),
        "n": int(len(y)),
    }
    with open(os.path.join(output_dir, "external_validation.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def write_low_end_diagnostics(
    output_dir: str,
    y: np.ndarray,
    expected: np.ndarray,
    probabilities: np.ndarray,
    predicted: np.ndarray,
    dpi: int = 180,
    write_figures: bool = True,
) -> Dict[str, Optional[str]]:
    """Write class-wise score summaries and a focused Mild-tail panel."""
    y = np.asarray(y, dtype=int)
    expected = np.asarray(expected, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = np.asarray(predicted, dtype=int)
    rows: List[Dict[str, Any]] = []
    for level, name in enumerate(CLASS_NAMES):
        mask = y == level
        values = expected[mask]
        if not values.size:
            continue
        rows.append({
            "observed_class": name,
            "observed_numeric": level,
            "n": int(values.size),
            "mean_expected_0_3": float(np.mean(values)),
            "median_expected_0_3": float(np.median(values)),
            "p05_expected_0_3": float(np.percentile(values, 5)),
            "p25_expected_0_3": float(np.percentile(values, 25)),
            "p75_expected_0_3": float(np.percentile(values, 75)),
            "p95_expected_0_3": float(np.percentile(values, 95)),
            "mean_probability_mild": float(np.mean(probabilities[mask, 0])),
            "class_recall": float(np.mean(predicted[mask] == level)),
        })
    csv_path = os.path.join(output_dir, "low_end_diagnostics.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    if not write_figures:
        return {"csv": csv_path, "png": None, "pdf": None}

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    grouped_expected = [expected[y == level] for level in range(NUM_CLASSES)]
    axes[0, 0].boxplot(grouped_expected, tick_labels=CLASS_NAMES, showfliers=False)
    axes[0, 0].set_ylabel("Expected severity (0–3)")
    axes[0, 0].set_title("Expected severity by true class")

    grouped_pmild = [probabilities[y == level, 0] for level in range(NUM_CLASSES)]
    axes[0, 1].boxplot(grouped_pmild, tick_labels=CLASS_NAMES, showfliers=False)
    axes[0, 1].set_ylabel("Calibrated P(Mild)")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Mild probability by true class")

    mild_values = expected[y == 0]
    axes[1, 0].hist(mild_values, bins=min(20, max(6, len(mild_values) // 5)))
    axes[1, 0].axvline(np.mean(mild_values), linestyle="--", label=f"mean={np.mean(mild_values):.3f}")
    axes[1, 0].axvline(np.median(mild_values), linestyle=":", label=f"median={np.median(mild_values):.3f}")
    axes[1, 0].set_xlabel("Expected severity for true Mild cases (0–3)")
    axes[1, 0].set_ylabel("Profiles")
    axes[1, 0].set_title("Focused Mild-tail distribution")
    axes[1, 0].legend()

    p_gt_mild = 1.0 - probabilities[:, 0]
    target = (y > 0).astype(int)
    observed_fraction, predicted_mean = calibration_curve(
        target, p_gt_mild, n_bins=min(10, max(4, len(y) // 20)), strategy="quantile"
    )
    axes[1, 1].plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    axes[1, 1].plot(predicted_mean, observed_fraction, marker="o", label="P(Y > Mild)")
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xlabel("Mean predicted probability")
    axes[1, 1].set_ylabel("Observed frequency")
    axes[1, 1].set_title("First-boundary calibration")
    axes[1, 1].legend()

    fig.suptitle("Low-end severity diagnostics — held-out cohort", fontsize=16)
    png_path = os.path.join(output_dir, "low_end_diagnostics_panel.png")
    pdf_path = os.path.join(output_dir, "low_end_diagnostics_panel.pdf")
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"csv": csv_path, "png": png_path, "pdf": pdf_path}


def train(args: argparse.Namespace) -> SeverityScorer:
    os.makedirs(args.output, exist_ok=True)
    pairs, csv_names, curation_audit = load_pairs(args.severity_csv, args.output, args.conflict_policy)
    expander = OntologyExpander().load(None if args.no_expand else args.hpo_obo)
    pairs, hpo_audit = prepare_terms(
        pairs,
        expander,
        csv_names,
        severity_term_policy=args.severity_term_policy,
    )
    id_to_name = dict(csv_names)
    id_to_name.update(expander.names)
    pd.DataFrame([hpo_audit]).to_csv(os.path.join(args.output, "hpo_curation_summary.csv"), index=False)

    development, test = optimised_disease_disjoint_split(
        pairs, args.test_fraction, RANDOM_SEED
    )
    split_membership = pd.concat(
        [
            development[["mondo_id", "associated_genes", "n_associated_genes", "sev_num"]]
            .assign(partition="development"),
            test[["mondo_id", "associated_genes", "n_associated_genes", "sev_num"]]
            .assign(partition="test"),
        ],
        ignore_index=True,
    )
    split_membership["associated_genes"] = split_membership["associated_genes"].apply(
        lambda values: "|".join(values)
    )
    split_membership.to_csv(
        os.path.join(args.output, "disease_split_membership.csv"), index=False
    )
    selected_config, selection_results = select_model_config(development, expander, id_to_name, args)
    dev_profiles = expansion_profiles(
        development,
        expander,
        selected_config.max_hops,
        severity_term_policy=selected_config.severity_term_policy,
    )
    test_profiles = expansion_profiles(
        test,
        expander,
        selected_config.max_hops,
        severity_term_policy=selected_config.severity_term_policy,
    )

    evaluation_scorer, X_dev_pheno, X_dev_combined = fit_scorer(
        development, dev_profiles, expander, id_to_name, selected_config, args.max_iter, args.cv_folds
    )
    X_test_combined, test_blocks = evaluation_scorer.builder.transform(test, test_profiles)
    X_test_pheno = test_blocks["phenotype"]
    y_test = test["sev_num"].to_numpy(dtype=int)
    held_out_pheno = evaluate_model(evaluation_scorer.pheno_model, X_test_pheno, y_test, True) if evaluation_scorer.pheno_model else {}
    held_out_combined = evaluate_model(evaluation_scorer.combined_model, X_test_combined, y_test, True)
    print(f'[eval] phenotype-derived unchanged test: {held_out_pheno}')
    print(f'[eval] combined unchanged test: {held_out_combined}')

    test_probabilities = evaluation_scorer.combined_model.class_proba(X_test_combined, True)
    test_expected = evaluation_scorer.combined_model.expected_severity(X_test_combined, True)

    predictions = pd.DataFrame({
        "mondo_id": test["mondo_id"],
        "associated_genes": test["associated_genes"].apply(lambda values: "|".join(values)),
        "n_associated_genes": test["n_associated_genes"],
        "observed": y_test,
        "phenotype_expected": (
            evaluation_scorer.pheno_model.expected_severity(X_test_pheno, True)
            if evaluation_scorer.pheno_model else np.full(len(test), np.nan)
        ),
        "combined_expected": test_expected,
        "predicted_class": np.clip(np.rint(test_expected), 0, NUM_CLASSES - 1).astype(int),
        "p_mild": test_probabilities[:, 0],
        "p_moderate": test_probabilities[:, 1],
        "p_severe": test_probabilities[:, 2],
        "p_profound": test_probabilities[:, 3],
    })
    predictions.to_csv(os.path.join(args.output, "held_out_test_predictions.csv"), index=False)
    low_end_artifacts = write_low_end_diagnostics(
        args.output,
        y_test,
        test_expected,
        test_probabilities,
        predictions["predicted_class"].to_numpy(dtype=int),
        dpi=args.dashboard_dpi,
        write_figures=not args.no_visualizations,
    )

    visualization_artifacts: Dict[str, Any] = {}
    if not args.no_visualizations:
        try:
            visualization_artifacts = write_model_visualization_dashboard(
                args.output,
                evaluation_scorer,
                X_dev_combined,
                X_test_combined,
                X_test_pheno,
                y_test,
                max_shap_samples=args.dashboard_max_samples,
                shap_background_size=args.dashboard_background_size,
                max_display=args.dashboard_max_display,
                dpi=args.dashboard_dpi,
            )
        except Exception as exc:
            warnings.warn(f"Visualisation dashboard was not generated: {exc}")

    all_profiles = expansion_profiles(
        pairs,
        expander,
        selected_config.max_hops,
        severity_term_policy=selected_config.severity_term_policy,
    )
    scorer, _, _ = fit_scorer(
        pairs, all_profiles, expander, id_to_name, selected_config, args.max_iter, args.cv_folds
    )
    scorer.min_supported_direct_terms = int(args.min_supported_direct_terms)
    scorer.min_recognised_expanded_terms = int(args.min_recognised_expanded_terms)
    write_feature_weights(args.output, scorer)

    scorer.metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "pipeline_version": "3.3-disease-level-ontology-severity-terms-configurable",
        "random_seed": RANDOM_SEED,
        "training_unit": "MONDO disease",
        "split_group": "mondo_id",
        "split_audit": {
            "development_diseases": int(development["mondo_id"].nunique()),
            "test_diseases": int(test["mondo_id"].nunique()),
            "overlapping_diseases": int(
                len(set(development["mondo_id"]) & set(test["mondo_id"]))
            ),
        },
        "n_diseases": int(len(pairs)),
        "n_pairs": int(len(pairs)),  # compatibility alias
        "n_gene_disease_associations": int(pairs["n_associated_genes"].sum()),
        "n_genes": int(
            len({gene for genes in pairs["associated_genes"] for gene in genes})
        ),
        "selected_config": asdict(selected_config),
        "severity_term_handling": {
            "policy": selected_config.severity_term_policy,
            "description": (
                "Severity-naming HPO terms are retained in the ordinary phenotype channel."
            ),
            "separate_explicit_channel_enabled": bool(selected_config.use_explicit_channel),
        },
        "organ_mapping": scorer.builder.organ.mapping_audit(),
        "explicit_channel": {
            "enabled": bool(selected_config.use_explicit_channel),
            "encoding": "maximum_observed_ordinal_level",
            "minimum_absolute_logit_contribution": selected_config.explicit_min_abs_logit,
            "training_level_counts": getattr(scorer.builder.explicit, "training_counts", {}),
            "unmapped_training_terms": getattr(scorer.builder.explicit, "unmapped_training_terms", {}),
        },
        "feature_counts": {
            "phenotype": len(scorer.builder.phenotype.vocab),
            "explicit_severity": len(scorer.builder.explicit.vocab),
            "organ_and_system_interactions": len(scorer.builder.organ.feature_names),
            "observed_low_burden": len(scorer.builder.low_burden.feature_names),
            "phenotype_interactions": len(scorer.builder.interactions.pairs),
            "absent": len(scorer.builder.absent.vocab),
            "combined_total": len(scorer.builder.feature_names),
        },
        "learned_dimensions": sorted(scorer.dimension_models),
        "curation": curation_audit,
        "hpo_curation": hpo_audit,
        "held_out_phenotype": held_out_pheno,
        "held_out_combined": held_out_combined,
        "visualization_artifacts": {
            "png": os.path.abspath(visualization_artifacts["png"]) if visualization_artifacts.get("png") else None,
            "pdf": os.path.abspath(visualization_artifacts["pdf"]) if visualization_artifacts.get("pdf") else None,
            "low_end_csv": os.path.abspath(low_end_artifacts["csv"]),
            "low_end_png": (
                os.path.abspath(low_end_artifacts["png"]) if low_end_artifacts.get("png") else None
            ),
            "low_end_pdf": (
                os.path.abspath(low_end_artifacts["pdf"]) if low_end_artifacts.get("pdf") else None
            ),
            "shap_definition": (
                visualization_artifacts.get("metrics", {}).get("shap_definition")
                if visualization_artifacts else None
            ),
        },
        "model_selection_complete_configs": int(selection_results["complete"].sum()),
        "inputs": {
            "severity_csv": {"path": os.path.abspath(args.severity_csv), "sha256": _sha256(args.severity_csv)},
            "hpo_obo": {"path": os.path.abspath(args.hpo_obo), "sha256": _sha256(args.hpo_obo)} if args.hpo_obo else None,
        },
        "caveats": [
            "The model learns associations from disease-level phenotype profiles unless patient-level rows are supplied.",
            "Severity-naming HPO terms are ordinary phenotype predictors under the default policy and can therefore strongly reflect annotation wording; this requires external validation.",
            "New organ-system features use HPO branch ancestry; boundary-aware text fallback is used only for unresolved branches.",
            "Absent terms affect scores only when enough negative annotations survive training-frequency filtering.",
            "Organ-system branch roots and any unresolved fallback mappings should be validated for the local HPO release.",
            "Pairwise interactions are selected by training-fold co-occurrence and are associations, not causal effects.",
            "Low-burden features describe observed evidence only; undocumented findings are never treated as absent.",
            "The Mild boundary is tuned and calibrated using severity labels only, never prenatal outcome.",
            "Optional dimension models are fitted only when corresponding target columns are present.",
            "Dashboard SHAP values sum exact linear SHAP values across the three raw ordinal logits; they do not include calibration or monotonic projection.",
            "Independent patient-level external validation is required before clinical use.",
        ],
    }
    with open(os.path.join(args.output, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(scorer.metadata, handle, indent=2, default=str)
    with open(os.path.join(args.output, "model.pkl"), "wb") as handle:
        pickle.dump(scorer, handle)
    if args.external_severity_csv:
        evaluate_external_cohort(scorer, args.external_severity_csv, args.output)
    print(f"[done] artifacts written to {os.path.abspath(args.output)}")
    return scorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--severity-csv", default="final_output_with_severity.csv")
    parser.add_argument("--hpo-obo", default="hp.obo")
    parser.add_argument("--output", default="results/scorer")
    parser.add_argument("--external-severity-csv", default=None)
    parser.add_argument("--conflict-policy", choices=["exclude", "mode"], default="exclude")
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--max-iter", type=int, default=5000)

    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--mild-C", type=float, default=0.25)
    parser.add_argument(
        "--mild-class-weight", choices=["none", "balanced"], default="none",
        help="Class weighting for P(Y > Mild). 'none' often improves low-end calibration.",
    )
    parser.add_argument(
        "--other-class-weight", choices=["none", "balanced"], default="balanced",
        help="Class weighting for the two higher ordinal boundaries.",
    )
    parser.add_argument(
        "--mild-calibration", choices=["platt", "isotonic"], default="isotonic",
        help="Out-of-fold calibration method for P(Y > Mild).",
    )
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--max-freq", type=float, default=0.95)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--ancestor-decay", type=float, default=0.75)
    parser.add_argument("--ic-weighting", action="store_true")
    parser.add_argument("--no-expand", action="store_true")
    parser.add_argument("--no-tune", action="store_true")

    parser.add_argument(
        "--severity-term-policy",
        choices=["phenotype"],
        default="phenotype",
        help=(
            "How to use HPO terms whose identifiers or names encode severity. "
            "The default 'phenotype' retains them as ordinary phenotype features "
            "and disables the separate explicit channel to prevent duplicate counting. "
        ),
    )
    parser.add_argument("--absent-channel-scale", type=float, default=0.50)
    parser.add_argument("--organ-channel-scale", type=float, default=0.50)
    parser.add_argument("--interaction-channel-scale", type=float, default=0.50)
    parser.add_argument("--low-burden-channel-scale", type=float, default=0.50)
    parser.add_argument("--interaction-min-count", type=int, default=10)
    parser.add_argument("--max-interactions", type=int, default=250)
    parser.add_argument("--disable-organ-burden", action="store_true")
    parser.add_argument("--disable-interactions", action="store_true")
    parser.add_argument("--disable-absent-channel", action="store_true")
    parser.add_argument("--disable-low-burden-channel", action="store_true")

    parser.add_argument("--C-grid", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    parser.add_argument("--mild-C-grid", nargs="+", type=float, default=[0.05, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument(
        "--mild-class-weight-grid", nargs="+", choices=["none", "balanced"],
        default=["none", "balanced"],
    )
    parser.add_argument(
        "--low-end-penalty-weight", type=float, default=0.15,
        help="Penalty applied to cross-validated Mild expected severity during model selection.",
    )
    parser.add_argument("--min-count-grid", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--max-freq-grid", nargs="+", type=float, default=[0.90, 0.95])
    parser.add_argument("--max-hops-grid", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--ancestor-decay-grid", nargs="+", type=float, default=[0.5, 0.75, 1.0])
    parser.add_argument("--compare-ic", action="store_true")
    parser.add_argument("--max-configs", type=int, default=24)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)

    parser.add_argument("--min-supported-direct-terms", type=int, default=1)
    parser.add_argument("--min-recognised-expanded-terms", type=int, default=1)
    parser.add_argument("--no-visualizations", action="store_true")
    parser.add_argument("--dashboard-max-samples", type=int, default=400)
    parser.add_argument("--dashboard-background-size", type=int, default=200)
    parser.add_argument("--dashboard-max-display", type=int, default=20)
    parser.add_argument("--dashboard-dpi", type=int, default=180)
    parser.add_argument("--load", default=None)
    parser.add_argument("--score", nargs="*", default=None)
    parser.add_argument("--absent-score", nargs="*", default=None)
    parser.add_argument("--allow-low-information", action="store_true")
    parser.set_defaults(**{key: ModelConfig.__dataclass_fields__[key].default
                           for key in ('explicit_channel_scale', 'explicit_min_abs_logit')})
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.load:
        with open(args.load, "rb") as handle:
            scorer: SeverityScorer = pickle.load(handle)
        if args.score:
            score = scorer.score_profile(
                args.score,
                absent_hpo_terms=args.absent_score,
                allow_low_information=args.allow_low_information,
            )
            print(json.dumps(score, indent=2, default=str))
        else:
            print("Loaded model. Pass --score HP:.... HP:.... to score a profile.")
        return
    scorer = train(args)
    if args.score:
        print("\nExample score:")
        score = scorer.score_profile(
            args.score,
            absent_hpo_terms=args.absent_score,
            allow_low_information=args.allow_low_information,
        )
        print(json.dumps(score, indent=2, default=str))


if __name__ == "__main__":
    main()
