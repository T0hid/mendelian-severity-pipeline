"""HPO and MONDO ontologies (Disease-context and phenotype features).

HPO terms are resolved to current identifiers and encoded as one binary column per term.
MONDO features are the direct subclasses of "human disease" (MONDO:0700096) that a disease
belongs to.
"""

import hashlib
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .runtime import logger


def safe_hpo_colname(hpo_id_or_name: str) -> str:
    s = str(hpo_id_or_name).strip()
    if re.match("^HP:\\d{7}$", s):
        return f"hpo_{s.replace(':', '_')}"
    digest = hashlib.md5(s.encode("utf-8")).hexdigest()[:10]
    sanitized = re.sub("[^a-zA-Z0-9]", "_", s)[:30].strip("_")
    return f"hpo_{sanitized}_{digest}" if sanitized else f"hpo_{digest}"


def safe_feature_name(prefix: str, raw_value: str, max_len: int = 40) -> str:
    s = re.sub("\\W+", "_", str(raw_value)).strip("_")
    if len(s) > max_len:
        digest = hashlib.md5(str(raw_value).encode("utf-8")).hexdigest()[:6]
        s = f"{s[:max_len]}_{digest}"
    return f"{prefix}_{s}"


def parse_hpo_obo(obo_path: str) -> Tuple[Dict[str, Dict], Dict[str, str], Dict[str, str]]:
    logger.log(f"  Parsing HPO ontology from {obo_path}...")
    if not os.path.exists(obo_path):
        raise FileNotFoundError(f"The pinned HPO ontology was not found: {obo_path}")
    records: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    def commit() -> None:
        nonlocal current
        if current and current.get("id"):
            records.append(current)
        current = None

    with open(obo_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line == "[Term]":
                commit()
                current = {
                    "id": None,
                    "name": "",
                    "parents": [],
                    "alt_ids": [],
                    "obsolete": False,
                    "replaced_by": None,
                }
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
                    current["alt_ids"].append(line[8:].strip())
                elif line.startswith("is_obsolete: true"):
                    current["obsolete"] = True
                elif line.startswith("replaced_by: HP:"):
                    current["replaced_by"] = line[13:].strip()
        commit()
    terms: Dict[str, Dict] = {}
    alt_to_id: Dict[str, str] = {}
    name_candidates: Dict[str, set] = defaultdict(set)
    for rec in records:
        tid = rec["id"]
        replacement = rec.get("replaced_by")
        if rec.get("obsolete"):
            if replacement:
                alt_to_id[tid] = replacement
                for alt in rec.get("alt_ids", []):
                    alt_to_id[alt] = replacement
            continue
        terms[tid] = {"name": rec.get("name", ""), "parents": list(rec.get("parents", []))}
        for alt in rec.get("alt_ids", []):
            alt_to_id[alt] = tid
        if rec.get("name"):
            name_candidates[str(rec["name"]).strip().lower()].add(tid)
    name_to_id = {name: next(iter(ids)) for name, ids in name_candidates.items() if len(ids) == 1}
    logger.log(f"  Parsed {len(terms)} current HPO terms, {len(alt_to_id)} alternate/replaced IDs")
    return (terms, alt_to_id, name_to_id)


def canonicalize_hpo_token(
    token: Any, alt_to_id: Dict[str, str], name_to_id: Dict[str, str]
) -> Optional[str]:
    """Return the current HPO identifier for an identifier, alternate identifier or exact term name."""
    value = str(token).strip()
    if not value:
        return None
    match = re.search("HP:\\d{7}", value)
    if match:
        hpo_id = match.group(0)
        return alt_to_id.get(hpo_id, hpo_id)
    return name_to_id.get(value.lower())


def resolve_hpo_terms(
    raw_terms: Sequence[Any], alt_to_id: Dict[str, str], name_to_id: Dict[str, str]
) -> Tuple[set, set]:
    """Split recorded terms into resolved HPO identifiers and unresolved labels."""
    direct_ids: set = set()
    unresolved: set = set()
    for raw in raw_terms or []:
        canonical = canonicalize_hpo_token(raw, alt_to_id, name_to_id)
        if canonical:
            direct_ids.add(canonical)
        else:
            unresolved.add(str(raw).strip())
    return direct_ids, unresolved


def parse_mondo_obo(obo_path: str) -> Dict[str, Dict]:
    logger.log(f"  Parsing MONDO ontology from {obo_path}...")
    if not os.path.exists(obo_path):
        raise FileNotFoundError(f"The pinned MONDO ontology was not found: {obo_path}")
    terms: Dict[str, Dict] = {}
    current_id = None
    current_name = None
    current_parents: List[str] = []
    current_is_obsolete = False
    in_term = False
    with open(obo_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                if current_id and (not current_is_obsolete):
                    terms[current_id] = {"name": current_name or "", "parents": current_parents}
                current_id = None
                current_name = None
                current_parents = []
                current_is_obsolete = False
                in_term = True
            elif line == "" or line.startswith("["):
                if current_id and (not current_is_obsolete) and in_term:
                    terms[current_id] = {"name": current_name or "", "parents": current_parents}
                current_id = None
                current_name = None
                current_parents = []
                current_is_obsolete = False
                in_term = line.startswith("[Term]")
            elif in_term:
                if line.startswith("id: "):
                    current_id = line[4:].strip()
                elif line.startswith("name: "):
                    current_name = line[6:].strip()
                elif line.startswith("is_a: "):
                    parent_id = line[6:].split("!")[0].split("{")[0].strip()
                    current_parents.append(parent_id)
                elif line.startswith("is_obsolete: true"):
                    current_is_obsolete = True
        if current_id and (not current_is_obsolete):
            terms[current_id] = {"name": current_name or "", "parents": current_parents}
    logger.log(f"  Parsed {len(terms)} MONDO terms")
    depth_cache: Dict[str, int] = {}

    def get_depth(tid: str, visiting: Optional[set] = None) -> int:
        if tid in depth_cache:
            return depth_cache[tid]
        if visiting is None:
            visiting = set()
        if tid in visiting:
            depth_cache[tid] = 0
            return 0
        if tid not in terms or not terms[tid]["parents"]:
            depth_cache[tid] = 0
            return 0
        visiting = visiting | {tid}
        valid_parents = [p for p in terms[tid]["parents"] if p in terms]
        if not valid_parents:
            depth_cache[tid] = 0
            return 0
        d = 1 + max((get_depth(p, visiting) for p in valid_parents))
        depth_cache[tid] = d
        return d

    for tid in terms:
        terms[tid]["depth"] = get_depth(tid)
    class_root = Config.MONDO_CLASS_ROOT
    if class_root not in terms:
        raise ValueError(f"MONDO class root {class_root} not found (or obsolete) in {obo_path}.")
    disease_class_roots: Dict[str, str] = {}
    used_labels: set = set()
    for tid in sorted(terms):
        if not tid.startswith("MONDO:") or class_root not in terms[tid]["parents"]:
            continue
        label = re.sub("[^a-z0-9]+", "_", terms[tid]["name"].lower()).strip("_")[:40]
        label = label or tid.replace(":", "_")
        if label in used_labels:
            label = f"{label[:31]}_{tid.split(':')[1]}"
        used_labels.add(label)
        disease_class_roots[tid] = label
    if not disease_class_roots:
        raise ValueError(f"No direct subclasses of {class_root} found in {obo_path}.")
    logger.log(
        f"  MONDO disease classes: {len(disease_class_roots)} direct subclasses of {class_root} ({terms[class_root]['name']})"
    )

    def get_ancestors(tid: str, visited: Optional[set] = None) -> set:
        if visited is None:
            visited = set()
        if tid in visited or tid not in terms:
            return set()
        visited.add(tid)
        ancestors = {tid}
        for p in terms[tid]["parents"]:
            ancestors.update(get_ancestors(p, visited))
        return ancestors

    for tid in terms:
        ancestors = get_ancestors(tid)
        classes = []
        for class_id, class_name in disease_class_roots.items():
            if class_id in ancestors and class_id != tid:
                classes.append(class_name)
        terms[tid]["disease_classes"] = classes
    return terms


def build_mondo_features(disease_df: pd.DataFrame, mondo_terms: Dict) -> Tuple[pd.DataFrame, List[str]]:
    logger.log("  Building MONDO disease class features...")
    if not mondo_terms:
        return (disease_df, [])
    depths = []
    class_lists = []
    known_flags = []
    for _, row in disease_df.iterrows():
        mid = row["mondo_id"]
        mid_clean = str(mid).replace("_", ":").strip() if pd.notna(mid) else ""
        known = mid_clean in mondo_terms
        known_flags.append(known)
        if known:
            depths.append(float(mondo_terms[mid_clean]["depth"]))
            class_lists.append(list(mondo_terms[mid_clean].get("disease_classes", [])))
        else:
            depths.append(np.nan)
            class_lists.append([])
    all_classes = sorted({class_name for classes in class_lists for class_name in classes})
    mondo_data = {
        "has_mondo_record": np.asarray(known_flags, dtype=np.uint8),
        "feat_mondo_depth": np.asarray(depths, dtype=float),
    }
    for class_name in all_classes:
        column = f"feat_mondo_{class_name[:40]}"
        mondo_data[column] = np.asarray(
            [
                (1.0 if class_name in classes else 0.0) if known else np.nan
                for classes, known in zip(class_lists, known_flags)
            ],
            dtype=float,
        )
    mondo_data["feat_mondo_n_classes"] = np.asarray(
        [float(len(classes)) if known else np.nan for classes, known in zip(class_lists, known_flags)],
        dtype=float,
    )
    mondo_frame = pd.DataFrame(mondo_data, index=disease_df.index)
    disease_df = pd.concat([disease_df, mondo_frame], axis=1).copy()
    new_cols = list(mondo_frame.columns)
    logger.log(f"  Created {len(new_cols)} MONDO features")
    return (disease_df, new_cols)
