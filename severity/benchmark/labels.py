"""Readable feature labels for tables and figures (for example "hpo_HP_0000252" -> "Microcephaly").

``write_feature_label_dictionaries`` writes ``feature_dictionary.csv``, which the figure code reads.
"""

import os
import re
from typing import Dict, List

import pandas as pd

from .config import Config
from .runtime import DataContainer, logger

FEATURE_DISPLAY_LABELS: Dict[str, str] = {
    "shet_log10": "LoF selection, log10 s_het",
    "shet_post_mean": "LoF selection (s_het)",
    "shet_percentile": "s_het percentile rank",
    "shet_ci_width_log10": "s_het estimate uncertainty (95% CI width)",
    "shet_post_lower95": "s_het lower 95% bound",
    "shet_post_upper95": "s_het upper 95% bound",
    "shet_prior_mean": "s_het prior mean",
    "hpa_tissue_specificity_ord": "Tissue-restricted expression (HPA)",
    "hpa_celltype_specificity_ord": "Cell-type-restricted expression (HPA)",
    "hpa_celltype_group_specificity_ord": "Cell-lineage-restricted expression (HPA)",
    "hpa_is_low_specificity": "Broadly expressed across tissues (HPA)",
    "hpa_is_tissue_enriched": "Enriched in a single tissue (HPA)",
    "hpa_is_celltype_enriched": "Enriched in a single cell type (HPA)",
    "hpa_not_detected": "Not detected in any tissue (HPA)",
    "hpa_brain_regional_specific": "Brain-region-specific expression (HPA)",
    "hpa_tissue_specificity_score": "Tissue enrichment score (HPA)",
    "hpa_celltype_specificity_score": "Cell-type enrichment score (HPA)",
    "brainspan_prenatal_mean_rpkm": "Prenatal brain expression (BrainSpan)",
    "brainspan_postnatal_mean_rpkm": "Postnatal brain expression (BrainSpan)",
    "brainspan_prenatal_fraction": "Prenatal share of brain expression (BrainSpan)",
    "brainspan_prenatal_log2fc": "Prenatal vs postnatal brain expression, log2",
    "brainspan_peak_dpc": "Age at peak brain expression (BrainSpan)",
    "brainspan_peak_is_prenatal": "Brain expression peaks before birth",
    "brainspan_max_rpkm_log": "Peak brain expression level (BrainSpan)",
    "brainspan_expressed_in_brain": "Expressed in brain at any stage (BrainSpan)",
    "domino_score": "Probability of dominant inheritance (DOMINO)",
    "domino_lda": "DOMINO discriminant score",
    "domino_likely_dominant": "Predicted dominant gene (DOMINO)",
    "domino_likely_recessive": "Predicted recessive gene (DOMINO)",
    "gdi_phred": "Gene damage index, phred-scaled (GDI)",
    "gdi_raw": "Gene damage index (GDI)",
    "gdi_log": "Gene damage index, log10 (GDI)",
    "gdi_mendelian_damage_high": "High background damage in Mendelian genes (GDI)",
    "gnomad_pLI": "Loss-of-function intolerance (pLI)",
    "pLI": "Loss-of-function intolerance (pLI)",
    "gnomad_regional_pLI": "Regional loss-of-function intolerance (pLI)",
    "feat_pli_squared": "pLI, squared",
    "gnomad_oe_lof": "Observed/expected LoF variants",
    "oe_lof": "Observed/expected LoF variants",
    "gnomad_oe_lof_upper": "LoF constraint, upper bound (LOEUF)",
    "gnomad_oe_mis": "Observed/expected missense variants",
    "oe_mis": "Observed/expected missense variants",
    "gnomad_regional_oe_lof": "Regional observed/expected LoF",
    "gnomad_regional_oe_mis": "Regional observed/expected missense",
    "gnomad_mis_z": "Missense constraint (Z score)",
    "gnomad_lof_z": "LoF constraint (Z score)",
    "gnomad_syn_z": "Synonymous constraint (Z score)",
    "gnomad_obs_lof": "Observed LoF variants",
    "gnomad_exp_lof": "Expected LoF variants",
    "gnomad_obs_mis": "Observed missense variants",
    "gnomad_exp_mis": "Expected missense variants",
    "gnomad_cds_length": "Coding sequence length",
    "gnomad_gene_length": "Gene length",
    "feat_missz_minus_synz": "Missense minus synonymous constraint",
    "feat_missz_x_pli": "Missense constraint x pLI",
    "feat_triple_constraint": "Constrained by LoF, missense and dosage",
    "pHaplo": "Haploinsufficiency probability (pHaplo)",
    "pTriplo": "Triplosensitivity probability (pTriplo)",
    "collins_pHI": "Haploinsufficiency probability (Collins)",
    "collins_pTS": "Triplosensitivity probability (Collins)",
    "clingen_hi_score": "ClinGen haploinsufficiency rating",
    "clingen_hi_score_x": "ClinGen haploinsufficiency rating",
    "clingen_hi_score_y": "ClinGen haploinsufficiency rating",
    "clingen_ts_score": "ClinGen triplosensitivity rating",
    "hi_sensitive": "ClinGen haploinsufficiency flag set",
    "ts_sensitive": "ClinGen triplosensitivity flag set",
    "is_haploinsufficient": "Haploinsufficient gene",
    "is_triplosensitive": "Triplosensitive gene",
    "dosage_sensitive_any": "ClinGen HI or TS flag set",
    "ordered_dosage_sensitive": "Dosage sensitivity, graded",
    "hi_no_backup": "Haploinsufficient with no paralogue backup",
    "depmap_mean_score": "Cell-line essentiality (DepMap mean)",
    "depmap_mean_effect": "Mean gene-effect score (DepMap)",
    "depmap_min_effect": "Strongest gene-effect score (DepMap)",
    "depmap_std_effect": "Variability of gene effect across lines (DepMap)",
    "depmap_essential_fraction": "Fraction of cell lines where essential (DepMap)",
    "depmap_n_essential": "Cell lines where essential (DepMap)",
    "depmap_is_pan_essential": "Essential in >90% of cell lines (DepMap)",
    "depmap_is_context_essential": "Essential in 10-90% of cell lines (DepMap)",
    "feat_essential_constrained": "Essential and constrained",
    "feat_central_essential": "Network-central and essential",
    "gtex_tau": "Tissue specificity of expression (GTEx tau)",
    "gtex_max_exp": "Peak tissue expression (GTEx)",
    "feat_is_dominant": "Dominant inheritance",
    "feat_is_recessive": "Recessive inheritance",
    "feat_is_xlinked": "X-linked inheritance",
    "feat_has_mixed_inheritance": "Gene with more than one inheritance mode",
    "is_dominant": "Dominant inheritance",
    "is_recessive": "Recessive inheritance",
    "is_xlinked": "X-linked inheritance",
    "has_mixed_inheritance": "Gene with more than one inheritance mode",
    "n_inheritance_modes": "Number of inheritance modes for the gene",
    "feat_inheritance_not_recorded": "Inheritance not recorded",
    "feat_pli_x_dominant": "pLI in dominant genes",
    "feat_pli_x_recessive": "pLI in recessive genes",
    "feat_dominant_low_constraint": "Dominant gene under weak constraint",
    "feat_recessive_high_constraint": "Recessive gene under strong constraint",
    "paralog_count": "Number of paralogues",
    "n_paralogs": "Number of paralogues",
    "max_paralog_identity_x": "Closest paralogue sequence identity",
    "max_paralog_identity": "Closest paralogue sequence identity",
    "has_close_paralog": "Has a close paralogue",
    "clingen_dosage_curated": "Curated by ClinGen dosage sensitivity",
    "max_paralog_identity_y": "Closest paralogue sequence identity",
    "closest_paralog_identity": "Closest paralogue sequence identity",
    "has_close_paralog_x": "Has a close paralogue",
    "has_close_paralog_y": "Has a close paralogue",
    "is_singleton": "No paralogue in the genome",
    "redundancy_score": "Functional redundancy score",
    "compensation_score": "Paralogue compensation potential",
    "compensation_failure": "No paralogue compensation available",
    "has_functional_backup": "Has a functional paralogue backup",
    "no_backup_flag": "No functional backup",
    "paralog_expression_corr": "Expression correlation with closest paralogue",
    "feat_constrained_no_paralog": "Constrained with no paralogue",
    "rigid_no_backup": "Structurally rigid with no paralogue backup",
    "af_mean_plddt": "AlphaFold confidence, mean pLDDT",
    "af_median_plddt": "AlphaFold confidence, median pLDDT",
    "af_plddt_std": "AlphaFold confidence variability",
    "af_frac_disordered": "Fraction of residues predicted disordered",
    "af_frac_confident": "Fraction of residues confidently folded",
    "af_frac_very_high": "Fraction of residues at very high confidence",
    "af_disorder_ratio": "Disordered-to-ordered residue ratio",
    "af_structural_rigidity": "Predicted structural rigidity",
    "af_protein_length": "Protein length (AlphaFold)",
    "alphafold_plddt": "AlphaFold confidence (pLDDT)",
    "protein_length": "Protein length",
    "mass_kda": "Protein mass (kDa)",
    "domain_count": "Annotated protein domains",
    "n_domains": "Annotated protein domains",
    "transmembrane_segments": "Transmembrane segments",
    "is_membrane_protein": "Membrane protein",
    "has_signal_peptide": "Has a signal peptide",
    "is_nuclear": "Nuclear-localised protein",
    "is_mitochondrial": "Mitochondrial protein",
    "is_secreted": "Secreted protein",
    "ppi_degree": "Protein interaction partners",
    "ppi_betweenness": "Interaction network betweenness",
    "ppi_closeness": "Interaction network closeness",
    "ppi_pagerank": "Interaction network PageRank",
    "string_degree": "STRING interaction partners",
    "feat_log_ppi_degree": "Protein interaction partners, log",
    "feat_betweenness_x_pli": "Network betweenness x pLI",
    "network_degree": "Network degree",
    "network_degree_log": "Network degree, log",
    "network_degree_centrality": "Network degree centrality",
    "network_clustering_coef": "Network clustering coefficient",
    "network_pagerank": "Network PageRank",
    "network_pagerank_log": "Network PageRank, log",
    "network_hub_score": "Network hub score",
    "network_core_number": "Network k-core number",
    "network_avg_neighbor_degree": "Mean degree of network neighbours",
    "network_bottleneck_estimate": "Network bottleneck score",
    "network_is_hub": "Network hub gene",
    "network_is_core": "Network core gene",
    "is_hub": "Network hub gene",
    "feat_hub_constrained": "Network hub under constraint",
    "feat_hub_essential": "Network hub and essential",
    "complex_corum_complex_count": "Protein complexes joined (CORUM)",
    "complex_corum_avg_complex_size": "Mean protein complex size (CORUM)",
    "var_n_pathogenic_clinvar": "Pathogenic ClinVar variants",
    "var_n_lof": "Reported LoF variants",
    "var_n_missense": "Reported missense variants",
    "var_total_variants": "Reported variants, total",
    "var_mean_cadd": "Mean CADD score of reported variants",
    "var_max_cadd": "Maximum CADD score of reported variants",
    "feat_prop_pathogenic": "Proportion of variants that are pathogenic",
    "feat_prop_lof": "Proportion of variants that are LoF",
    "feat_density_pathogenic": "Pathogenic variants per coding base",
    "feat_density_lof": "LoF variants per coding base",
    "feat_density_missense": "Missense variants per coding base",
    "feat_ratio_lof_missense": "LoF-to-missense variant ratio",
    "conservation_score": "Cross-species conservation",
    "gene_age": "Evolutionary age of the gene",
    "is_ancient": "Ancient gene family",
    "feat_conserved_constrained": "Conserved and constrained",
    "has_mouse_ortholog": "Mouse orthologue present",
    "has_zebrafish_ortholog": "Zebrafish orthologue present",
    "has_fly_ortholog": "Fly orthologue present",
    "has_worm_ortholog": "Worm orthologue present",
    "has_yeast_ortholog": "Yeast orthologue present",
    "n_transcripts": "Annotated transcripts",
    "n_transcripts_log": "Annotated transcripts, log",
    "has_multiple_isoforms": "Has multiple isoforms",
    "gene_length": "Gene length",
    "gene_length_log": "Gene length, log",
    "mouse_lethal_pheno_max": "Strongest mouse-lethal phenotype match",
    "mouse_lethal_pheno_mean": "Mean mouse-lethal phenotype match",
    "mouse_lethal_pheno_n_hpos_with_data": "Phenotype terms with mouse-lethal data",
    "mouse_subviable_pheno_mean": "Mean mouse-subviable phenotype match",
    "mouse_subviable_pheno_max": "Strongest mouse-subviable phenotype match",
    "mouse_lethal_vs_pheno_group": "Mouse lethal vs phenotype-group contrast",
    "mouse_viability_score": "Mouse knockout viability",
    "mouse_phenotype_count": "Mouse knockout phenotypes recorded",
    "mouse_lethal": "Mouse knockout is lethal",
    "mouse_subviable": "Mouse knockout is subviable",
    "feat_lethal_x_pli": "Mouse-lethal knockout x pLI",
    "feat_lethal_x_essential": "Mouse-lethal knockout x essentiality",
    "feat_viability_x_pli": "Mouse viability x pLI",
    "feat_has_lethal_x_pli": "Has mouse-lethal record x pLI",
}


_FEATURE_SOURCE_PREFIXES = (
    ("feat_gnn_biological_processes_", "GO process: ", "GO biological processes annotated"),
    ("feat_gnn_molecular_activities_", "GO function: ", "GO molecular functions annotated"),
    ("feat_gnn_expression_sites_", "Expressed in: ", "Expression sites annotated"),
    ("feat_gnn_interacting_genes_", "Interacts with: ", "Interaction partners annotated"),
    ("feat_gnn_pathways_", "Pathway: ", "Pathways annotated"),
)


def _load_feature_label_map() -> Dict[str, str]:
    return FEATURE_DISPLAY_LABELS


_HPO_ID_NAME_CACHE: Dict[str, str] = {}


def _load_hpo_id_name_map() -> Dict[str, str]:
    """HPO identifier -> term name, read once from the pinned hp.obo."""
    if not _HPO_ID_NAME_CACHE:
        current = None
        with open(Config.HPO_OBO_PATH, encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if line == "[Term]":
                    current = None
                elif line.startswith("id: HP:"):
                    current = line[4:].strip()
                elif line.startswith("name:") and current:
                    _HPO_ID_NAME_CACHE[current] = line[5:].strip()
                    current = None
    return _HPO_ID_NAME_CACHE


def readable_feature_label(name, max_len: int = 48) -> str:
    """Display label for a feature column."""
    name = str(name)
    missing_suffix = ""
    base = name
    if base.endswith("__missing"):
        base = base[: -len("__missing")]
        missing_suffix = " (not available)"
    m = _load_feature_label_map()
    if base in m:
        out = m[base]
    else:
        low = base.lower()
        hpo_id_match = re.match("^(.*?)_?hpo_HP_(\\d{7})$", base)
        text_hash_match = re.match("^(?:feat_)?hpo_(.+?)_([0-9a-f]{6,12})$", base)
        source_prefix = next((pair for pair in _FEATURE_SOURCE_PREFIXES if base.startswith(pair[0])), None)
        if low.startswith("feat_hpo_sys_"):
            out = "System: " + base[len("feat_hpo_sys_") :].replace("_", " ").strip()
        elif low.startswith("feat_mondo_"):
            out = "MONDO: " + base[len("feat_mondo_") :].replace("_", " ").strip()
        elif low.startswith("feat_tier_"):
            out = "Tier: " + base[len("feat_tier_") :].replace("_", " ").strip()
        elif source_prefix is not None:
            prefix, human, count_label = source_prefix
            tail = base[len(prefix) :]
            if tail == "count":
                out = count_label + ", count"
            else:
                tail = re.sub("_[0-9a-f]{6}$", "…", tail)
                out = human + tail.replace("_", " ")
        elif hpo_id_match and hpo_id_match.group(1):
            id_name = _load_hpo_id_name_map()
            term = id_name.get(f"HP:{hpo_id_match.group(2)}", f"HP:{hpo_id_match.group(2)}")
            stem = hpo_id_match.group(1)
            if stem.startswith("mouse_lethal_for"):
                out = f"Mouse-lethal knockout for {term}"
            elif stem.startswith("mouse_subviable_for"):
                out = f"Mouse-subviable knockout for {term}"
            elif stem.startswith("mouse"):
                out = f"Mouse phenotype for {term}"
            else:
                out = f"{stem.replace('_', ' ').capitalize()}: {term}"
        elif low.startswith("hpo_") or low.startswith("feat_hpo_"):
            stripped = re.sub("^(feat_)?hpo_", "", base)
            hp_match = re.match("^HP_(\\d{7})$", stripped)
            id_name = _load_hpo_id_name_map()
            if hp_match and f"HP:{hp_match.group(1)}" in id_name:
                out = id_name[f"HP:{hp_match.group(1)}"]
            elif text_hash_match:
                term_text = text_hash_match.group(1).replace("_", " ").strip()
                out = re.sub("\\s{2,}", ", ", term_text)
                if len(text_hash_match.group(1)) >= 33:
                    out += "…"
            else:
                out = "Phenotype: " + stripped.replace("_", " ").strip()
        elif low.startswith("has_") and low.endswith("_record"):
            out = base[len("has_") : -len("_record")].replace("_", " ") + " data available"
        elif low.startswith("feat_"):
            out = base[len("feat_") :].replace("_", " ").strip()
        else:
            out = base.replace("_", " ").strip()
        out = out[:1].upper() + out[1:] if out else base
    out = out + missing_suffix
    if max_len and len(out) > max_len:
        out = out[: max_len - 1].rstrip() + "…"
    return out


def write_feature_label_dictionaries(container: DataContainer) -> None:
    """Write the display label of every candidate feature."""
    names: List[str] = []
    for columns in container.scenario_cols.values():
        names.extend(columns)
    names.extend(FEATURE_DISPLAY_LABELS)
    names.extend((f"{n}__missing" for n in FEATURE_DISPLAY_LABELS))
    names = sorted(dict.fromkeys((str(n) for n in names)))
    rows = [(n, readable_feature_label(n, max_len=0)) for n in names]
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    for filename, columns in (
        ("feature_label_dictionary_extra.csv", ("feature_name", "label")),
        ("feature_dictionary.csv", ("feature", "readable_label")),
    ):
        path = os.path.join(Config.OUTPUT_DIR, filename)
        pd.DataFrame(rows, columns=list(columns)).to_csv(path, index=False)
    logger.log(
        f"  Wrote display labels for {len(rows)} features to feature_dictionary.csv and feature_label_dictionary_extra.csv"
    )
