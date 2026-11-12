"""The primary split of the published run (Online Methods, equation 1)."""

import pandas as pd
from conftest import ROOT

SPLIT = ROOT / "published_run" / "benchmark" / "results" / "split_identifiers_primary.csv"


def test_partition_sizes():
    split = pd.read_csv(SPLIT)
    assert split["partition"].value_counts().to_dict() == {"train": 4306, "test": 1325, "validation": 994}


def test_each_pair_in_one_partition():
    split = pd.read_csv(SPLIT)
    assert not split.duplicated(["gene_symbol", "mondo_id"]).any()
    assert split["row_index"].is_unique


def test_partitions_share_no_gene():
    split = pd.read_csv(SPLIT)
    genes = {name: set(group["gene_symbol"]) for name, group in split.groupby("partition")}
    assert not genes["train"] & genes["validation"]
    assert not genes["train"] & genes["test"]
    assert not genes["validation"] & genes["test"]
