"""Split correctness tests."""

import pytest

from unimarket.data.splits import (
    cell_protocol_split,
    split_by_cell,
    split_by_pair,
    split_by_protocol,
    split_by_session,
    split_by_time_block,
)
from unimarket.utils.validation import (
    CircularityError,
    assert_no_overlap,
    assert_time_blocks_disjoint,
    assert_window_random_split_allowed,
)


def test_split_by_cell_no_overlap():
    cells = [f"c{i}" for i in range(20)]
    splits = split_by_cell(cells, seed=42)
    assert_no_overlap("cell_id", splits["train"], splits["val"], splits["test"])
    assert set(splits["train"]) | set(splits["val"]) | set(splits["test"]) == set(cells)


def test_split_by_session_no_overlap():
    splits = split_by_session([f"s{i}" for i in range(15)], seed=0)
    assert_no_overlap("session_id", splits["train"], splits["val"], splits["test"])


def test_split_by_pair_no_overlap():
    splits = split_by_pair([f"p{i}" for i in range(12)], seed=0)
    assert_no_overlap("pair_id", splits["train"], splits["val"], splits["test"])


def test_split_by_protocol_explicit():
    splits = split_by_protocol(
        ["A", "B", "C", "D", "E"], test_protocol_families=["E"], val_protocol_families=["D"]
    )
    assert "E" in splits["test"]
    assert "D" in splits["val"]
    assert "E" not in splits["train"]


def test_split_by_time_block_disjoint():
    sp = split_by_time_block(T=100, fracs=(0.7, 0.15, 0.15))
    assert sp["train"][0][1] == sp["val"][0][0]
    assert sp["val"][0][1] == sp["test"][0][0]
    assert_time_blocks_disjoint(sp)


def test_overlapping_blocks_caught():
    with pytest.raises(CircularityError):
        assert_time_blocks_disjoint(
            {"train": [(0, 50)], "val": [(40, 60)], "test": [(60, 80)]}
        )


def test_overlapping_cells_caught():
    with pytest.raises(CircularityError):
        assert_no_overlap("cell_id", ["a", "b"], ["b", "c"], ["d"])


def test_window_random_split_guard():
    with pytest.raises(CircularityError):
        assert_window_random_split_allowed("window_random", allow_leaky_split=False)
    # Must not raise when allowed.
    assert_window_random_split_allowed("window_random", allow_leaky_split=True)


def test_cell_protocol_heldout_mode():
    pairs = [("c1", "A"), ("c1", "B"), ("c2", "A"), ("c2", "B"), ("c3", "A"), ("c4", "B")]
    out = cell_protocol_split(pairs, mode="heldout_cell_seen_protocol_family", seed=0)
    train_cells = {c for c, _ in out["train"]}
    test_cells = {c for c, _ in out["test"]}
    val_cells = {c for c, _ in out["val"]}
    assert not (train_cells & test_cells)
    assert not (train_cells & val_cells)
