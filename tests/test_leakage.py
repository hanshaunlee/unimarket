"""Leakage and circularity guard tests."""

import pytest

from unimarket.utils.validation import (
    CircularityError,
    assert_checkpoint_not_test,
    assert_no_forbidden_id_embeddings,
    assert_no_forbidden_stimulus_features,
    assert_normalization_from_train_only,
)


def test_normalization_must_be_train_only():
    with pytest.raises(CircularityError):
        assert_normalization_from_train_only({"computed_on_splits": ["train", "val"]})
    with pytest.raises(CircularityError):
        assert_normalization_from_train_only({"computed_on_splits": ["test"]})
    assert_normalization_from_train_only({"computed_on_splits": ["train"]})


def test_test_split_selection_blocked():
    with pytest.raises(CircularityError):
        assert_checkpoint_not_test("test")
    assert_checkpoint_not_test("val")


def test_forbidden_stimulus_features_blocked():
    with pytest.raises(CircularityError):
        assert_no_forbidden_stimulus_features(
            ["stimulus_id"],
            allow_stimulus_identity=False,
            allow_global_time_features=False,
            allow_trial_index=False,
            allow_block_index=False,
        )
    # Allowed when explicit.
    assert_no_forbidden_stimulus_features(
        ["stimulus_id"],
        allow_stimulus_identity=True,
        allow_global_time_features=False,
        allow_trial_index=False,
        allow_block_index=False,
    )


def test_global_time_features_blocked():
    with pytest.raises(CircularityError):
        assert_no_forbidden_stimulus_features(
            ["global_time"],
            allow_stimulus_identity=False,
            allow_global_time_features=False,
            allow_trial_index=False,
            allow_block_index=False,
        )


def test_trial_block_index_blocked():
    with pytest.raises(CircularityError):
        assert_no_forbidden_stimulus_features(
            ["trial_index"],
            allow_stimulus_identity=False,
            allow_global_time_features=False,
            allow_trial_index=False,
            allow_block_index=False,
        )
    with pytest.raises(CircularityError):
        assert_no_forbidden_stimulus_features(
            ["block_index"],
            allow_stimulus_identity=False,
            allow_global_time_features=False,
            allow_trial_index=False,
            allow_block_index=False,
        )


def test_id_embeddings_blocked_by_default():
    with pytest.raises(CircularityError):
        assert_no_forbidden_id_embeddings(
            {"cell_id_embedding": True},
            allow_cell_id_embedding=False,
            allow_pair_id_embedding=False,
            allow_session_id_embedding=False,
        )
    # Even if allow flag is set, held-out-cell split must not use cell embedding.
    with pytest.raises(CircularityError):
        assert_no_forbidden_id_embeddings(
            {"cell_id_embedding": True},
            allow_cell_id_embedding=True,
            allow_pair_id_embedding=False,
            allow_session_id_embedding=False,
            split_mode="heldout_cell_seen_protocol_family",
        )
