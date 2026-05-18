"""Stimulus leakage guard tests (extends test_leakage.py)."""

import pytest

from unimarket.utils.validation import CircularityError, assert_no_forbidden_stimulus_features


def test_one_hot_stimulus_blocked():
    with pytest.raises(CircularityError):
        assert_no_forbidden_stimulus_features(
            ["stimulus_onehot"],
            allow_stimulus_identity=False,
            allow_global_time_features=False,
            allow_trial_index=False,
            allow_block_index=False,
        )


def test_absolute_timestamp_blocked():
    with pytest.raises(CircularityError):
        assert_no_forbidden_stimulus_features(
            ["absolute_timestamp"],
            allow_stimulus_identity=False,
            allow_global_time_features=False,
            allow_trial_index=False,
            allow_block_index=False,
        )


def test_local_elapsed_time_allowed():
    # 'local_elapsed_time' is NOT in the forbidden set and should pass.
    assert_no_forbidden_stimulus_features(
        ["history_voltage", "history_current", "local_elapsed_time"],
        allow_stimulus_identity=False,
        allow_global_time_features=False,
        allow_trial_index=False,
        allow_block_index=False,
    )


def test_explicitly_allowed_stimulus_passes():
    assert_no_forbidden_stimulus_features(
        ["stimulus_id", "stimulus_label"],
        allow_stimulus_identity=True,
        allow_global_time_features=False,
        allow_trial_index=False,
        allow_block_index=False,
    )
