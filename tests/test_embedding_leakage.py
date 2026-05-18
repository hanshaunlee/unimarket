"""Identity-embedding leakage guard tests."""

import pytest

from unimarket.utils.validation import CircularityError, assert_no_forbidden_id_embeddings


def test_pair_id_blocked_by_default():
    with pytest.raises(CircularityError):
        assert_no_forbidden_id_embeddings(
            {"pair_id_embedding": True},
            allow_cell_id_embedding=False,
            allow_pair_id_embedding=False,
            allow_session_id_embedding=False,
        )


def test_session_id_blocked_by_default():
    with pytest.raises(CircularityError):
        assert_no_forbidden_id_embeddings(
            {"session_id_embedding": True},
            allow_cell_id_embedding=False,
            allow_pair_id_embedding=False,
            allow_session_id_embedding=False,
        )


def test_all_allowed_passes_within_split_when_not_heldout():
    # Allowed flags + non-heldout split should pass.
    assert_no_forbidden_id_embeddings(
        {"cell_id_embedding": True},
        allow_cell_id_embedding=True,
        allow_pair_id_embedding=False,
        allow_session_id_embedding=False,
        split_mode="seen_cell_heldout_sweep",
    )
