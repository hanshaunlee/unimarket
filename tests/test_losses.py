"""Loss scalar tests."""

import torch

from unimarket.models.losses import (
    derivative_loss,
    graph_sparsity_loss,
    poisson_nll,
    rollout_consistency_loss,
    smoothness_loss,
    spike_bce,
    voltage_mse,
)


def test_voltage_mse_scalar():
    a = torch.randn(4, 5)
    b = torch.randn(4, 5)
    out = voltage_mse(a, b)
    assert out.shape == ()
    assert out.item() >= 0.0


def test_spike_bce_scalar():
    logits = torch.randn(4, 5)
    target = torch.randint(0, 2, (4, 5)).float()
    out = spike_bce(logits, target)
    assert out.shape == ()
    assert out.item() >= 0.0


def test_poisson_nll_positive():
    rate = torch.rand(4, 5) + 0.1
    counts = torch.poisson(rate)
    out = poisson_nll(rate, counts)
    assert out.shape == ()


def test_rollout_consistency_loss_scalar():
    a = torch.randn(4, 5, 3)
    b = torch.randn(4, 5, 3)
    out = rollout_consistency_loss(a, b)
    assert out.shape == ()
    assert out.item() >= 0.0


def test_graph_sparsity_loss_in_unit_range():
    logits = torch.randn(8, 8)
    out = graph_sparsity_loss(logits)
    assert 0.0 <= out.item() <= 1.0


def test_smoothness_loss_zero_for_constant_traj():
    pred = torch.zeros(2, 6)
    out = smoothness_loss(pred)
    assert out.item() == 0.0


def test_derivative_loss_scalar():
    a = torch.randn(2, 6)
    b = torch.randn(2, 6)
    out = derivative_loss(a, b)
    assert out.shape == ()
