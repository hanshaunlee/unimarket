"""Forward-shape tests for models."""

import torch

from unimarket.models.adaptive_lif import AdaptiveLIFResidualModel
from unimarket.models.baselines import (
    ARVoltage,
    GLMPoissonPopulation,
    MeanRatePopulation,
    PersistenceVoltage,
    RidgeVoltage,
)
from unimarket.models.graph_dynamics import GraphLatentDynamicsModel
from unimarket.models.lif import LIFResidualModel
from unimarket.models.neural_ode import NeuralODECell
from unimarket.models.spike_transformer import SpikeTransformerBaseline
from unimarket.models.synapse_model import SynapseResponseModel


def _voltage_batch(B=4, T=20, H=5):
    return {
        "history_voltage": torch.randn(B, T),
        "history_current": torch.randn(B, T),
        "target_voltage": torch.randn(B, H),
        "target_spikes": torch.zeros(B, H),
        "neuron_idx": torch.zeros(B, dtype=torch.long),
    }


def test_lif_residual_forward_shape():
    m = LIFResidualModel(history_len=20, horizon=5, learn_per_neuron_params=True, n_neurons=4)
    batch = _voltage_batch()
    out = m(batch)
    assert out["pred"].shape == (4, 5)
    assert out["pred_spike_logits"].shape == (4, 5)
    # Constrained parameters are positive after softplus.
    assert float(m.tau_m(torch.tensor([0]))) > 0
    assert float(m.R_m(torch.tensor([0]))) > 0
    # Threshold > reset.
    assert float(m.threshold(torch.tensor([0]))) > m.reset_voltage


def test_adaptive_lif_forward_shape():
    m = AdaptiveLIFResidualModel(history_len=20, horizon=5, learn_per_neuron_params=False)
    out = m(_voltage_batch())
    assert out["pred"].shape == (4, 5)
    assert "tau_w_s" in out["aux"]


def test_neural_ode_forward_shape():
    m = NeuralODECell(history_len=20, horizon=5)
    out = m(_voltage_batch())
    assert out["pred"].shape == (4, 5)


def test_baselines_shapes():
    batch = _voltage_batch()
    for m in (
        PersistenceVoltage(horizon=5),
        ARVoltage(p=4, horizon=5),
        RidgeVoltage(history_len=20, horizon=5),
    ):
        out = m(batch)
        assert out["pred"].shape == (4, 5)


def test_graph_dynamics_forward_shape():
    B, T, N, H = 2, 6, 5, 3
    m = GraphLatentDynamicsModel(
        n_neurons=N, history_len=T, horizon=H, hidden_dim=16, graph_mode="learned"
    )
    batch = {
        "history_counts": torch.rand(B, T, N),
        "future_counts": torch.rand(B, H, N),
    }
    out = m(batch)
    assert out["pred"].shape == (B, H, N)
    assert out["pred"].min().item() >= 0  # softplus positive rates


def test_spike_transformer_forward_shape():
    B, T, N, H = 2, 6, 5, 3
    m = SpikeTransformerBaseline(n_neurons=N, history_len=T, horizon=H, hidden_dim=16, n_layers=1)
    out = m({"history_counts": torch.rand(B, T, N), "future_counts": torch.rand(B, H, N)})
    assert out["pred"].shape == (B, H, N)
    assert out["pred"].min().item() >= 0


def test_population_baselines_shapes():
    B, T, N, H = 2, 6, 4, 3
    batch = {"history_counts": torch.rand(B, T, N), "future_counts": torch.rand(B, H, N)}
    for m in (
        MeanRatePopulation(n_neurons=N, horizon=H),
        GLMPoissonPopulation(n_neurons=N, history_len=T, horizon=H),
    ):
        out = m(batch)
        assert out["pred"].shape == (B, H, N)


def test_synapse_response_forward_shape():
    m = SynapseResponseModel(max_delay_steps=10, hidden_dim=16)
    B, T = 3, 20
    out = m({"pre_event": torch.rand(B, T), "target_response": torch.zeros(B, T)})
    assert out["pred"].shape == (B, T)
    assert m.rise_tau().item() > 0
    assert m.decay_tau().item() > 0
