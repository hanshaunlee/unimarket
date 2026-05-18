"""Pydantic schemas for UniMarket datasets."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class CellRecord(_Base):
    dataset_name: str
    cell_id: str
    cell_type: Optional[str] = None
    brain_area: Optional[str] = None
    species: Optional[str] = None
    subject_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SweepRecord(_Base):
    sweep_id: str
    cell_id: str
    protocol_name: Optional[str] = None
    stimulus_path: str
    response_path: str
    sampling_rate_hz: float
    start_time_s: float = 0.0
    duration_s: float
    split: str = "train"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SpikeUnitRecord(_Base):
    unit_id: str
    session_id: str
    probe_id: Optional[str] = None
    brain_area: Optional[str] = None
    spike_times_path: str
    quality_metrics: dict[str, Any] = Field(default_factory=dict)
    split: str = "train"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SynapsePairRecord(_Base):
    pair_id: str
    pre_cell_id: str
    post_cell_id: str
    connection_label: Optional[str] = None
    stimulus_path: str
    response_path: str
    sampling_rate_hz: float
    split: str = "train"
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphRecord(_Base):
    graph_id: str
    node_ids: list[str]
    edge_index_path: str
    edge_attr_path: Optional[str] = None
    construction_method: str  # 'synthetic', 'random', 'distance', 'train_correlation', 'learned'
    allowed_for_training: bool = True
    allowed_for_eval: bool = False
    notes: str = ""
