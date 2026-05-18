# Dataset limitations

UniMarket distinguishes four adapter statuses, recorded in the `adapter_status` field of each dataset's `preprocessing_report.json`:

| Status | Meaning |
| --- | --- |
| `fully_functional` | The full pipeline works locally end-to-end with no external dependencies. (Synthetic LIF circuit.) |
| `functional_with_local_nwb` | Works if the user provides a local `.nwb` file (or directory). Uses `pynwb`. No download is performed. |
| `schema_validated_only` | A user-provided manifest CSV/Parquet is validated against a documented schema and normalized into UniMarket's representation. Full raw ingestion from the upstream format is not implemented. |
| `planned` | Reserved; not implemented. |

| Adapter | Status | What you must supply |
| --- | --- | --- |
| `synthetic` | `fully_functional` | nothing |
| `nwb_generic` | `functional_with_local_nwb` | local `.nwb` path; `pip install -e .[nwb]` |
| `allen_cell_types` | `functional_with_local_nwb` | directory of `.nwb` files; `pip install -e .[nwb]` |
| `allen_synphys` | `schema_validated_only` | CSV manifest with columns: `pair_id, pre_cell_id, post_cell_id, stimulus_path, response_path, sampling_rate_hz [, connection_label]` |
| `allen_neuropixels` | `schema_validated_only` | spike table (CSV/Parquet) with columns: `unit_id, session_id, spike_time_s [, brain_area]` |

## Why this matters for claims

- Synthetic experiments can support mechanism recovery claims because ground truth is known.
- Real Allen / DANDI training primarily supports predictive generalization unless independent metadata is held out and used only post hoc.
- Mechanism recovery on real data requires:
  - the predictive model did not train on the relevant metadata
  - the split is appropriate
  - the latent survives random-latent and label-shuffle controls
  - the claim gate explicitly allows it

If your study requires real-biology mechanism claims, plan a held-out metadata setup before you train.
