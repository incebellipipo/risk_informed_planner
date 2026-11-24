# Notebook guide

Use the `risk_aware_planner3.12` kernel. Run each notebook with its own directory as the kernel working directory. Maps remain in `maps/`.

For the main workflow, run `experiments/kristiansund_bridge_planning.ipynb`, then its `kristiansund_bridge_figures.ipynb` and `kristiansund_bridge_threshold_figures.ipynb` companions.

Figures load `results/study_03.pkl`; campaign figures load `results/campaign.pkl` after running `kristiansund_bridge_replanning.ipynb`. Saved-data filenames are unchanged.

The studies preserve the original grounding evidence and sampling policies.
`DEPTH_SAMPLING_OCCUPANCY_CHECKS` explicitly reproduces the original notebooks'
depth-based sampling with occupancy edge checks. For reproducible comparisons,
use the same NumPy random seed, inputs, and Python environment in both packages.
Regenerate results and figures produced before this behavior was restored.

## experiments

| Notebook | Previous filename |
| --- | --- |
| [trondheim_parameter_sweep.ipynb](experiments/trondheim_parameter_sweep.ipynb) | `experiment_matrix_trondheim_01.ipynb` |
| [kristiansund_bridge_replanning.ipynb](experiments/kristiansund_bridge_replanning.ipynb) | `plotting_campaign.ipynb` |
| [open_water_replanning.ipynb](experiments/open_water_replanning.ipynb) | `plotting_campaign2.ipynb` |
| [replanning_campaign_figures.ipynb](experiments/replanning_campaign_figures.ipynb) | `plotting_campaign_plots.ipynb` |
| [tree_growth_figures.ipynb](experiments/tree_growth_figures.ipynb) | `plotting_different_steps.ipynb` |
| [open_water_planning.ipynb](experiments/open_water_planning.ipynb) | `study_01.ipynb` |
| [kristiansund_port_entry_planning.ipynb](experiments/kristiansund_port_entry_planning.ipynb) | `study_02.ipynb` |
| [kristiansund_bridge_planning.ipynb](experiments/kristiansund_bridge_planning.ipynb) | `study_03.ipynb` |
| [kristiansund_bridge_figures.ipynb](experiments/kristiansund_bridge_figures.ipynb) | `study_03_plots.ipynb` |
| [kristiansund_bridge_threshold_figures.ipynb](experiments/kristiansund_bridge_threshold_figures.ipynb) | `study_03_plotting_with_treshold.ipynb` |
| [trondheim_fjord_planning.ipynb](experiments/trondheim_fjord_planning.ipynb) | `study_04.ipynb` |
| [interpolate_risk_surface.ipynb](experiments/interpolate_risk_surface.ipynb) | `study_build_map_01.ipynb` |

## statistics

| Notebook | Previous filename |
| --- | --- |
| [risk_evaluation_counts.ipynb](statistics/risk_evaluation_counts.ipynb) | `BBN_query_count.ipynb` |
| [risk_inference_timing.ipynb](statistics/risk_inference_timing.ipynb) | `experiment_compute_average_risk_compute_time.ipynb` |
| [neighborhood_crowding.ipynb](statistics/neighborhood_crowding.ipynb) | `experiment_compute_average_risk_compute_time_2.ipynb` |
| [cost_evaluation_counts.ipynb](statistics/cost_evaluation_counts.ipynb) | `experiment_cost_count.ipynb` |
| [cost_evaluation_counts_neighbor_limit.ipynb](statistics/cost_evaluation_counts_neighbor_limit.ipynb) | `experiment_cost_count_max_neighbor.ipynb` |
| [risk_inference_and_crowding.ipynb](statistics/risk_inference_and_crowding.ipynb) | `experiment_risk_compute_time_and_crowding.ipynb` |
| [risk_inference_and_crowding_figures.ipynb](statistics/risk_inference_and_crowding_figures.ipynb) | `experiment_risk_compute_time_and_crowding_plots.ipynb` |
