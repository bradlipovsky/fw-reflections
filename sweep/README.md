# Sweep workflows

This directory contains the compact parameter-sweep drivers and comparison
notebooks for the grounding-line reflection project.

The sweep scripts save lightweight summaries and compact gathers rather than
full `OUTPUT_FILES/` directories. Those generated result directories are
ignored by git and can be recreated as needed.

## Main drivers

- [`run_grounded_material_sweep.py`](run_grounded_material_sweep.py):
  grounded-substrate material sweep, with frequency-specific output
  directories such as `results_10hz`, `results_5hz`, and `results_1hz`
- [`run_supported_cavity_sweep.py`](run_supported_cavity_sweep.py):
  reduced material sweep for the `100 m` supported-cavity geometry
- [`run_ice_thickness_sweep.py`](run_ice_thickness_sweep.py):
  focused `10 Hz` ice-thickness sweep
- [`run_groundingline_position_sweep.py`](run_groundingline_position_sweep.py):
  small grounding-line offset sweep
- [`run_groundingline_position_large_sweep.py`](run_groundingline_position_large_sweep.py):
  larger grounding-line offset sweep with matched-filter recovery
- [`run_sloping_seafloor_comparison.py`](run_sloping_seafloor_comparison.py):
  flat-cavity versus near-pinch-out sloping-seafloor comparison

## Main notebooks

- [`grounded_material_sweep_results.ipynb`](grounded_material_sweep_results.ipynb)
- [`supported_cavity_comparison.ipynb`](supported_cavity_comparison.ipynb)
- [`ice_thickness_sweep_results.ipynb`](ice_thickness_sweep_results.ipynb)
- [`groundingline_position_sweep_results.ipynb`](groundingline_position_sweep_results.ipynb)
- [`groundingline_position_large_offsets_matched_filter.ipynb`](groundingline_position_large_offsets_matched_filter.ipynb)
- [`sloping_seafloor_comparison.ipynb`](sloping_seafloor_comparison.ipynb)

## Storage strategy

To avoid committing large files:

- full `OUTPUT_FILES/` directories are not archived
- compact `surface_gather.npz` files are regenerated when needed
- only the scripts and notebooks are versioned here

## Typical usage

From the project root:

```bash
cd sweep
python3 run_grounded_material_sweep.py --frequency-hz 10
python3 run_supported_cavity_sweep.py
python3 run_ice_thickness_sweep.py
python3 run_groundingline_position_large_sweep.py
python3 run_sloping_seafloor_comparison.py
```

Then open the corresponding notebook.
