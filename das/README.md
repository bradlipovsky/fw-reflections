# DAS postprocessing

This directory contains lightweight Python utilities for turning the
grounding-line simulation output into synthetic DAS-style products.

## Files

- [`io_utils.py`](io_utils.py): load SPECFEM `.semv` files or compact `.npz` gathers
- [`synthetic.py`](synthetic.py): DAS synthesis and FK filtering helpers
- [`reflection.py`](reflection.py): signed reflection-coefficient utilities
- [`make_synthetic_das.py`](make_synthetic_das.py): command-line DAS product generator
- [`groundingline_das_workflow.ipynb`](groundingline_das_workflow.ipynb): main DAS workflow notebook
- [`groundingline_reflection_coefficient.ipynb`](groundingline_reflection_coefficient.ipynb): baseline reflection-coefficient notebook

## Assumption used for the first-pass DAS observable

For a straight horizontal fiber, the workflow uses the standard approximation

`strain_rate(x,t) ~= [v_x(x + L/2, t) - v_x(x - L/2, t)] / L`

where `L` is the gauge length.

This is a starter approximation for comparing incident and reflected
surface-wave energy, not a full instrument model.

## Quick start

From the project root, after a baseline SPECFEM2D run:

```bash
cd das
python3 make_synthetic_das.py \
  --input ../OUTPUT_FILES \
  --station-prefix S \
  --gauge-length 6.28 \
  --channel-spacing 6.28 \
  --output products/surface_das.npz
```

Then open the notebooks and run them top to bottom.
