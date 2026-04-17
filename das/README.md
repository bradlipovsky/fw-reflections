# DAS Postprocessing

This directory contains lightweight Python utilities for turning the grounding-line example output into a synthetic distributed acoustic sensing product.

## What is here

- [`io_utils.py`](/Users/bradlipovsky/specfem2d-master/WORK/groundingline/das/io_utils.py): load SPECFEM `.semv` files or simple `spec2npy`-style `.npz/.npy` gathers
- [`synthetic.py`](/Users/bradlipovsky/specfem2d-master/WORK/groundingline/das/synthetic.py): project particle velocity onto a fiber direction and form gauge-length DAS strain-rate records
- [`make_synthetic_das.py`](/Users/bradlipovsky/specfem2d-master/WORK/groundingline/das/make_synthetic_das.py): command-line helper that writes a DAS `.npz` product
- [`groundingline_das_workflow.ipynb`](/Users/bradlipovsky/specfem2d-master/WORK/groundingline/das/groundingline_das_workflow.ipynb): notebook version of the same workflow

## Assumptions

For a straight horizontal fiber, the notebook uses the usual first-pass approximation

`strain_rate(x,t) ~= [v_x(x + L/2, t) - v_x(x - L/2, t)] / L`

where `L` is the gauge length.

That is a starter approximation, not a full instrument model. It is useful for comparing incident and reflected surface-wave energy and for testing the effect of gauge length and channel spacing.

## Quick start

From [`/Users/bradlipovsky/specfem2d-master/WORK/groundingline/das`](/Users/bradlipovsky/specfem2d-master/WORK/groundingline/das):

```bash
python3 make_synthetic_das.py \
  --input ../OUTPUT_FILES \
  --station-prefix S \
  --gauge-length 50 \
  --channel-spacing 25 \
  --output products/surface_das.npz
```

Then open the notebook and run it from top to bottom.
