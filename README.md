# Grounding-Line Starter Example

This example builds a 2D P-SV model for Rayleigh-like wave propagation in a uniform ice layer over two laterally joined half spaces:

- ice for the full model width from `z = 1200 m` to `z = 1500 m`
- water below the ice for `x < 0`
- rock below the ice for `x > 0`

## Meshing choice

This example uses the **internal interface-based mesher**, not an external mesh workflow.

That is the cleanest route here because the geometry is still rectilinear:

- the ice base is a flat horizontal interface
- the grounding line is a vertical material split at `x = 0`
- SPECFEM2D's internal region table can assign different materials to rectangular blocks in `(nx, nz)` index space

By choosing `xmin = -12000 m`, `xmax = 6000 m`, and `nx = 360`, the `x = 0` transition falls exactly on an element edge, which keeps the setup simple and robust while placing the source about 10 km to the left of the grounding line and giving transmitted energy a larger rock-side buffer before it reaches the right absorbing boundary.

## Key model parameters

- Total width: `18000 m`
- Total depth: `1500 m`
- Ice thickness `H`: `300 m`
- Grounding-line position: `x = 0`
- Source: `x = -10000 m` on the free surface (`source_surf = .true.`)
- Receiver span: `x = -3500 m` to `x = -200 m`
- Receiver spacing: `50 m`
- Dominant source frequency: `10 Hz`
- Total simulation time: `10.8 s`

Material placeholders are defined near the middle of [`DATA/Par_file`](/Users/bradlipovsky/specfem2d-master/WORK/groundingline/DATA/Par_file):

- Water: `rho = 1000 kg/m^3`, `Vp = 1500 m/s`
- Rock: `rho = 2700 kg/m^3`, `Vp = 4000 m/s`, `Vs = 2300 m/s`
- Ice: `rho = 917 kg/m^3`, `Vp = 3800 m/s`, `Vs = 1900 m/s`

Receiver parameters live in [`DATA/make_stations.py`](/Users/bradlipovsky/specfem2d-master/WORK/groundingline/DATA/make_stations.py), so you can quickly tighten spacing later for DAS-style channel studies.

## Files

- [`DATA/Par_file`](/Users/bradlipovsky/specfem2d-master/WORK/groundingline/DATA/Par_file): main simulation and mesh setup
- [`DATA/SOURCE`](/Users/bradlipovsky/specfem2d-master/WORK/groundingline/DATA/SOURCE): impulsive opening-mode moment tensor source on the ice free surface
- [`DATA/interfaces_groundingline.dat`](/Users/bradlipovsky/specfem2d-master/WORK/groundingline/DATA/interfaces_groundingline.dat): flat interfaces for bottom, ice base, and free surface
- [`DATA/make_stations.py`](/Users/bradlipovsky/specfem2d-master/WORK/groundingline/DATA/make_stations.py): receiver-array generator
- [`DATA/STATIONS`](/Users/bradlipovsky/specfem2d-master/WORK/groundingline/DATA/STATIONS): generated receiver list
- [`run_this_example.sh`](/Users/bradlipovsky/specfem2d-master/WORK/groundingline/run_this_example.sh): convenience runner

## How to run

From `/Users/bradlipovsky/specfem2d-master`:

```bash
cd WORK/groundingline
python3 DATA/make_stations.py
chmod +x run_this_example.sh
./run_this_example.sh
```

Or run the executables manually after generating `DATA/STATIONS`:

```bash
cd /Users/bradlipovsky/specfem2d-master/WORK/groundingline
python3 DATA/make_stations.py
./bin/xmeshfem2D
./bin/xspecfem2D
```

## What to expect

Qualitatively, you should see:

- a strong Rayleigh-like wave packet traveling rightward along the ice free surface
- transmitted energy continuing across the `x = 0` transition
- a reflected surface-guided phase returning into the left-side ice
- mode conversion and body-wave energy radiated into the ice and substrate near the grounding line

## Sanity checks

- Confirm the mesh snapshot shows a flat 300 m ice layer over a left-water/right-rock substrate.
- Check that the source plots inside the ice, not on the free surface.
- Verify the early direct surface-wave train moves rightward first.
- Look for the reflected Rayleigh-like arrival on the left-side stations after the direct arrival and before any obvious left-boundary contamination.

With the current geometry, a good first place to inspect reflected energy is the left receiver line around `x = -2500 m` to `x = -1000 m`.

## Limitations

- This version is intentionally elongated so the grounding line sees a more mature incoming packet before scattering.
- This is still a starter model with flat interfaces and simple isotropic properties.
- The source is a simple opening-mode moment tensor approximation, not a full fracture-dynamics model.
- The `50 m` station spacing is intentionally coarse for a first pass; DAS-style work will likely need denser spacing later.
- PMLs help, but if you extend the simulation time much further you may still want a wider left-side buffer before measuring small reflection coefficients.
